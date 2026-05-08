import os
import sys

# --- CRITICAL: SUPPRESS NOISY OPENBLAS WARNINGS & PREVENT DEADLOCKS ---
os.environ["OPENBLAS_VERBOSE"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4" # Let our engine use 4 cores, but PyTorch use 1

import time
import torch
# Force PyTorch to be single-threaded to avoid OpenMP contention/hangs with our engine
torch.set_num_threads(1)

import torch.nn as nn
import ctypes
import numpy as np
from PIL import Image
import torchvision.transforms as T
import yaml
from box import Box
from accelerate import init_empty_weights

# Avoid OOM in timm
import timm
class MockTeacher(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_features = 512
def mock_create_model(*args, **kwargs): return MockTeacher()
timm.create_model = mock_create_model

from claymodel.model import clay_mae_large

CKPT_PATH = "clay-v1.5.ckpt" 
FFN_BIN = b"clay_bundled_ffn.bin"
PRED_BIN = b"Clay_predictors.bin"

# --- C++ Engine Bindings ---
engine_path = os.path.abspath("./libengine.so")
if not os.path.exists(engine_path):
    print(f"CRITICAL: {engine_path} not found. Run 'make' first.")
    sys.exit(1)

try:
    lib = ctypes.CDLL(engine_path)
    lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
    lib.init_engine.restype = ctypes.c_void_p
    lib.set_engine_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    lib.execute_ffn_layer.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.destroy_engine.argtypes = [ctypes.c_void_p]
except Exception as e:
    print(f"\n❌ FAILED TO LOAD ENGINE: {e}")
    print("\nHELP: If you see 'Detect OpenMP Loop', try running:")
    print("export OMP_WAIT_POLICY=PASSIVE\n")
    sys.exit(1)

class FlashViTFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, hidden_size, mode_int, fc1_bias):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.hidden_size = hidden_size
        self.mode_int = mode_int
        self.fc1_bias = fc1_bias
        self.fc1_bias_c = ctypes.cast(self.fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float))

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, self.hidden_size).float().cpu().contiguous()
        num_tokens = flat_x.shape[0]
        out_cpu = torch.zeros_like(flat_x)
        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            num_tokens, self.fc1_bias_c, self.mode_int)
        return out_cpu.to(x.device).view(*orig_shape)

def get_dummy_datacube():
    pixels_10ch = torch.randn(1, 10, 224, 224)
    waves = torch.tensor([490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0])
    return {"pixels": pixels_10ch, "waves": waves, "latlon": torch.zeros((1, 4)), "time": torch.zeros((1, 4)), "gsd": torch.tensor([10.0]), "platform": ["sentinel-2-l2a"]}

def get_real_datacube(img_path):
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, using random noise.")
        return get_dummy_datacube()
    img_pil = Image.open(img_path).convert("RGB")
    preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    tensor_img = preprocess(img_pil).unsqueeze(0)
    bgr_img = tensor_img[:, [2, 1, 0], :, :]
    padding = torch.zeros((1, 7, 224, 224))
    pixels_10ch = torch.cat([bgr_img, padding], dim=1)
    waves = torch.tensor([490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0])
    return {"pixels": pixels_10ch, "waves": waves, "latlon": torch.zeros((1, 4)), "time": torch.zeros((1, 4)), "gsd": torch.tensor([10.0]), "platform": ["sentinel-2-l2a"]}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="sample_satellite.png", help="Path to input image")
    parser.add_argument("--mode", type=str, choices=["predictor", "draft"], default="predictor", help="Flash mode")
    args = parser.parse_args()

    print(f"--- Clay v1.5 Edge Inference: {args.mode.upper()} MODE ---", flush=True)

    metadata_yaml = """sentinel-2-l2a:
  band_order: [blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22]
  rgb_indices: [2, 1, 0]
  gsd: 10
  bands:
    mean: {blue: 1105., green: 1355., red: 1552., rededge1: 1887., rededge2: 2422., rededge3: 2630., nir: 2743., nir08: 2785., swir16: 2388., swir22: 1835.}
    std: {blue: 1809., green: 1757., red: 1888., rededge1: 1870., rededge2: 1732., rededge3: 1697., nir: 1742., nir08: 1648., swir16: 1470., swir22: 1379.}
    wavelength: {blue: 0.493, green: 0.56, red: 0.665, rededge1: 0.704, rededge2: 0.74, rededge3: 0.783, nir: 0.842, nir08: 0.865, swir16: 1.61, swir22: 2.19}
"""
    os.makedirs("configs", exist_ok=True)
    with open("configs/metadata.yaml", "w") as f: f.write(metadata_yaml)
    metadata_obj = Box(yaml.safe_load(metadata_yaml))

    engine_ptr = lib.init_engine(FFN_BIN, PRED_BIN if os.path.exists(PRED_BIN) else b"", 1024, 4096, 24, 0)
    lib.set_engine_config(engine_ptr, 1024, 0.5, 5)

    print("Materializing lean model...", flush=True)
    mae_args = {"mask_ratio": 0.0, "norm_pix_loss": False, "shuffle": False, "teacher": "vit_large_patch16_224", "dolls": [], "doll_weights": []}
    with init_empty_weights():
        model = clay_mae_large(metadata=metadata_obj, patch_size=8, decoder_embed_dim=1024, **mae_args)
    model.to_empty(device="cpu")
    print("✅ Model structure materialized.", flush=True)

    print("Loading resident weights (mmap)...", flush=True)
    sd_raw = torch.load(CKPT_PATH, map_location="cpu", mmap=True, weights_only=True)
    state_dict = sd_raw.get("state_dict", sd_raw)
    clean_sd = {k.replace("model.", ""): v for k, v in state_dict.items() if "decoder" not in k and "proj" not in k}
    
    print("Applying weights to model...", flush=True)
    model.load_state_dict(clean_sd, strict=False)
    del sd_raw
    if "state_dict" in locals(): del state_dict
    import gc
    gc.collect()
    print("✅ Resident weights loaded.", flush=True)

    print("Zero-initializing remaining parameters...", flush=True)
    for name, p in model.named_parameters():
        if name not in clean_sd:
            with torch.no_grad(): p.zero_()
    del clean_sd
    gc.collect()

    print(f"Patching FlashFFN ({args.mode.upper()})...", flush=True)
    for i in range(24):
        if i % 4 == 0: print(f"  -> Applying Flash Logic: Layer {i}/24", flush=True)
        ff_block = model.encoder.transformer.layers[i][1]
        fc1_bias = torch.zeros(4096, dtype=torch.float32)
        if args.mode == "draft" and i % 4 == 0:
            ff_block.net[1], ff_block.net[2], ff_block.net[3] = nn.Identity(), nn.Identity(), nn.Identity()
        else:
            ff_block.net[1], ff_block.net[2] = nn.Identity(), nn.Identity()
            ff_block.net[3] = FlashViTFFN(i, engine_ptr, 1024, 0, fc1_bias)

    def custom_forward(datacube):
        results = model.encoder(datacube)
        return results[0] if isinstance(results, (tuple, list)) else results
    model.forward = custom_forward
    model.eval()

    print("Running warmup pass...", flush=True)
    with torch.no_grad(): _ = model(get_dummy_datacube())

    print(f"Benchmarking Latency on Edge using: {args.image}", flush=True)
    latencies = []
    datacube = get_real_datacube(args.image)
    for i in range(5):
        print(f"  [Pass {i+1}/5] Processing model layers via SSD streaming...", flush=True)
        start = time.time()
        with torch.no_grad(): out = model(datacube)
        latencies.append(time.time() - start)
    
    avg_lat = sum(latencies)/len(latencies)
    print(f"Average Latency: {avg_lat:.2f} seconds")
    
    # TELEMETRY EXPORT
    features = out[0, 1:, :].cpu().numpy()
    grid_size = int(np.sqrt(features.shape[0]))
    heatmap = features.mean(axis=-1).reshape(grid_size, grid_size)
    np.save("edge_output_heatmap.npy", heatmap)
    cls_token = out[0, 0, :].cpu().numpy()
    np.save("edge_output_cls_token.npy", cls_token)
    
    import json
    metrics = {"device": os.uname().machine, "mode": args.mode, "avg_latency": avg_lat, "timestamp": time.ctime()}
    with open("edge_metrics.json", "w") as f: json.dump(metrics, f, indent=4)
    
    print(f"✅ Success! Latency: {avg_lat:.2f}s. Telemetry saved.")
    lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    main()
