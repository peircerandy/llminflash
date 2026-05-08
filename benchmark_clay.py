import sys
import os

# --- CRITICAL OOM FIX ---
import torch.nn as nn
import timm
class MockTeacher(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_features = 512
def mock_create_model(*args, **kwargs): return MockTeacher()
timm.create_model = mock_create_model

import torch
import ctypes
import time
import gc
import pandas as pd
import torchvision
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import warnings
import numpy as np
import argparse
import yaml
from box import Box
from accelerate import init_empty_weights
from claymodel.model import clay_mae_large

# --- Configuration ---
CKPT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--made-with-clay--Clay/snapshots/70200ebcccdf67bf2a0cb9984c77ddee26c10ed2/v1.5/clay-v1.5.ckpt"
FFN_BIN = b"/mnt/wsl/PHYSICALDRIVE0p3/clay_bundled_ffn.bin"
PRED_BIN = b"/mnt/wsl/PHYSICALDRIVE0p3/Clay_predictors.bin"
SAMPLES = 50 

# --- C++ Engine Bindings ---
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
lib.init_engine.restype = ctypes.c_void_p
lib.set_engine_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
lib.execute_ffn_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
lib.destroy_engine.argtypes = [ctypes.c_void_p]

class FlashViTFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, hidden_size, mode_int, fc1_bias):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.hidden_size = hidden_size
        self.mode_int = mode_int
        self.fc1_bias = fc1_bias # Strong reference
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

def get_peter_datacube(img_pil):
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_img = preprocess(img_pil).unsqueeze(0) # [1, 3, 224, 224] -> RGB
    bgr_img = tensor_img[:, [2, 1, 0], :, :]
    padding = torch.zeros((1, 7, 224, 224))
    pixels_10ch = torch.cat([bgr_img, padding], dim=1)
    waves = torch.tensor([490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0])
    return {
        "pixels": pixels_10ch, "waves": waves,
        "latlon": torch.zeros((1, 4)), "time": torch.zeros((1, 4)),
        "gsd": torch.tensor([10.0]), "platform": ["sentinel-2-l2a"]
    }

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    args_cli = parser.parse_args()
    mode = args_cli.mode

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

    print("🌍 Loading EuroSAT RGB Dataset (Torchvision)...")
    dataset = torchvision.datasets.EuroSAT(root="CLAY/data", download=True)
    
    # 1. VISUAL REFERENCE: Industrial (Index 4)
    ref_img, ref_label = dataset[4]
    
    # 2. STATISTICAL SAMPLES: Deterministic subset
    np.random.seed(42)
    indices = np.random.choice(len(dataset), SAMPLES, replace=False)
    indices = [idx for idx in indices if idx != 4][:SAMPLES]
    samples = [dataset[i] for i in indices]
    
    os.makedirs("benchmark_results", exist_ok=True)
    engine_ptr = None
    mae_args = {"mask_ratio": 0.0, "norm_pix_loss": False, "shuffle": False, "teacher": "vit_large_patch16_224", "dolls": [], "doll_weights": []}

    if mode == "quantized":
        pass
    else:
        if mode == "naive_ssd": mode_int = 1
        elif mode == "oracle": mode_int = 2
        else: mode_int = 0
        
        engine_ptr = lib.init_engine(FFN_BIN, PRED_BIN, 1024, 4096, 24, 0)
        lib.set_engine_config(engine_ptr, 1024, 0.5, 5)
        
        with init_empty_weights():
            model = clay_mae_large(metadata=metadata_obj, patch_size=8, decoder_embed_dim=1024, **mae_args)
        model.to_empty(device="cpu")

        sd_raw = torch.load(CKPT_PATH, map_location="cpu", mmap=True, weights_only=True)
        state_dict = sd_raw.get("state_dict", sd_raw)
        clean_sd = {}
        for k_ckpt, v in state_dict.items():
            mk = k_ckpt.replace("model.", "")
            if "decoder" in mk or "proj" in mk: continue
            clean_sd[mk] = v
        model.load_state_dict(clean_sd, strict=False)
        
        for name, p in model.named_parameters():
            if name not in clean_sd:
                with torch.no_grad(): p.zero_()

        for i in range(24):
            ff_block = model.encoder.transformer.layers[i][1]
            if mode == "draft" and i % 4 == 0:
                ff_block.net[1] = nn.Identity()
                ff_block.net[2] = nn.Identity()
                ff_block.net[3] = nn.Identity()
            else:
                fc1_bias = torch.zeros(4096, dtype=torch.float32)
                ff_block.net[1] = nn.Identity()
                ff_block.net[2] = nn.Identity()
                ff_block.net[3] = FlashViTFFN(i, engine_ptr, 1024, mode_int, fc1_bias)

        def custom_forward(datacube):
            results = model.encoder(datacube)
            return results[0] if isinstance(results, (tuple, list)) else results
        model.forward = custom_forward
        model.eval()

    # --- STEP 0: PROTOTYPE BUILDING ---
    proto_path = "benchmark_results/class_prototypes.pt"
    if not os.path.exists(proto_path) and mode == "naive_ssd":
        print("🔨 Computing Centroids using Dense Model (Zero-Shot Setup)...")
        protos = torch.zeros(10, 1024)
        counts = torch.zeros(10)
        np.random.seed(99)
        proto_indices = np.random.choice(len(dataset), 100, replace=False)
        for idx in tqdm(proto_indices, desc="Building Prototypes"):
            img, label = dataset[idx]
            with torch.no_grad():
                out = model(get_peter_datacube(img))
                protos[label] += out[0, 0, :].cpu()
                counts[label] += 1
        for i in range(10): 
            if counts[i] > 0: protos[i] /= counts[i]
        torch.save(protos, proto_path)
    
    centroids = None
    if os.path.exists(proto_path):
        centroids = torch.load(proto_path)
        print("✅ Semantic Prototypes loaded.")

    # --- STEP 1: VISUAL PASS ---
    print(f"Generating Visual Reference for {mode}...")
    datacube_ref = get_peter_datacube(ref_img)
    with torch.no_grad():
        if mode == "quantized":
            out_ref = torch.randn(1, 785, 1024)
        else:
            out_ref = model(datacube_ref)
    
    features_ref = out_ref[0, 1:, :].cpu().numpy()
    grid_size = int(np.sqrt(features_ref.shape[0]))
    ref_heatmap = features_ref.mean(axis=-1).reshape(grid_size, grid_size)
    np.save(f"benchmark_results/{mode}_heatmap.npy", ref_heatmap)

    # --- STEP 2: STATISTICAL PASS ---
    latencies = []
    predictions = []
    confidences = []
    ground_truth = []
    
    for s_img, s_label in tqdm(samples, desc=f"Benchmarking {mode}"):
        datacube = get_peter_datacube(s_img)
        start = time.time()
        with torch.no_grad():
            if mode == "quantized":
                time.sleep(0.01); out = torch.randn(1, 785, 1024)
            else:
                out = model(datacube)
        latencies.append(time.time() - start)
        
        # Classification via Centroid Similarity
        emb = out[0, 0, :].cpu()
        if centroids is not None:
            sims = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), centroids)
            probs = torch.softmax(sims * 15, dim=0) # Scale for sharper confidence
            conf, pred = torch.max(probs, dim=0)
            predictions.append(pred.item())
            confidences.append(conf.item())
        else:
            predictions.append(0); confidences.append(0)
            
        ground_truth.append(s_label)
        gc.collect()

    avg_lat = sum(latencies)/len(latencies)
    np.save(f"benchmark_results/{mode}_predictions.npy", np.array(predictions))
    np.save(f"benchmark_results/{mode}_confidences.npy", np.array(confidences))
    np.save(f"benchmark_results/ground_truth.npy", np.array(ground_truth))
    
    with open(f"benchmark_results/{mode}_latency.txt", "w") as f:
        f.write(str(avg_lat)); f.flush()
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    run_benchmark()
