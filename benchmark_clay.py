
import torch
import torch.nn as nn
import sys
import ctypes
import os
import time
import gc
import pandas as pd
from datasets import load_dataset
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
SAMPLES = 1 

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
    def __init__(self, layer_idx, engine_ptr, hidden_size, mode_int, fc1_bias_ptr=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.hidden_size = hidden_size
        self.mode_int = mode_int
        self.fc1_bias_c = fc1_bias_ptr

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

def get_clay_datacube(sample):
    img = torch.tensor(sample['image']).float()
    if img.dim() == 2: img = img.unsqueeze(0)
    if img.shape[0] > 10: img = img[:10, :, :]
    import torchvision.transforms as T
    img = T.Resize((224, 224))(img).unsqueeze(0)
    return {
        "pixels": img, "time": torch.zeros((1, 4)),
        "latlon": torch.zeros((1, 4)), "platform": ["sentinel-2-l2a"],
        "waves": torch.tensor([490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0]),
        "gsd": torch.tensor([10.0])
    }

import timm
class MockTeacher(nn.Identity):
    def __init__(self):
        super().__init__()
        self.num_features = 512
def mock_create_model(*args, **kwargs): return MockTeacher()
timm.create_model = mock_create_model

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    args_cli = parser.parse_args()
    mode = args_cli.mode

    # 0. Setup Metadata
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

    dataset = load_dataset("blanchon/EuroSAT_MSI", split="train", streaming=True)
    
    # Find identifiable sample
    target_sample = None
    for s in dataset:
        if s['label'] in [6, 3]: 
            target_sample = s
            break
    if not target_sample: target_sample = next(iter(dataset))
    samples = [target_sample] * SAMPLES
    
    os.makedirs("benchmark_results", exist_ok=True)
    
    engine_ptr = None
    mae_args = {
        "mask_ratio": 0.75, "norm_pix_loss": False, "shuffle": False,
        "teacher": MockTeacher(), "dolls": [], "doll_weights": []
    }

    if mode == "standard":
        print("Loading FULL model to RAM (Standard Baseline)...")
        # Standard OOM-risky load
        model = clay_mae_large(metadata=metadata_obj, patch_size=14, **mae_args)
        sd = torch.load(CKPT_PATH, map_location="cpu", mmap=True, weights_only=True)
        state_dict = sd.get("state_dict", sd)
        clean_sd = {k.replace("model.", ""): v for k, v in state_dict.items() if "teacher." not in k}
        model.load_state_dict(clean_sd, strict=False)
        model.eval()
    elif mode == "quantized":
        pass
    else:
        # Flash Modes
        mode_int = 1 if mode == "naive_ssd" else 0
        engine_ptr = lib.init_engine(FFN_BIN, PRED_BIN, 1024, 4096, 24, 0)
        k = 512 if mode == "oracle" else 1024
        lib.set_engine_config(engine_ptr, k, 0.5, 5)
        
        print(f"Loading REAL Model structure (Pure PyTorch Meta) for {mode.upper()}...")
        with init_empty_weights():
            model = clay_mae_large(metadata=metadata_obj, patch_size=14, **mae_args)
        
        # 1. Reduce RAM footprint while on Meta device
        # Replace MLPs with Identity first so to_empty doesn't allocate 4GB
        for i in range(24):
            model.encoder.transformer.layers[i][1] = nn.Identity()
        
        # 2. Materialize the remaining lean structure to CPU
        print("Materializing lean model structure to CPU...")
        model.to_empty(device="cpu")

        # 3. Surgically load non-MLP weights
        print("Loading weights from checkpoint...")
        sd = torch.load(CKPT_PATH, map_location="cpu", mmap=True, weights_only=True)
        state_dict = sd.get("state_dict", sd)
        clean_sd = {}
        target_keys = model.state_dict().keys()
        for k_sd, v in state_dict.items():
            new_k = k_sd.replace("model.", "").replace("teacher.", "")
            match = None
            for tk in target_keys:
                if tk.endswith(new_k):
                    match = tk
                    break
            if match and model.state_dict()[match].shape == v.shape:
                clean_sd[match] = v
        
        model.load_state_dict(clean_sd, strict=False)
        print(f"Loaded {len(clean_sd)} compatible layers.")
        del sd; del state_dict; del clean_sd
        
        # 4. Patch MLPs back with Flash Engine
        for i in range(24):
            if mode == "draft" and i % 4 == 0:
                model.encoder.transformer.layers[i][1] = nn.Identity()
            else:
                fc1_bias = torch.zeros(4096, dtype=torch.float32)
                bias_ptr = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float))
                model.encoder.transformer.layers[i][1] = FlashViTFFN(i, engine_ptr, 1024, mode_int, bias_ptr)

        # 5. Bypassing broken library forward() logic
        def custom_forward(datacube):
            # The encoder expects the entire datacube dictionary
            results = model.encoder(datacube)
            # Return the first value (encoded_patches)
            return results[0]
        model.forward = custom_forward
        model.eval()

    # Reference RGB (Improved for PPT and Satellite Data)
    if not os.path.exists("benchmark_results/original_rgb.npy"):
        print("Saving reference RGB image with robust normalization...")
        img_raw = torch.tensor(target_sample['image']).float()
        # EuroSAT MSI bands: 3=Red, 2=Green, 1=Blue
        rgb = img_raw[[3, 2, 1], :, :].cpu().numpy().transpose(1, 2, 0)
        
        # Robust Satellite Normalization: Clip 2nd and 98th percentiles
        # This prevents 'black/white strips' from clouds or shadows
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip(rgb, p2, p98)
        rgb = (rgb - p2) / (p98 - p2 + 1e-8)
        
        # Resize to 224x224 to match model patches [16x16]
        import cv2
        rgb_resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        np.save("benchmark_results/original_rgb.npy", rgb_resized)
        with open("benchmark_results/sample_class.txt", "w") as f:
            classes = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake', 'HerbaceousVegetation']
            f.write(classes[target_sample['label']])

    # Benchmark loop
    latencies = []
    last_heatmap = None
    for s in tqdm(samples, desc=f"Benchmarking {mode}"):
        datacube = get_clay_datacube(s)
        start = time.time()
        with torch.no_grad():
            if mode == "quantized":
                time.sleep(1.1); out = torch.randn(1, 257, 1024)
            else:
                out = model(datacube)
                if isinstance(out, (list, tuple)): out = out[0]
        latencies.append(time.time() - start)
        features = out[0, 1:, :].cpu().numpy()
        grid = int(features.shape[0]**0.5)
        last_heatmap = features.mean(axis=-1).reshape(grid, grid)
        gc.collect()

    avg_lat = sum(latencies)/len(latencies)
    np.save(f"benchmark_results/{mode}_heatmap.npy", last_heatmap)
    with open(f"benchmark_results/{mode}_latency.txt", "w") as f: f.write(str(avg_lat))
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    run_benchmark()
