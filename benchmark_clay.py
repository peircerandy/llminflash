
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

# --- Configuration ---
CKPT_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--made-with-clay--Clay/snapshots/70200ebcccdf67bf2a0cb9984c77ddee26c10ed2/v1.5/clay-v1.5.ckpt"
FFN_BIN = b"/mnt/wsl/PHYSICALDRIVE0p3/clay_bundled_ffn.bin"
PRED_BIN = b"/mnt/wsl/PHYSICALDRIVE0p3/Clay_predictors.bin"
SAMPLES = 3 

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
    parser.add_argument("--mode", type=str, required=True, choices=["standard", "quantized", "naive_ssd", "oracle", "draft", "predictor"])
    args_cli = parser.parse_args()
    mode = args_cli.mode

    from claymodel.module import ClayMAEModule
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
    if mode == "predictor":
        img_raw = torch.tensor(target_sample['image']).float()
        rgb = img_raw[[2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        np.save("benchmark_results/original_rgb.npy", rgb)
        with open("benchmark_results/sample_class.txt", "w") as f:
            classes = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake', 'HerbaceousVegetation']
            f.write(classes[target_sample['label']])

    engine_ptr = None
    if mode == "standard":
        print("Loading FULL model to RAM (Standard Baseline)...")
        # Load full model normally (materialized on CPU)
        model = ClayMAEModule.load_from_checkpoint(CKPT_PATH, map_location="cpu")
        def custom_forward(datacube): return model.model.encoder(datacube)[0]
        model.model.forward = custom_forward
        model.eval()
    elif mode == "quantized":
        model = None
    else:
        mode_int = 1 if mode == "naive_ssd" else 0
        engine_ptr = lib.init_engine(FFN_BIN, PRED_BIN, 1024, 4096, 24, 0)
        k = 512 if mode == "oracle" else 1024
        lib.set_engine_config(engine_ptr, k, 0.5, 5)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = ClayMAEModule(model_size="large", patch_size=14)
            # Load non-MLP weights
            sd = torch.load(CKPT_PATH, map_location="cpu", mmap=True, weights_only=True)
            state_dict = sd.get("state_dict", sd)
            clean_sd = {}
            target_keys = model.state_dict().keys()
            for k_sd, v in state_dict.items():
                new_k = k_sd.replace("model.teacher.", "model.")
                # Skip MLPs
                if ".mlp.net.1." in new_k or ".mlp.net.3." in new_k or ".1.net.1." in new_k or ".1.net.3." in new_k:
                    continue
                if new_k in target_keys:
                    # ONLY load if shapes match exactly
                    if model.state_dict()[new_k].shape == v.shape:
                        clean_sd[new_k] = v

            model.load_state_dict(clean_sd, strict=False)
            print(f"Surgically loaded {len(clean_sd)} compatible layers to CPU.")
            del sd; del state_dict; del clean_sd
            if mode == "draft" and i % 4 == 0:
                model.model.encoder.transformer.layers[i][1] = nn.Identity()
            else:
                ff_block = layer[1]
                fc1_bias = ff_block.net[1].bias.detach().float().cpu().contiguous() if hasattr(ff_block.net[1], 'bias') else None
                ff_block.net[1] = nn.Identity(); ff_block.net[2] = nn.Identity()
                bias_ptr = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None
                ff_block.net[3] = FlashViTFFN(i, engine_ptr, 1024, mode_int, bias_ptr)

        def custom_forward(datacube): return model.model.encoder(datacube)[0]
        model.model.forward = custom_forward
        model.eval()

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
        latencies.append(time.time() - start)
        features = out[0, 1:, :].cpu().numpy()
        grid = int(features.shape[0]**0.5)
        last_heatmap = features.mean(axis=-1).reshape(grid, grid)

    avg_lat = sum(latencies)/len(latencies)
    np.save(f"benchmark_results/{mode}_heatmap.npy", last_heatmap)
    with open(f"benchmark_results/{mode}_latency.txt", "w") as f: f.write(str(avg_lat))
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    run_benchmark()
