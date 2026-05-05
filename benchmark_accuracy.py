"""
benchmark_accuracy.py: Automated testing suite for "LLM in a Flash".
Optimized for 8GB RAM systems using Layer-wise Loading.
"""

import torch
import torch.nn as nn
import transformers
from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import ctypes
import os
import glob
import time
import gc
import logging
import argparse
import pandas as pd

# Suppress HF Hub warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
transformers.logging.set_verbosity_error() 
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Configuration
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
LAYERS_DIR = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_layers"
HIDDEN_SIZE = 4096
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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

class FlashFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, fc1_bias, fc2_bias, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.fc1_bias = fc1_bias 
        self.fc1_bias_c = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float))
        self.fc2_bias = fc2_bias.to(DEVICE).to(torch.float16)

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, HIDDEN_SIZE).float().cpu().contiguous()
        num_tokens = flat_x.shape[0]
        out_cpu = torch.zeros_like(flat_x)
        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            num_tokens, self.fc1_bias_c, self.mode_int)
        res = out_cpu.to(x.device, dtype=torch.float16).view(*orig_shape)
        return res + self.fc2_bias

TEST_QUESTIONS = [
    {"prompt": "Question: What is the capital of France?\nAnswer:", "expected": "Paris", "desc": "Basic Knowledge"},
    {"prompt": "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers.", "expected": "teach", "desc": "Overfit Recall"},
]

def load_flash_model(mode_name, top_k=1024, threshold=0.2):
    print(f"Loading model in {mode_name.upper()} mode...")
    pred_path = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_predictors.bin"
    engine_ptr = lib.init_engine(FFN_BIN_PATH, pred_path if os.path.exists(pred_path.decode()) else b"", ctypes.c_size_t(HIDDEN_SIZE), ctypes.c_size_t(16384), ctypes.c_size_t(32), ctypes.c_int(0))
    lib.set_engine_config(engine_ptr, top_k, threshold, 5)
    
    mode_int = {"predictor": 0, "oracle": 1, "naive": 2}[mode_name]
    
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
    with init_empty_weights():
        model = OPTForCausalLM(config)
        
    globals_sd = torch.load(os.path.join(LAYERS_DIR, "globals.pt"), map_location="cpu")
    for k, v in globals_sd.items():
        target = model.model if k.startswith("decoder.") else model
        set_module_tensor_to_device(target, k, DEVICE, value=v, dtype=torch.float16)
    del globals_sd
    set_module_tensor_to_device(model, "lm_head.weight", DEVICE, value=model.model.decoder.embed_tokens.weight, dtype=torch.float16)

    for i in range(32):
        layer_file = os.path.join(LAYERS_DIR, f"layer_{i}.pt")
        layer_sd = torch.load(layer_file, map_location="cpu")
        fc1_b = layer_sd[f"decoder.layers.{i}.fc1.bias"].float().cpu().contiguous()
        fc2_b = layer_sd[f"decoder.layers.{i}.fc2.bias"].float().cpu().contiguous()
        layer = model.model.decoder.layers[i]
        for k, v in layer_sd.items():
            if ".bias" in k and ("fc1" in k or "fc2" in k): continue
            rel_k = k.replace(f"decoder.layers.{i}.", "")
            set_module_tensor_to_device(layer, rel_k, DEVICE, value=v, dtype=torch.float16)
        layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
        layer.fc2 = nn.Identity()
        layer.activation_fn = nn.Identity()
        del layer_sd; gc.collect()
    
    return model, engine_ptr

def run_suite():
    results = []
    modes = ["oracle", "predictor"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)

    for mode in modes:
        model, engine_ptr = load_flash_model(mode)
        model.eval()
        
        total_tps = 0
        total_correct = 0
        
        for q in TEST_QUESTIONS:
            inputs = tokenizer(q["prompt"], return_tensors="pt").to(model.device)
            start = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            elapsed = time.time() - start
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            tps = 10 / elapsed
            total_tps += tps
            is_correct = q["expected"].lower() in text.lower()
            if is_correct: total_correct += 1
            print(f"Mode {mode} | Q: {q['desc']} | TPS: {tps:.2f} | Correct: {is_correct}")
            
        results.append({
            "Mode": mode.upper(),
            "Avg Tokens/Sec": total_tps / len(TEST_QUESTIONS),
            "Avg Time (s)": 10 / (total_tps / len(TEST_QUESTIONS)),
            "Accuracy %": (total_correct / len(TEST_QUESTIONS)) * 100
        })
        
        del model
        lib.destroy_engine(engine_ptr)
        gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df)
    df.to_csv("presentation_stats.csv", index=False)

if __name__ == "__main__":
    run_suite()
