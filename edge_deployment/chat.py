"""
chat.py: The primary Python orchestration layer for the local chatbot.
Optimized for 8GB RAM systems using Lean Materialization and Layer-wise Loading.
"""

import os
import warnings
import logging
import ctypes
import torch
import torch.nn as nn
import sys
import time
import threading
import gc
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights

# --- 0. Setup & Silence ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OPENBLAS_VERBOSE"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "./hf_cache"
FFN_BIN_PATH = b"./opt_6_7b_bundled_ffn.bin"
LAYERS_DIR = "./opt_6_7b_layers"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
DEVICE = "cpu"
PREDICTOR_BIN_PATH = b"./opt_6_7b_predictors.bin"

# --- 1. C++ Engine Bindings ---
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
lib.init_engine.restype = ctypes.c_void_p
lib.set_engine_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
lib.set_predictor_layer_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
lib.execute_ffn_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
lib.destroy_engine.argtypes = [ctypes.c_void_p]

class ZeroModule(nn.Module):
    def forward(self, x): return torch.zeros_like(x)

class FlashFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, fc1_bias, fc2_bias, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.fc1_bias_c = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None
        self.fc2_bias = fc2_bias.half() if fc2_bias is not None else torch.zeros(HIDDEN_SIZE).half()

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, HIDDEN_SIZE).float().cpu().contiguous()
        out_cpu = torch.zeros_like(flat_x)
        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            flat_x.shape[0], self.fc1_bias_c, self.mode_int)
        return out_cpu.to(x.device, dtype=torch.float16).view(*orig_shape) + self.fc2_bias

def load_predictor_metadata(bin_path):
    if not bin_path: return None
    path_str = bin_path.decode() if isinstance(bin_path, bytes) else bin_path
    meta_path = path_str.replace(".bin", ".json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f: return json.load(f)
    return None

class StreamAndTimer:
    def __init__(self, tokenizer, mode_name):
        self.tokenizer = tokenizer
        self.mode_name = mode_name
        self.token_count = 0
        self.start_time = None
        self.full_text = ""
        self.print_len = 0
    
    def start(self):
        self.token_count = 0
        self.start_time = time.time()
        self.full_text = ""
        self.print_len = 0
        print(f"ASSISTANT [{self.mode_name.upper()}]: ", end="", flush=True)

    def put(self, value):
        if torch.is_tensor(value): value = value.view(-1).tolist()
        elif isinstance(value, int): value = [value]
        self.token_count += len(value)
        self.full_text = self.tokenizer.decode(self.tokenizer.encode(self.full_text) + value, skip_special_tokens=True)
        print(self.full_text[self.print_len:], end="", flush=True)
        self.print_len = len(self.full_text)
    
    def stop(self):
        elapsed = time.time() - self.start_time
        print(f"\n[Done] TPS: {self.token_count / elapsed:.2f}\n")
        # Save metrics for graphing
        import json
        metrics = {"device": os.uname().machine, "model": "opt", "mode": self.mode_name, "avg_latency": 1.0 / (self.token_count / elapsed) if self.token_count > 0 else 0, "tps": self.token_count / elapsed, "timestamp": time.ctime()}
        with open(f"edge_metrics_{self.mode_name}.json", "w") as f: json.dump(metrics, f, indent=4)

def chat(args):
    global CACHE_PATH, FFN_BIN_PATH, LAYERS_DIR, PREDICTOR_BIN_PATH
    if args.cache: CACHE_PATH = args.cache
    if args.ffn_bin: FFN_BIN_PATH = args.ffn_bin.encode()
    if args.layers: LAYERS_DIR = args.layers
    if args.predictor: PREDICTOR_BIN_PATH = args.predictor.encode()

    print(f"--- PATH VERIFICATION ---")
    print(f"  FFN Binary: {FFN_BIN_PATH.decode()}")
    print(f"  Predictor:  {PREDICTOR_BIN_PATH.decode()}")
    print(f"  Layers Dir: {LAYERS_DIR}")
    print(f"  HF Cache:   {CACHE_PATH}")
    print(f"--------------------------\n")

    load_path = MODEL_ID
    if os.path.exists(os.path.join(CACHE_PATH, "config.json")):
        load_path = CACHE_PATH
        print(f"Loading metadata directly from local cache folder: {CACHE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(load_path, cache_dir=CACHE_PATH, local_files_only=True)
    
    pred_path = PREDICTOR_BIN_PATH
    engine_ptr = lib.init_engine(FFN_BIN_PATH, pred_path, ctypes.c_size_t(HIDDEN_SIZE), ctypes.c_size_t(16384), ctypes.c_size_t(NUM_LAYERS), ctypes.c_int(0))
    meta = load_predictor_metadata(pred_path)
    if meta:
        offset = 0
        for i in range(NUM_LAYERS):
            lib.set_predictor_layer_info(engine_ptr, i, meta['rank'], ctypes.c_size_t(offset))
            offset += (meta['hidden_size'] * meta['rank']) + (meta['ffn_dim'] * meta['rank'])
    lib.set_engine_config(engine_ptr, args.top_k, args.threshold, args.window)
    mode_int = {"predictor": 0, "oracle": 2, "naive": 1, "draft": 0}[args.mode]
    
    config = AutoConfig.from_pretrained(load_path, cache_dir=CACHE_PATH, local_files_only=True)
    
    print("Materializing Lean Model Structure (FP16)...", flush=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        # Surgical Lean Materialization: Swap FFN for Identity BEFORE materialize to save 8GB RAM
        for i in range(NUM_LAYERS):
            layer = model.model.decoder.layers[i]
            layer.fc1 = nn.Identity(); layer.fc2 = nn.Identity(); layer.activation_fn = nn.Identity()

    model.to_empty(device="cpu")
    print("✅ Lean structure ready.", flush=True)
    
    def suffix_load(target_module, state_dict, name_hint=""):
        target_sd = target_module.state_dict()
        clean_sd = {}
        loaded_count = 0
        suffix_map = {}
        for k in target_sd.keys():
            suffix = ".".join(k.split('.')[-2:])
            if suffix not in suffix_map: suffix_map[suffix] = []
            suffix_map[suffix].append(k)
        for k_ckpt, v in state_dict.items():
            suffix_ckpt = ".".join(k_ckpt.split('.')[-2:])
            if suffix_ckpt in suffix_map:
                for k_target in suffix_map[suffix_ckpt]:
                    if target_sd[k_target].shape == v.shape:
                        clean_sd[k_target] = v.half(); loaded_count += 1; break
        target_module.load_state_dict(clean_sd, strict=False)
        print(f"  -> {name_hint}: Loaded {loaded_count} layers.", flush=True)

    print("Loading global resident layers...", flush=True)
    globals_sd = torch.load(os.path.join(LAYERS_DIR, "globals.pt"), map_location="cpu")
    suffix_load(model, globals_sd, "Globals")
    
    # CRITICAL: Tie LM Head to Embeddings (Brain fix)
    model.lm_head.weight = model.model.decoder.embed_tokens.weight
    print("✅ LM Head tied to Embeddings.", flush=True)
    
    print(f"Loading layers and patching ({args.mode.upper()})...", flush=True)
    for i in range(NUM_LAYERS):
        if i % 4 == 0: print(f"  -> Patching FlashFFN: Decoder Block {i}/{NUM_LAYERS}...", flush=True)
        layer_sd = torch.load(os.path.join(LAYERS_DIR, f"layer_{i}.pt"), map_location="cpu")
        layer = model.model.decoder.layers[i]
        
        if args.mode == "draft" and i % 4 == 0:
            layer.fc1 = ZeroModule()
        else:
            suffix_load(layer, layer_sd, f"Block {i}")
            fc1_b = None; fc2_b = None
            for k, v in layer_sd.items():
                if "fc1.bias" in k: fc1_b = v.float().cpu().contiguous()
                if "fc2.bias" in k: fc2_b = v.float().cpu().contiguous()
            layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
        del layer_sd; gc.collect()
            
    print("Finalizing model materialization...", flush=True)
    for p in model.parameters():
        if torch.isnan(p).any(): p.data.zero_()
    model.eval()
    
    timer = StreamAndTimer(tokenizer, args.mode)
    print("\nREADY. Type 'exit' to quit.\n")
    while True:
        try:
            try: user_input = input("YOU: ")
            except EOFError: print("\n[EOF] Ending benchmark."); break
            if user_input.lower() in ["quit", "exit"]: break
            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
            timer.start()
            model.generate(**inputs, max_new_tokens=20, streamer=timer, do_sample=False)
            timer.stop()
        except KeyboardInterrupt: break
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['predictor', 'oracle', 'naive', 'draft'], default='predictor')
    p.add_argument('--predictor', type=str)
    p.add_argument('--ffn_bin', type=str)
    p.add_argument('--layers', type=str)
    p.add_argument('--cache', type=str)
    p.add_argument('--top_k', type=int, default=1024); p.add_argument('--threshold', type=float, default=0.2); p.add_argument('--window', type=int, default=5)
    chat(p.parse_args())
