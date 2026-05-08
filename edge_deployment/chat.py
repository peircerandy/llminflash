"""
chat.py: The primary Python orchestration layer for the local chatbot.
Optimized for 8GB RAM / 6GB VRAM systems using Layer-wise Loading.
"""

import os
import warnings
import logging
import ctypes
import glob
import torch
import torch.nn as nn
import sys
import time
import threading
import gc
import argparse
import shutil
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

# --- 0. Setup & Silence ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
LAYERS_DIR = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_layers"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PREDICTOR_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_predictors.bin"

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

def load_predictor_metadata(bin_path):
    if not bin_path: return None
    path_str = bin_path.decode() if isinstance(bin_path, bytes) else bin_path
    meta_path = path_str.replace(".bin", ".json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None

class FlashFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, fc1_bias, fc2_bias, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.fc1_bias_c = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None
        self.fc2_bias = fc2_bias.to(DEVICE).half() if fc2_bias is not None else torch.zeros(HIDDEN_SIZE).to(DEVICE).half()

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, HIDDEN_SIZE).float().cpu().contiguous()
        num_tokens = flat_x.shape[0]
        out_cpu = torch.zeros_like(flat_x)
        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            num_tokens, self.fc1_bias_c, self.mode_int)
        return out_cpu.to(x.device, dtype=torch.float16).view(*orig_shape) + self.fc2_bias

class StreamAndTimer:
    def __init__(self, tokenizer, mode_name):
        self.tokenizer = tokenizer
        self.mode_name = mode_name
        self.token_count = 0
        self.start_time = None
        self.full_text = ""
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None
        self.print_len = 0
    
    def start(self):
        with self.lock:
            self.token_count = 0
            self.start_time = time.time()
            self.full_text = ""
            self.is_running = True
            self.print_len = 0
            print(f"ASSISTANT [{self.mode_name.upper()}]: ", end="", flush=True)
            self.thread = threading.Thread(target=self._spin)
            self.thread.daemon = True
            self.thread.start()

    def _spin(self):
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while self.is_running:
            time.sleep(0.1)
            idx = (idx + 1) % 4
    
    def put(self, value):
        with self.lock:
            if torch.is_tensor(value): value = value.view(-1).tolist()
            elif isinstance(value, int): value = [value]
            self.token_count += len(value)
            self.full_text = self.tokenizer.decode(self.tokenizer.encode(self.full_text) + value, skip_special_tokens=True)
            print(self.full_text[self.print_len:], end="", flush=True)
            self.print_len = len(self.full_text)
    
    def stop(self):
        self.is_running = False
        if self.thread: self.thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time
        print(f"\n[Done] TPS: {self.token_count / elapsed:.2f}\n")

def chat(args):
    print(f"Initializing {args.mode.upper()} Mode...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
    engine_ptr = None; assistant_model = None

    if args.mode == "quantized":
        q_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q_config, device_map="auto", cache_dir=CACHE_PATH, local_files_only=True)
    else:
        pred_path = args.predictor.encode() if args.predictor else PREDICTOR_BIN_PATH
        engine_ptr = lib.init_engine(FFN_BIN_PATH, pred_path, ctypes.c_size_t(HIDDEN_SIZE), ctypes.c_size_t(16384), ctypes.c_size_t(NUM_LAYERS), ctypes.c_int(0))
        meta = load_predictor_metadata(pred_path)
        if meta:
            offset = 0
            for i in range(NUM_LAYERS):
                lib.set_predictor_layer_info(engine_ptr, i, meta['rank'], ctypes.c_size_t(offset))
                offset += (meta['hidden_size'] * meta['rank']) + (meta['ffn_dim'] * meta['rank'])
        lib.set_engine_config(engine_ptr, args.top_k, args.threshold, args.window)
        mode_int = {"predictor": 0, "oracle": 1, "naive": 2, "speculative_custom": 0}[args.mode]
        config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
        with init_empty_weights(): model = AutoModelForCausalLM.from_config(config)
        globals_sd = torch.load(os.path.join(LAYERS_DIR, "globals.pt"), map_location="cpu")
        for k, v in globals_sd.items():
            target = model.model if k.startswith("decoder.") else model
            set_module_tensor_to_device(target, k, DEVICE, value=v, dtype=torch.float16)
        print("Loading layers and patching...")
        for i in range(NUM_LAYERS):
            layer_sd = torch.load(os.path.join(LAYERS_DIR, f"layer_{i}.pt"), map_location="cpu")
            layer = model.model.decoder.layers[i]
            fc1_b = layer_sd[f"decoder.layers.{i}.fc1.bias"].float().cpu().contiguous()
            fc2_b = layer_sd[f"decoder.layers.{i}.fc2.bias"].float().cpu().contiguous()
            for k, v in layer_sd.items():
                if ".bias" in k: continue
                set_module_tensor_to_device(layer, k.replace(f"decoder.layers.{i}.", ""), DEVICE, value=v, dtype=torch.float16)
            layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
            layer.fc2 = nn.Identity(); layer.activation_fn = nn.Identity()
            del layer_sd; gc.collect()

    if "speculative" in args.mode:
        assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", cache_dir=CACHE_PATH, local_files_only=True)

    model.eval()
    timer = StreamAndTimer(tokenizer, args.mode)
    print("\nREADY. Type 'exit' to quit.\n")
    while True:
        try:
            try:
                user_input = input("YOU: ")
            except EOFError:
                print("\n[EOF] Ending non-interactive benchmark.")
                break
                
            if user_input.lower() in ["quit", "exit"]: break
            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
            timer.start()
            kwargs = {"max_new_tokens": 50, "streamer": timer, "do_sample": False}
            if assistant_model: kwargs["assistant_model"] = assistant_model
            model.generate(**inputs, **kwargs)
            timer.stop()
        except KeyboardInterrupt: break
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interactive Chat CLI for LLM-in-a-Flash.")
    p.add_argument('--mode', choices=['predictor', 'oracle', 'naive', 'quantized', 'speculative_custom'], default='predictor', help="Inference mode")
    p.add_argument('--predictor', type=str, help="Path to .bin predictor")
    p.add_argument('--top_k', type=int, default=1024)
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--window', type=int, default=5)
    chat(p.parse_args())
