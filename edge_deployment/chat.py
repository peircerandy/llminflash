"""
chat.py: The primary Python orchestration layer for the local chatbot.
Optimized for 8GB RAM systems using Speculative Decoding and Layer-wise Loading.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

# --- 0. Setup & Silence ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OPENBLAS_VERBOSE"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration
MODEL_ID = "facebook/opt-6.7b"
ASST_ID = "facebook/opt-125m"
CACHE_PATH = "./hf_cache"
FFN_BIN_PATH = b"./opt_6_7b_bundled_ffn.bin"
LAYERS_DIR = "./opt_6_7b_layers"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PREDICTOR_BIN_PATH = b"./opt_6_7b_predictors.bin"

# --- 1. C++ Engine Bindings ---
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
lib.init_engine.restype = ctypes.c_void_p
lib.set_engine_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
lib.set_predictor_layer_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
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
        self.fc1_bias_ref = fc1_bias
        self.fc2_bias_ref = fc2_bias
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
        
        res = out_cpu.to(x.device, dtype=torch.float16).view(*orig_shape) + self.fc2_bias
        if torch.isnan(res).any():
            res = torch.nan_to_num(res)
        return res

def load_predictor_metadata(bin_path):
    if not bin_path: return None
    path_str = bin_path.decode() if isinstance(bin_path, bytes) else bin_path
    meta_path = path_str.replace(".bin", ".json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f: return json.load(f)
    return None

class WhitespaceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, max_whitespace=5):
        self.tokenizer = tokenizer
        self.max_whitespace = max_whitespace
        self.whitespace_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        last_token_id = input_ids[0, -1].item()
        last_token_text = self.tokenizer.decode([last_token_id])
        if last_token_text.isspace() or last_token_text == "":
            self.whitespace_count += 1
        else:
            self.whitespace_count = 0
        return self.whitespace_count >= self.max_whitespace

class StreamAndTimer:
    def __init__(self, tokenizer, mode_name):
        self.tokenizer = tokenizer
        self.mode_name = mode_name
        self.token_count = 0
        self.start_time = None
        self.tokens = []
        self.print_len = 0
        self.done_printed = False
    
    def start(self):
        self.token_count = 0
        self.start_time = None 
        self.tokens = []
        self.print_len = 0
        self.done_printed = False
        print(f"ASSISTANT [{self.mode_name.upper()}]: ", end="", flush=True)

    def put(self, value):
        if torch.is_tensor(value): value = value.view(-1).tolist()
        elif isinstance(value, int): value = [value]
        
        # Timing starts on the VERY FIRST token produced by generate()
        if self.start_time is None:
            self.start_time = time.time()
            
        self.token_count += len(value)
        self.tokens.extend(value)
        
        full_text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        print(full_text[self.print_len:], end="", flush=True)
        self.print_len = len(full_text)
    
    def end(self):
        self.stop()

    def stop(self):
        if self.done_printed or self.start_time is None: return
        self.done_printed = True
        elapsed = time.time() - self.start_time
        tps = self.token_count / elapsed if elapsed > 0.001 else 0
        print(f"\n[Done] Tokens: {self.token_count} | Elapsed: {elapsed:.2f}s | TPS: {tps:.2f}\n")
        
        metrics = {
            "device": os.uname().machine,
            "model": "opt",
            "mode": self.mode_name,
            "avg_latency": 1.0 / tps if tps > 0 else 0,
            "tps": tps,
            "timestamp": time.ctime()
        }
        with open(f"edge_metrics_{self.mode_name}.json", "w") as f:
            json.dump(metrics, f, indent=4)

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
    if os.path.exists(os.path.join(CACHE_PATH, "config.json")): load_path = CACHE_PATH
    tokenizer = AutoTokenizer.from_pretrained(load_path, cache_dir=CACHE_PATH, local_files_only=True)
    
    engine_ptr = lib.init_engine(FFN_BIN_PATH, PREDICTOR_BIN_PATH, ctypes.c_size_t(HIDDEN_SIZE), ctypes.c_size_t(16384), ctypes.c_size_t(NUM_LAYERS), ctypes.c_int(0), ctypes.c_int(args.max_cache))
    meta = load_predictor_metadata(PREDICTOR_BIN_PATH)
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

    print("Loading global resident layers...", flush=True)
    globals_sd = torch.load(os.path.join(LAYERS_DIR, "globals.pt"), map_location="cpu")
    for k, v in globals_sd.items():
        target = model.model if k.startswith("decoder.") else model
        set_module_tensor_to_device(target, k, DEVICE, value=v, dtype=torch.float16)
    del globals_sd; gc.collect()
    
    print(f"Loading layers and patching ({args.mode.upper()})...", flush=True)
    for i in range(NUM_LAYERS):
        if i % 4 == 0: print(f"  -> Decoder Block {i}/{NUM_LAYERS}...", flush=True)
        layer_sd = torch.load(os.path.join(LAYERS_DIR, f"layer_{i}.pt"), map_location="cpu")
        layer = model.model.decoder.layers[i]
        for k, v in layer_sd.items():
            if "fc" in k: continue
            clean_k = k.replace(f"decoder.layers.{i}.", "")
            set_module_tensor_to_device(layer, clean_k, DEVICE, value=v, dtype=torch.float16)
        fc1_b = layer_sd[f"decoder.layers.{i}.fc1.bias"].float().cpu().contiguous()
        fc2_b = layer_sd[f"decoder.layers.{i}.fc2.bias"].float().cpu().contiguous()
        layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
        layer.fc2 = nn.Identity(); layer.activation_fn = nn.Identity()
        del layer_sd; gc.collect()
            
    print("Tying weights and finalizing...", flush=True)
    model.tie_weights()
    with torch.no_grad():
        for p in model.parameters():
            if torch.isnan(p).any(): p.data.zero_()
    model.eval()

    with torch.no_grad():
        test_input = tokenizer("Hello", return_tensors="pt").to(DEVICE)
        test_out = model(**test_input)
        logits_max = test_out.logits.abs().max().item()
        print(f"✅ Model 'Alive' Check: Max Logit Intensity = {logits_max:.2f}", flush=True)

    assistant_model = None
    if args.mode == "draft":
        print("Initializing Assistant (OPT-125M) for Speculative Decoding...", flush=True)
        asst_path = os.path.join(CACHE_PATH, "assistant")
        load_id = asst_path if os.path.exists(os.path.join(asst_path, "pytorch_model.bin")) else ASST_ID
        assistant_model = AutoModelForCausalLM.from_pretrained(load_id, torch_dtype=torch.float16, device_map=DEVICE)
        assistant_model.eval()

    timer = StreamAndTimer(tokenizer, args.mode)
    print("\nREADY. Type 'exit' to quit.\n")
    
    gen_kwargs = {
        "max_new_tokens": 15, 
        "streamer": timer, 
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList([WhitespaceStoppingCriteria(tokenizer)])
    }

    if args.prompt:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(DEVICE)
        timer.start()
        if assistant_model: gen_kwargs["assistant_model"] = assistant_model
        model.generate(**inputs, **gen_kwargs)
        timer.stop()
        return

    while True:
        try:
            try: user_input = input("YOU: ")
            except EOFError: print("\n[EOF] Ending benchmark."); break
            if user_input.lower() in ["quit", "exit"]: break
            inputs = tokenizer(user_input, return_tensors="pt").to(DEVICE)
            timer.start()
            if assistant_model: gen_kwargs["assistant_model"] = assistant_model
            model.generate(**inputs, **gen_kwargs)
            timer.stop()
        except KeyboardInterrupt: break
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['predictor', 'oracle', 'naive', 'draft'], default='predictor')
    p.add_argument('--predictor', type=str); p.add_argument('--ffn_bin', type=str); p.add_argument('--layers', type=str); p.add_argument('--cache', type=str)
    p.add_argument('--prompt', type=str)
    p.add_argument('--max_cache', type=int, default=1024)
    p.add_argument('--top_k', type=int, default=1024); p.add_argument('--threshold', type=float, default=0.2); p.add_argument('--window', type=int, default=5)
    chat(p.parse_args())
