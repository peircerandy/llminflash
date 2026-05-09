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
os.environ["OPENBLAS_VERBOSE"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration (Defaults for Edge)
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "./hf_cache"
FFN_BIN_PATH = b"./opt_6_7b_bundled_ffn.bin"
LAYERS_DIR = "./opt_6_7b_layers"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
DEVICE = "cpu" # Default to CPU for Pi/Phone
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
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
    def forward(self, x):
        # Return zeros matching the input shape to correctly skip FFN residual in OPT
        return torch.zeros_like(x)

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
        # Resident bias on CPU (FP16)
        self.fc2_bias = fc2_bias.half() if fc2_bias is not None else torch.zeros(HIDDEN_SIZE).half()

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
        tps = self.token_count / elapsed
        print(f"\n[Done] TPS: {tps:.2f}\n")
        
        # Save metrics for graphing
        import json
        metrics = {
            "device": os.uname().machine,
            "model": "opt",
            "mode": self.mode_name,
            "avg_latency": 1.0 / tps if tps > 0 else 0, # Latency per token
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

    print(f"Initializing {args.mode.upper()} Mode...")
    
    # --- OFFLINE LOADING FIX ---
    load_path = MODEL_ID
    if os.path.exists(os.path.join(CACHE_PATH, "config.json")):
        load_path = CACHE_PATH
        print(f"Loading metadata directly from local cache folder: {CACHE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(load_path, cache_dir=CACHE_PATH, local_files_only=True)
    engine_ptr = None; assistant_model = None

    if args.mode == "quantized":
        q_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(load_path, quantization_config=q_config, device_map="auto", cache_dir=CACHE_PATH, local_files_only=True)
    else:
        pred_path = PREDICTOR_BIN_PATH
        engine_ptr = lib.init_engine(FFN_BIN_PATH, pred_path, ctypes.c_size_t(HIDDEN_SIZE), ctypes.c_size_t(16384), ctypes.c_size_t(NUM_LAYERS), ctypes.c_int(0))
        meta = load_predictor_metadata(pred_path)
        if meta:
            offset = 0
            for i in range(NUM_LAYERS):
                lib.set_predictor_layer_info(engine_ptr, i, meta['rank'], ctypes.c_size_t(offset))
                offset += (meta['hidden_size'] * meta['rank']) + (meta['ffn_dim'] * meta['rank'])
        lib.set_engine_config(engine_ptr, args.top_k, args.threshold, args.window)
        
        # Engine Modes: 0=Predictor, 1=Naive, 2=Oracle
        mode_int = {"predictor": 0, "oracle": 2, "naive": 1, "draft": 0}[args.mode]
        
        config = AutoConfig.from_pretrained(load_path, cache_dir=CACHE_PATH, local_files_only=True)
        with init_empty_weights(): 
            # FORCE FP16 for Edge Devices
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        
        # Surgical Materialization to CPU
        print("Materializing Model Structure to CPU (FP16)...", flush=True)
        model.to_empty(device="cpu")
        print("✅ Structure ready.", flush=True)
        
        def smart_load(target_module, state_dict, name_hint=""):
            """Tries to load weights even if prefixes (model. or decoder.) are slightly different."""
            target_sd = target_module.state_dict()
            clean_sd = {}
            loaded_count = 0
            
            # Try to match keys by stripping common prefixes
            for k_ckpt, v in state_dict.items():
                # Potential keys: 'model.decoder.embed_tokens.weight', 'decoder.embed_tokens.weight'
                # Target keys: 'model.decoder.embed_tokens.weight' or 'decoder.embed_tokens.weight'
                
                match_key = None
                if k_ckpt in target_sd: match_key = k_ckpt
                elif k_ckpt.replace("model.", "") in target_sd: match_key = k_ckpt.replace("model.", "")
                elif f"model.{k_ckpt}" in target_sd: match_key = f"model.{k_ckpt}"
                
                if match_key and target_sd[match_key].shape == v.shape:
                    clean_sd[match_key] = v.half()
                    loaded_count += 1
            
            msg = target_module.load_state_dict(clean_sd, strict=False)
            print(f"  -> {name_hint}: Loaded {loaded_count} layers. Missing: {len(msg.missing_keys)}", flush=True)
            return msg

        print("Loading global resident layers (Embeddings/Norms/Head)...", flush=True)
        globals_sd = torch.load(os.path.join(LAYERS_DIR, "globals.pt"), map_location="cpu")
        smart_load(model, globals_sd, "Globals")
        
        print(f"Loading layers and patching ({args.mode.upper()})...", flush=True)
        for i in range(NUM_LAYERS):
            if i % 4 == 0: print(f"  -> Applying Flash Logic: Decoder Block {i}/{NUM_LAYERS}...", flush=True)
            layer_sd = torch.load(os.path.join(LAYERS_DIR, f"layer_{i}.pt"), map_location="cpu")
            layer = model.model.decoder.layers[i] if hasattr(model.model, "decoder") else model.model.layers[i]
            
            if args.mode == "draft" and i % 4 == 0:
                layer.fc1 = ZeroModule(16384)
                layer.fc2 = nn.Identity()
                layer.activation_fn = nn.Identity()
            else:
                # Load Attention weights into the specific layer
                smart_load(layer, layer_sd, f"Block {i}")
                
                # Extract FFN biases for the engine
                # OPT keys: 'decoder.layers.0.fc1.bias' -> 'fc1.bias'
                fc1_b, fc2_b = None, None
                for k, v in layer_sd.items():
                    if "fc1.bias" in k: fc1_b = v.float().cpu().contiguous()
                    if "fc2.bias" in k: fc2_b = v.float().cpu().contiguous()
                
                layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
                layer.fc2 = nn.Identity(); layer.activation_fn = nn.Identity()
            del layer_sd; gc.collect()
            
        # Ensure NO parameters are left on 'meta' device or uninitialized
        print("Finalizing model materialization...", flush=True)
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                with torch.no_grad(): p.zero_()

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
    p.add_argument('--mode', choices=['predictor', 'oracle', 'naive', 'quantized', 'speculative_custom', 'draft'], default='predictor', help="Inference mode")
    p.add_argument('--predictor', type=str, help="Path to .bin predictor")
    p.add_argument('--ffn_bin', type=str, help="Path to bundled FFN binary")
    p.add_argument('--layers', type=str, help="Path to directory containing resident layers (.pt files)")
    p.add_argument('--cache', type=str, help="Path to HuggingFace cache")
    p.add_argument('--top_k', type=int, default=1024)
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--window', type=int, default=5)
    chat(p.parse_args())
