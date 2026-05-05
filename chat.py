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
from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from accelerate.hooks import remove_hook_from_module

# --- 0. Setup & Silence ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
LAYERS_DIR = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_layers"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 1. C++ Engine Bindings ---
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
lib.init_engine.restype = ctypes.c_void_p
lib.set_engine_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
lib.execute_ffn_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
lib.destroy_engine.argtypes = [ctypes.c_void_p]

PREDICTOR_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_predictors.bin"

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

class StreamAndTimer:
    def __init__(self, tokenizer, mode_name):
        self.tokenizer = tokenizer
        self.mode_name = mode_name
        self.token_count = 0
        self.start_time = None
        self.prompt_skipped = False
        self.tokens = []
        self.print_len = 0
        self.full_text = ""
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None
    
    def start(self, skip_prompt=True):
        with self.lock:
            self.token_count = 0
            self.start_time = time.time()
            self.prompt_skipped = not skip_prompt
            self.tokens = []
            self.print_len = 0
            self.full_text = ""
            self.is_running = True
            
            # Make room for status line
            print(f"\n[{' '*10}] 0.0s | 0 tokens") 
            print(f"OPT [{self.mode_name.upper()}]: ", end="", flush=True)
            
            self.thread = threading.Thread(target=self._spin)
            self.thread.daemon = True
            self.thread.start()

    def _spin(self):
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while self.is_running:
            with self.lock:
                elapsed = time.time() - self.start_time
                cols = shutil.get_terminal_size((80, 20)).columns
                
                header = f"OPT [{self.mode_name.upper()}]: "
                content = header + self.full_text
                
                # Calculate how many lines the output has occupied
                lines = 0
                for part in content.split('\n'):
                    lines += (len(part) // cols) + 1
                
                status = f"[{spinner[idx]}] {elapsed:.1f}s | {self.token_count} tokens"
                # ANSI: Save Cursor, Up 'lines' lines, Carriage Return, Clear Line, Print Status, Restore Cursor
                sys.stdout.write(f"\033[s\033[{lines}A\r\033[K{status}\033[u")
                sys.stdout.flush()
                
            idx = (idx + 1) % 4
            time.sleep(0.1)
    
    def put(self, value):
        if not self.prompt_skipped:
            self.prompt_skipped = True
            return
        
        with self.lock:
            if torch.is_tensor(value):
                value = value.view(-1).tolist()
            elif isinstance(value, int):
                value = [value]
            
            self.token_count += len(value)
            self.tokens.extend(value)
            
            full_text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
            new_text = full_text[self.print_len:]
            self.full_text = full_text
            print(new_text, end="", flush=True)
            self.print_len = len(full_text)
    
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        elapsed = time.time() - self.start_time
        tps = self.token_count / elapsed if elapsed > 0 else 0
        spt = 1 / tps if tps > 0 else 0
        
        # Clear the spinner line one last time
        with self.lock:
            cols = shutil.get_terminal_size((80, 20)).columns
            header = f"OPT [{self.mode_name.upper()}]: "
            content = header + self.full_text
            lines = 0
            for part in content.split('\n'):
                lines += (len(part) // cols) + 1
            sys.stdout.write(f"\033[s\033[{lines}A\r\033[K[DONE] {elapsed:.1f}s | {self.token_count} tokens\033[u")
            sys.stdout.flush()

        print(f"\n\n" + "-"*30)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Tokens: {self.token_count}")
        print(f"Tokens per second: {tps:.2f}")
        print(f"Seconds per token: {spt:.4f}")
        print("-" * 30 + "\n")
        
        return elapsed, self.token_count
    
    def end(self):
        pass

def chat(args):
    mode_name = args.mode
    if not os.path.exists(FFN_BIN_PATH.decode()): 
        print(f"SSD weights not found!"); sys.exit(1)
    
    print(f"Initializing {mode_name.upper()} Mode (Layer-wise Loading)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
    engine_ptr = None; assistant_model = None

    # APPLE CORE: Initialize Flash Engine
    pred_path = PREDICTOR_BIN_PATH if os.path.exists(PREDICTOR_BIN_PATH.decode()) else b""
    engine_ptr = lib.init_engine(FFN_BIN_PATH, pred_path, ctypes.c_size_t(HIDDEN_SIZE), ctypes.c_size_t(16384), ctypes.c_size_t(NUM_LAYERS), ctypes.c_int(0))
    lib.set_engine_config(engine_ptr, args.top_k, args.threshold, args.window)
    
    mode_int = {"predictor": 0, "oracle": 1, "naive": 2, "speculative_hf": 0, "speculative_custom": 0}[mode_name]
    
    # 1. Initialize an empty model shell
    print("Constructing empty model shell...")
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
    with init_empty_weights():
        model = OPTForCausalLM(config)
        
    # 2. Sequential Loading from surgically extracted files
    print("Loading global weights (Embeddings, Norms)...")
    globals_sd = torch.load(os.path.join(LAYERS_DIR, "globals.pt"), map_location="cpu")
    for k, v in globals_sd.items():
        # Correctly map decoder keys to the base model attribute
        target = model.model if k.startswith("decoder.") else model
        set_module_tensor_to_device(target, k, DEVICE, value=v, dtype=torch.float16)
    del globals_sd; gc.collect()
    
    # Set LM head to embed_tokens
    set_module_tensor_to_device(model, "lm_head.weight", DEVICE, value=model.model.decoder.embed_tokens.weight, dtype=torch.float16)

    print("Loading layers and patching with Flash Engine...")
    for i in range(NUM_LAYERS):
        layer_file = os.path.join(LAYERS_DIR, f"layer_{i}.pt")
        if not os.path.exists(layer_file):
            print(f"  Warning: Missing {layer_file}!"); continue
            
        layer_sd = torch.load(layer_file, map_location="cpu")
        
        # Extract FFN biases for C++ engine
        fc1_b = layer_sd[f"decoder.layers.{i}.fc1.bias"].float().cpu().contiguous()
        fc2_b = layer_sd[f"decoder.layers.{i}.fc2.bias"].float().cpu().contiguous()
        
        # Load Attention/Norm weights to model
        layer = model.model.decoder.layers[i]
        for k, v in layer_sd.items():
            if ".bias" in k and ("fc1" in k or "fc2" in k): continue
            # Map k to relative path within layer
            rel_k = k.replace(f"decoder.layers.{i}.", "")
            set_module_tensor_to_device(layer, rel_k, DEVICE, value=v, dtype=torch.float16)
            
        # Patch the layer
        layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
        layer.fc2 = nn.Identity()
        layer.activation_fn = nn.Identity()
        
        del layer_sd; gc.collect()
        if i % 4 == 0: print(f"  Processed {i+1}/{NUM_LAYERS} layers...")

    if "speculative" in mode_name:
        print("Loading 125M Draft Model...")
        assistant_model = OPTForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", cache_dir=CACHE_PATH, local_files_only=True)
        assistant_model.eval()

    # --- Chat loop ---
    model.eval()
    print("\n" + "="*60 + f"\nLOCAL AI ASSISTANT READY ({mode_name.upper()})\n" + "="*60 + "\n")
    timer = StreamAndTimer(tokenizer, mode_name)
    history = "The following is a conversation between a Human and a highly intelligent AI Assistant.\n\n"

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["quit", "exit"]: break
            if not user_input.strip(): continue
            
            prompt = history + f"Human: {user_input}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            if mode_name == "speculative_custom":
                timer.start(skip_prompt=False)
                K = 4; cur_ids = inputs.input_ids; past_key_values = None
                with torch.no_grad():
                    outputs = model(cur_ids, use_cache=True)
                    past_key_values = outputs.past_key_values; last_logit = outputs.logits[:, -1, :]
                
                gen_len = 0
                while gen_len < 50:
                    draft_ids = cur_ids
                    for _ in range(K):
                        with torch.no_grad():
                            draft_ids = torch.cat([draft_ids, torch.argmax(assistant_model(draft_ids).logits[:, -1, :], dim=-1).unsqueeze(0)], dim=-1)
                    
                    with torch.no_grad():
                        main_out = model(draft_ids[:, -K:], past_key_values=past_key_values, use_cache=True)
                    
                    accepted_ids = []
                    for j in range(K):
                        d_tok = draft_ids[0, cur_ids.shape[1] + j]
                        m_tok = torch.argmax(last_logit if j == 0 else main_out.logits[0, j-1, :], dim=-1)
                        accepted_ids.append(m_tok); timer.put(m_tok); gen_len += 1
                        if d_tok != m_tok: break
                    
                    cur_ids = torch.cat([cur_ids, torch.tensor([accepted_ids], device=model.device)], dim=-1)
                    with torch.no_grad():
                        outputs = model(cur_ids[:, -len(accepted_ids):], past_key_values=past_key_values, use_cache=True)
                        past_key_values = outputs.past_key_values; last_logit = outputs.logits[:, -1, :]
                    if any(tid == tokenizer.eos_token_id for tid in accepted_ids): break
                timer.stop()
                history += f"Human: {user_input}\nAssistant: {tokenizer.decode(cur_ids[0][inputs.input_ids.shape[1]:])}\n"
            else:
                timer.start(skip_prompt=True)
                kwargs = {"max_new_tokens": 50, "pad_token_id": tokenizer.eos_token_id, "streamer": timer, "do_sample": False}
                if mode_name == "speculative_hf": kwargs["assistant_model"] = assistant_model
                out = model.generate(**inputs, **kwargs)
                timer.stop()
                history += f"Human: {user_input}\nAssistant: {tokenizer.decode(out[0][inputs.input_ids.shape[1]:])}\n"
        except KeyboardInterrupt: break
    
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['predictor', 'oracle', 'naive', 'standard', 'quantized', 'speculative_hf', 'speculative_custom'], default='predictor')
    p.add_argument('--top_k', type=int, default=1024)
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--window', type=int, default=5)
    chat(p.parse_args())
