import os
import warnings
import logging
import ctypes
import glob
import torch
import torch.nn as nn
import sys
import time

# Configuration
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
DEVICE = "cuda:0"

# C++ Engine Bindings
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p]
lib.init_engine.restype = ctypes.c_void_p
lib.execute_ffn_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

class ManualFlashEngine:
    def __init__(self, mode="predictor"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
        self.engine_ptr = lib.init_engine(FFN_BIN_PATH)
        self.mode_int = {"predictor": 0, "oracle": 1, "naive": 2}[mode]
        
        print("Loading Model Weights...")
        snap_dir = sorted(glob.glob(os.path.join(CACHE_PATH, "models--facebook--opt-6.7b/snapshots/*")))[-1]
        shards = glob.glob(os.path.join(snap_dir, "pytorch_model-*.bin"))
        
        self.embed_tokens = None
        self.embed_positions = None
        self.final_ln_w = None
        self.final_ln_b = None
        self.layers = [{} for _ in range(32)]
        self.fc1_biases = [None]*32
        self.fc2_biases = [None]*32

        for shard in shards:
            print(f"  Reading shard {os.path.basename(shard)}...")
            sd = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
            for k, v in sd.items():
                if "fc1.weight" in k or "fc2.weight" in k: continue
                
                if "embed_tokens" in k: self.embed_tokens = v.to(torch.float16).to(DEVICE)
                elif "embed_positions" in k: self.embed_positions = v.to(torch.float16).to(DEVICE)
                elif "decoder.final_layer_norm.weight" in k: self.final_ln_w = v.to(torch.float16).to(DEVICE)
                elif "decoder.final_layer_norm.bias" in k: self.final_ln_b = v.to(torch.float16).to(DEVICE)
                elif "decoder.layers" in k:
                    parts = k.split(".")
                    l_idx = int(parts[2])
                    sub_k = ".".join(parts[3:])
                    if sub_k == "fc1.bias": self.fc1_biases[l_idx] = v.float().cpu().contiguous()
                    elif sub_k == "fc2.bias": self.fc2_biases[l_idx] = v.to(torch.float16).to(DEVICE)
                    else: self.layers[l_idx][sub_k] = v.to(torch.float16).to(DEVICE)
            del sd
        
        self.lm_head_w = self.embed_tokens 
        print("Model Loaded.")

    def ln(self, x, w, b):
        return nn.functional.layer_norm(x, (HIDDEN_SIZE,), w, b)

    def attn(self, x, weights):
        q = nn.functional.linear(x, weights["self_attn.q_proj.weight"], weights["self_attn.q_proj.bias"])
        k = nn.functional.linear(x, weights["self_attn.k_proj.weight"], weights["self_attn.k_proj.bias"])
        v = nn.functional.linear(x, weights["self_attn.v_proj.weight"], weights["self_attn.v_proj.bias"])
        # (Simplified Attn for speed verification)
        return nn.functional.linear(v, weights["self_attn.out_proj.weight"], weights["self_attn.out_proj.bias"])

    def ffn(self, x, l_idx):
        bsz, seq, dim = x.shape
        flat_x = x.view(-1, dim).float().cpu().contiguous()
        out_cpu = torch.zeros_like(flat_x)
        
        bias_ptr = ctypes.cast(self.fc1_biases[l_idx].data_ptr(), ctypes.POINTER(ctypes.c_float))
        lib.execute_ffn_layer(self.engine_ptr, l_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            flat_x.shape[0], bias_ptr, self.mode_int)
            
        res = out_cpu.to(DEVICE, dtype=torch.float16).view(bsz, seq, dim)
        return res + self.fc2_biases[l_idx]

    def forward(self, input_ids):
        x = self.embed_tokens[input_ids]
        pos = torch.arange(2, input_ids.shape[1] + 2, device=DEVICE)
        x = x + self.embed_positions[pos].unsqueeze(0)
        
        for i in range(32):
            residual = x
            x = self.ln(x, self.layers[i]["self_attn_layer_norm.weight"], self.layers[i]["self_attn_layer_norm.bias"])
            x = self.attn(x, self.layers[i])
            x = residual + x
            
            residual = x
            x = self.ln(x, self.layers[i]["final_layer_norm.weight"], self.layers[i]["final_layer_norm.bias"])
            x = self.ffn(x, i)
            x = residual + x
            
        x = self.ln(x, self.final_ln_w, self.final_ln_b)
        logits = nn.functional.linear(x, self.lm_head_w)
        return logits

    def generate(self, text):
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        print("Bot:", end=" ", flush=True)
        for _ in range(15):
            with torch.no_grad():
                logits = self.forward(ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1)
            word = self.tokenizer.decode(next_id, skip_special_tokens=True)
            print(word, end="", flush=True)
            if next_id.item() == 2: break
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
        print()

if __name__ == "__main__":
    engine = ManualFlashEngine()
    engine.generate("The capital of France is")
