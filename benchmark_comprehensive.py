"""
benchmark_comprehensive.py: Rigorous multi-model benchmarking for LLM-in-a-Flash.
"""

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import ctypes
import os
import time
import gc
import pandas as pd
import argparse

# --- Configuration ---
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LAYERS_DIR_OPT = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_layers"
LAYERS_DIR_LLAMA = "/mnt/wsl/PHYSICALDRIVE0p3/llama3_8b_layers"

MODELS = {
    "opt-6.7b": {
        "id": "facebook/opt-6.7b",
        "ffn_bin": b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin",
        "layers_dir": LAYERS_DIR_OPT,
        "pred_bin": b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_predictors.bin",
        "hidden": 4096, "ffn": 16384, "layers": 32, "is_llama": False,
        "draft_id": "facebook/opt-125m"
    },
    "llama3-8b": {
        "id": "unsloth/llama-3-8b",
        "ffn_bin": b"/mnt/wsl/PHYSICALDRIVE0p3/llama3_bundled_ffn.bin",
        "layers_dir": LAYERS_DIR_LLAMA,
        "pred_bin": b"/mnt/wsl/PHYSICALDRIVE0p3/llama3_predictors.bin",
        "hidden": 4096, "ffn": 14336, "layers": 32, "is_llama": True,
        "draft_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }
}

TEST_QUESTIONS = [
    {"prompt": "Question: What is the capital of France? Answer:", "expected": "Paris"},
    {"prompt": "Question: The first president of the United States was George... Answer:", "expected": "Washington"},
]

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
    def __init__(self, layer_idx, engine_ptr, fc1_bias, hidden_size, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.hidden_size = hidden_size
        self.fc1_bias_c = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, self.hidden_size).float().cpu().contiguous()
        num_tokens = flat_x.shape[0]
        out_cpu = torch.zeros_like(flat_x)
        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            num_tokens, self.fc1_bias_c, self.mode_int)
        return out_cpu.to(x.device, dtype=x.dtype).view(*orig_shape)

class DotStreamer:
    def put(self, value): print(".", end="", flush=True)
    def end(self): print(" [OK]")

def load_flash_model(model_key, mode_name, top_k=1024, threshold=0.2):
    m = MODELS[model_key]
    print(f"Loading {model_key.upper()} in {mode_name.upper()} mode...")
    engine_ptr = lib.init_engine(m["ffn_bin"], m["pred_bin"], m["hidden"], m["ffn"], m["layers"], int(m["is_llama"]))
    lib.set_engine_config(engine_ptr, top_k, threshold, 5)
    mode_int = {"predictor": 0, "speculative": 0}[mode_name]
    
    config = AutoConfig.from_pretrained(m["id"], cache_dir=CACHE_PATH)
    with init_empty_weights(): model = AutoModelForCausalLM.from_config(config)
    
    globals_sd = torch.load(os.path.join(m["layers_dir"], "globals.pt"), map_location="cpu")
    for k, v in globals_sd.items():
        if not m["is_llama"] and k.startswith("decoder."):
            set_module_tensor_to_device(model.model, k, DEVICE, value=v, dtype=torch.float16)
        else:
            set_module_tensor_to_device(model, k, DEVICE, value=v, dtype=torch.float16)
    del globals_sd; gc.collect()
    
    for i in range(m["layers"]):
        layer_file = os.path.join(m["layers_dir"], f"layer_{i}.pt")
        layer_sd = torch.load(layer_file, map_location="cpu")
        layer = (model.model.layers if m["is_llama"] else model.model.decoder.layers)[i]
        
        fc1_b = None
        fc2_b = None
        if not m["is_llama"]:
            fc1_b = layer_sd[f"decoder.layers.{i}.fc1.bias"].float().cpu().contiguous()
            fc2_b = layer_sd[f"decoder.layers.{i}.fc2.bias"].to(DEVICE).to(torch.float16)
            
        for k, v in layer_sd.items():
            rel_k = k.replace(f"model.layers.{i}." if m["is_llama"] else f"decoder.layers.{i}.", "")
            if not m["is_llama"] and ".bias" in k and ("fc1" in k or "fc2" in k): continue
            set_module_tensor_to_device(layer, rel_k, DEVICE, value=v, dtype=torch.float16)
            
        if m["is_llama"]:
            layer.mlp = FlashFFN(i, engine_ptr, None, m["hidden"], mode_int)
        else:
            layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, m["hidden"], mode_int)
            layer.fc2 = nn.Identity(); layer.activation_fn = nn.Identity()
            # Wrap forward properly
            def make_fwd(f1, b2, lyr):
                def fwd(hidden_states, *args, **kwargs):
                    residual = hidden_states
                    hidden_states = lyr.self_attn_layer_norm(hidden_states)
                    attn_outputs = lyr.self_attn(hidden_states, *args, **kwargs)
                    hidden_states = attn_outputs[0]
                    hidden_states = residual + hidden_states
                    residual = hidden_states
                    hidden_states = lyr.final_layer_norm(hidden_states)
                    hidden_states = f1(hidden_states) + b2
                    hidden_states = residual + hidden_states
                    return (hidden_states,) + attn_outputs[1:]
                return fwd
            layer.forward = make_fwd(layer.fc1, fc2_b, layer)
        del layer_sd; gc.collect()
    return model, engine_ptr

def run_benchmark():
    parser = argparse.ArgumentParser(description="Multi-model benchmark for LLM in a Flash.")
    parser.add_argument('--model', choices=['opt-6.7b', 'llama3-8b'], default='llama3-8b')
    parser.add_argument('--skip_quantized', action='store_true')
    parser.add_argument('--tokens', type=int, default=10)
    args_cli = parser.parse_args()
    
    results = []
    m = MODELS[args_cli.model]
    tokenizer = AutoTokenizer.from_pretrained(m["id"], cache_dir=CACHE_PATH)
    modes = ["quantized", "predictor", "speculative"]
    if args_cli.skip_quantized: modes = ["predictor", "speculative"]
    
    streamer = DotStreamer()
    for mode in modes:
        assistant = None
        if mode == "quantized":
            print(f"Loading {args_cli.model} in QUANTIZED mode...")
            q_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
            try:
                max_mem = {0: "1GiB", "cpu": "30GiB"} if args_cli.model == "llama3-8b" else None
                model = AutoModelForCausalLM.from_pretrained(m["id"], quantization_config=q_config, device_map="auto", max_memory=max_mem, cache_dir=CACHE_PATH)
                engine_ptr = None
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({"Model": args_cli.model, "Mode": mode, "TPS": 0.0, "Accuracy": 0.0})
                continue
        elif mode == "speculative":
            model, engine_ptr = load_flash_model(args_cli.model, "speculative")
            assistant = AutoModelForCausalLM.from_pretrained(m["draft_id"], device_map="auto", cache_dir=CACHE_PATH)
        else: # Predictor
            model, engine_ptr = load_flash_model(args_cli.model, "predictor")
            
        model.eval()
        total_tps = 0; total_acc = 0
        for q in TEST_QUESTIONS:
            inputs = tokenizer(q["prompt"], return_tensors="pt").to(DEVICE)
            print(f"[{mode.upper()}] Generating", end="")
            start = time.time()
            with torch.no_grad():
                gen_kwargs = {"max_new_tokens": args_cli.tokens, "do_sample": False, "streamer": streamer}
                if assistant: gen_kwargs["assistant_model"] = assistant
                out = model.generate(**inputs, **gen_kwargs)
            elapsed = time.time() - start
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            tps = args_cli.tokens / elapsed
            total_tps += tps
            if q["expected"].lower() in text.lower(): total_acc += 1
            print(f" -> TPS: {tps:.2f}")

        results.append({
            "Model": args_cli.model, "Mode": mode, 
            "TPS": total_tps / len(TEST_QUESTIONS), 
            "Accuracy": (total_acc / len(TEST_QUESTIONS)) * 100
        })
        
        del model
        if assistant: del assistant
        if engine_ptr: lib.destroy_engine(engine_ptr)
        gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{args_cli.model}_comparison.csv", index=False)
    print("\nRESULTS:\n", df)

if __name__ == "__main__":
    run_benchmark()
