import torch
import torch.nn as nn
import transformers
from transformers import OPTForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate.hooks import remove_hook_from_module
import ctypes
import os
import time
import gc
import logging
import argparse

# --- Suppress HF Warnings safely ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
transformers.logging.set_verbosity_error() 
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# --- Configuration ---
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
HIDDEN_SIZE = 4096
DEVICE = "cuda:0"

# --- C++ Engine Bindings ---
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p]
lib.init_engine.restype = ctypes.c_void_p
lib.execute_ffn_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
lib.destroy_engine.argtypes = [ctypes.c_void_p]

class FlashFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, fc1_bias, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.fc1_bias_c = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, HIDDEN_SIZE).float().cpu().contiguous()
        num_tokens = flat_x.shape[0]
        out_cpu = torch.zeros_like(flat_x)
        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            num_tokens, self.fc1_bias_c, self.mode_int)
        return out_cpu.to(x.device, dtype=x.dtype).view(*orig_shape)

TEST_QUESTIONS = [
    {"prompt": "Question: What is the capital of France?\nAnswer:", "expected": "Paris", "desc": "Basic Knowledge"},
    {"prompt": "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers.", "expected": "teach", "desc": "Overfit Recall"},
]

def generate_graph(results):
    print("\nGenerating Benchmark Graph...")
    try: import matplotlib.pyplot as plt
    except: 
        import pip; pip.main(['install', 'matplotlib']); import matplotlib.pyplot as plt
    modes = list(results.keys())
    speeds = [results[m]["avg_tps"] for m in modes]
    plt.figure(figsize=(12, 8))
    colors = ['#ffcc99', '#99ccff', '#ff9999', '#c2c2f0', '#b3e2cd', '#fdcdac']
    bars = plt.bar([m.upper() for m in modes], speeds, color=colors[:len(modes)])
    plt.title('Hardware Architecture Performance vs. Accuracy (OPT-6.7B)', fontweight='bold')
    plt.ylabel('Tokens Per Second')
    plt.ylim(0, max(speeds) * 1.5)
    base = results.get("naive", {}).get("avg_tps", 0)
    for bar, m in zip(bars, modes):
        y = bar.get_height()
        speedup = f"\n({y/base:.2f}x)" if base > 0 and m != "naive" else ""
        plt.text(bar.get_x()+bar.get_width()/2, y, f"{y:.2f} TPS{speedup}\nAcc: {results[m]['accuracy_pct']}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout(); plt.savefig('presentation_benchmark.png', dpi=300)

def run_suite():
    if not os.path.exists(FFN_BIN_PATH.decode()): print("SSD not found"); return
    modes = ["naive", "oracle", "predictor", "speculative_hf", "speculative_custom"]
    benchmark_results = {}
    for mode_name in modes:
        print(f"\n--- LOADING {mode_name.upper()} ---")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
        engine_ptr = lib.init_engine(FFN_BIN_PATH)
        mode_int = {"predictor": 0, "oracle": 1, "naive": 2, "speculative_hf": 0, "speculative_custom": 0}[mode_name]
        assistant = None
        if "speculative" in mode_name: assistant = OPTForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", cache_dir=CACHE_PATH, local_files_only=True)
        custom_map = {"model.decoder.embed_tokens": DEVICE, "model.decoder.embed_positions": DEVICE, "model.decoder.final_layer_norm": DEVICE, "lm_head": DEVICE}
        for i in range(32):
            p = f"model.decoder.layers.{i}"
            custom_map[f"{p}.self_attn"] = DEVICE; custom_map[f"{p}.self_attn_layer_norm"] = DEVICE; custom_map[f"{p}.final_layer_norm"] = DEVICE; custom_map[f"{p}.fc1"] = "disk"; custom_map[f"{p}.fc2"] = "disk"
        model = OPTForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=custom_map, cache_dir=CACHE_PATH, offload_folder=OFFLOAD_PATH, local_files_only=True)
        with torch.no_grad():
            for i, layer in enumerate(model.model.decoder.layers):
                z = torch.zeros(1, HIDDEN_SIZE, dtype=model.dtype, device=model.device)
                fc1_b = layer.fc1(z).squeeze().float().cpu().contiguous()
                z2 = torch.zeros(1, 16384, dtype=model.dtype, device=model.device)
                fc2_b = layer.fc2(z2).squeeze().to(model.device)
                layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, mode_int)
                layer.fc2 = nn.Linear(16384, 4096, bias=True); layer.fc2.weight.data.zero_(); layer.fc2.bias.data.copy_(fc2_b)
                layer.activation_fn = nn.Identity(); remove_hook_from_module(layer, recurse=True)
        
        mode_tps, mode_passes = 0, 0
        for q in TEST_QUESTIONS:
            inputs = tokenizer(q["prompt"], return_tensors="pt").to(model.device)
            start = time.time()
            if mode_name == "speculative_custom":
                K, cur_ids, gen_ids = 4, inputs.input_ids, []
                while len(gen_ids) < 10:
                    draft_ids = cur_ids
                    for _ in range(K):
                        draft_ids = torch.cat([draft_ids, torch.argmax(assistant(draft_ids).logits[:, -1, :], dim=-1).unsqueeze(0)], dim=-1)
                    main_logits = model(draft_ids).logits
                    n_orig, accepted = cur_ids.shape[1], []
                    for j in range(K):
                        d_tok, m_tok = draft_ids[0, n_orig+j], torch.argmax(main_logits[0, n_orig+j-1, :], dim=-1)
                        accepted.append(m_tok)
                        if d_tok != m_tok: break
                    gen_ids.extend(accepted)
                    cur_ids = torch.cat([cur_ids, torch.tensor([accepted], device=DEVICE)], dim=-1)
                    if any(tid == tokenizer.eos_token_id for tid in accepted): break
                gen_ids = torch.tensor(gen_ids[:10], device=DEVICE)
            else:
                gen_kwargs = {"max_new_tokens": 10, "pad_token_id": tokenizer.eos_token_id, "do_sample": False}
                if mode_name == "speculative_hf": gen_kwargs["assistant_model"] = assistant
                outputs = model.generate(**inputs, **gen_kwargs)
                gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            elapsed = time.time() - start
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            print(f"OPT [{mode_name.upper()}]: {text}")
            mode_tps += 10/elapsed
            if q["expected"].lower() in text.lower(): mode_passes += 1
        benchmark_results[mode_name] = {"avg_tps": mode_tps/len(TEST_QUESTIONS), "accuracy_pct": int(mode_passes/len(TEST_QUESTIONS)*100)}
        del model; gc.collect(); torch.cuda.empty_cache()
    generate_graph(benchmark_results)

if __name__ == "__main__": run_suite()
