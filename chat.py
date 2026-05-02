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
from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from accelerate.hooks import remove_hook_from_module

# --- Force HuggingFace Offline ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Configuration ---
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
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
    def __init__(self, layer_idx, engine_ptr, fc1_bias, fc2_bias, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.fc1_bias_c = ctypes.cast(fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None
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
            
        res = out_cpu.to(DEVICE, dtype=torch.float16).view(*orig_shape)
        # NaN / Inf safety gate
        if torch.isnan(res).any() or torch.isinf(res).any():
            res = torch.nan_to_num(res, 0.0)
            
        return res + self.fc2_bias

class StreamAndTimer:
    def __init__(self, tokenizer, mode_name):
        self.tokenizer = tokenizer
        self.mode_name = mode_name
        self.is_running = False
        self.generated_text = ""
        self.token_count = 0
        self.prompt_skipped = False
    def start(self):
        self.is_running = True; self.generated_text = ""; self.token_count = 0; self.prompt_skipped = False; self.start_time = time.time(); self.thread = threading.Thread(target=self._spin); self.thread.daemon = True; self.thread.start()
    def _spin(self):
        spinner = ['|', '/', '-', '\\']; idx = 0
        while self.is_running:
            elapsed = time.time() - self.start_time
            sys.stdout.write(f"\r\033[KOPT [{self.mode_name.upper()}]: {self.generated_text.replace(chr(10), ' ')} {spinner[idx]} {elapsed:.1f}s")
            sys.stdout.flush(); idx = (idx + 1) % 4; time.sleep(0.1)
    def stop(self):
        self.is_running = False
        if self.thread.is_alive(): self.thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time
        sys.stdout.write(f"\r\033[KOPT [{self.mode_name.upper()}]: {self.generated_text}\n")
        return elapsed, self.token_count
    def put(self, value):
        if not self.prompt_skipped:
            self.prompt_skipped = True
            return
        if torch.is_tensor(value): value = value.view(-1).tolist()
        elif isinstance(value, int): value = [value]
        self.token_count += len(value); self.generated_text += self.tokenizer.decode(value, skip_special_tokens=True)
    def end(self):
        pass

def chat(mode_name):
    if not os.path.exists(FFN_BIN_PATH.decode()): print(f"SSD weights not found"); sys.exit(1)
    
    print(f"Initializing {mode_name.upper()} Mode...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
    engine_ptr = lib.init_engine(FFN_BIN_PATH)
    assistant_model = None

    if mode_name == "quantized":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = OPTForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", cache_dir=CACHE_PATH, local_files_only=True)
    elif mode_name == "standard":
        model = OPTForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", cache_dir=CACHE_PATH, offload_folder=OFFLOAD_PATH, local_files_only=True)
    else:
        mode_int = {"predictor": 0, "oracle": 1, "naive": 2, "speculative_hf": 0, "speculative_custom": 0}[mode_name]
        if "speculative" in mode_name:
            print("Loading 125M Draft Model...")
            assistant_model = OPTForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", cache_dir=CACHE_PATH, local_files_only=True)
            assistant_model.eval()

        print("Initializing Model Shell...")
        config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
        with init_empty_weights():
            model = OPTForCausalLM(config)
            
        print("Loading weights to GPU...")
        snap_dir = sorted(glob.glob(os.path.join(CACHE_PATH, "models--facebook--opt-6.7b/snapshots/*")))[-1]
        shards = glob.glob(os.path.join(snap_dir, "pytorch_model-*.bin"))
        
        biases = {}
        for shard in shards:
            sd = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
            for k, v in sd.items():
                if "fc1.weight" in k or "fc2.weight" in k: continue
                if ".bias" in k and ("fc1" in k or "fc2" in k):
                    biases[k] = v.float().cpu().contiguous()
                    continue
                target = model.model if k.startswith("decoder.") else model
                set_module_tensor_to_device(target, k, DEVICE, value=v, dtype=torch.float16)
            del sd
        
        set_module_tensor_to_device(model, "lm_head.weight", DEVICE, value=model.model.decoder.embed_tokens.weight, dtype=torch.float16)
        model.eval()

        print("Patching layers...")
        for i, layer in enumerate(model.model.decoder.layers):
            fc1_b = biases[f"decoder.layers.{i}.fc1.bias"]
            fc2_b = biases[f"decoder.layers.{i}.fc2.bias"]
            layer.fc1 = FlashFFN(i, engine_ptr, fc1_b, fc2_b, mode_int)
            layer.fc2 = nn.Identity()
            layer.activation_fn = nn.Identity()
            remove_hook_from_module(layer, recurse=True)
        gc.collect()

    model.eval()
    print("\n" + "="*60 + f"\nREADY (OPT-6.7B)\n" + "="*60 + "\n")
    timer = StreamAndTimer(tokenizer, mode_name)
    history = "The following is a conversation between a Human and a highly intelligent AI Assistant.\n\n"

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]: break
            
            prompt = history + f"Human: {user_input}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            timer.start()
            
            if mode_name == "speculative_custom":
                K = 4; cur_ids = inputs.input_ids; past_key_values = None
                with torch.no_grad():
                    outputs = model(cur_ids, use_cache=True); past_key_values = outputs.past_key_values; last_logit = outputs.logits[:, -1, :]
                
                gen_len = 0; bot_text = ""
                while gen_len < 50:
                    draft_ids = cur_ids
                    for _ in range(K):
                        with torch.no_grad():
                            draft_ids = torch.cat([draft_ids, torch.argmax(assistant_model(draft_ids).logits[:, -1, :], dim=-1).unsqueeze(0)], dim=-1)
                    verification_ids = draft_ids[:, -K:]
                    with torch.no_grad():
                        main_out = model(verification_ids, past_key_values=past_key_values, use_cache=True)
                        main_logits = main_out.logits
                    accepted_ids = []
                    for j in range(K):
                        d_tok = draft_ids[0, cur_ids.shape[1] + j]
                        m_tok = torch.argmax(last_logit if j == 0 else main_logits[0, j-1, :], dim=-1)
                        accepted_ids.append(m_tok); timer.put(m_tok); gen_len += 1
                        if d_tok != m_tok: break
                    accepted_text = tokenizer.decode(accepted_ids, skip_special_tokens=True)
                    bot_text += accepted_text
                    cur_ids = torch.cat([cur_ids, torch.tensor([accepted_ids], device=DEVICE)], dim=-1)
                    with torch.no_grad():
                        outputs = model(cur_ids, use_cache=True); past_key_values = outputs.past_key_values; last_logit = outputs.logits[:, -1, :]
                    if any(tid == tokenizer.eos_token_id for tid in accepted_ids) or "Human:" in accepted_text: break
                timer.stop()
                history += f"Human: {user_input}\nAssistant: {bot_text}\n"
            else:
                kwargs = {
                    "max_new_tokens": 50, 
                    "pad_token_id": tokenizer.eos_token_id, 
                    "streamer": timer, 
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.7
                }
                if mode_name == "speculative_hf": kwargs["assistant_model"] = assistant_model
                
                out = model.generate(**inputs, **kwargs)
                # Ensure history is updated correctly
                bot_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                timer.stop()
                history += f"Human: {user_input}\nAssistant: {bot_text}\n"
        except KeyboardInterrupt: break
    if engine_ptr: lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['predictor', 'oracle', 'naive', 'standard', 'quantized', 'speculative_hf', 'speculative_custom'], default='oracle')
    chat(p.parse_args().mode)
