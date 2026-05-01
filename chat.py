import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate.hooks import remove_hook_from_module
import ctypes
import os
import sys
import time
import threading
import gc
import argparse

# --- Force HuggingFace Offline ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# --- Configuration ---
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"
FFN_BIN_PATH = b"/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
HIDDEN_SIZE = 4096

# --- C++ Engine Bindings ---
lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
lib.init_engine.argtypes = [ctypes.c_char_p]
lib.init_engine.restype = ctypes.c_void_p

lib.execute_ffn_layer.argtypes = [
    ctypes.c_void_p, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int # NEW: Execution Mode
]
lib.destroy_engine.argtypes = [ctypes.c_void_p]

# --- The "Monkey Patch" Module ---
class FlashFFN(nn.Module):
    def __init__(self, layer_idx, engine_ptr, fc1_bias, fc2_bias, mode_int):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.mode_int = mode_int
        self.fc1_bias = fc1_bias
        self.fc1_bias_c = ctypes.cast(self.fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fc1_bias is not None else None
        self.fc2_bias = fc2_bias

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        device = hidden_states.device
        
        hidden_flat = hidden_states.view(-1, HIDDEN_SIZE).float().cpu().contiguous()
        num_tokens = hidden_flat.shape[0]

        out_flat = torch.zeros_like(hidden_flat)

        in_ptr = hidden_flat.data_ptr()
        out_ptr = out_flat.data_ptr()

        in_c = ctypes.cast(in_ptr, ctypes.POINTER(ctypes.c_float))
        out_c = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float))

        # Call the SSD Streaming C++ Engine with our specific flag!
        lib.execute_ffn_layer(self.engine_ptr, self.layer_idx, in_c, out_c, num_tokens, self.fc1_bias_c, self.mode_int)

        out_tensor = out_flat.to(device).to(hidden_states.dtype).view(orig_shape)
        if self.fc2_bias is not None:
            out_tensor = out_tensor + self.fc2_bias
            
        return out_tensor

class StreamAndTimer:
    def __init__(self, tokenizer, mode_name):
        self.tokenizer = tokenizer
        self.mode_name = mode_name
        self.is_running = False
        self.start_time = 0
        self.generated_text = ""
        self.prompt_processed = False
        self.token_count = 0

    def start(self):
        self.is_running = True
        self.start_time = time.time()
        self.generated_text = ""
        self.prompt_processed = False
        self.token_count = 0
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True # CRITICAL FIX: Kills the spinner if the main program crashes!
        self.thread.start()

    def _spin(self):
        spinner_chars = ['|', '/', '-', '\\']
        idx = 0
        while self.is_running:
            elapsed = time.time() - self.start_time
            
            # VISUAL FIX: Replace actual newlines with a 'return' symbol. 
            # This keeps everything safely on one line while letting you see EXACTLY what it's generating!
            display_text = self.generated_text.replace('\n', ' ↵ ')
            
            sys.stdout.write(f"\r\033[KOPT [{self.mode_name.upper()}]: {display_text} {spinner_chars[idx]} {elapsed:.1f}s")
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner_chars)
            time.sleep(0.1)

    def stop(self):
        if not self.is_running:
            return 0, 0
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time
        
        # Erase the spinner line completely
        sys.stdout.write(f"\r\033[K")
        # Print the FINAL multi-line text exactly as the model formatted it
        sys.stdout.write(f"OPT [{self.mode_name.upper()}]: {self.generated_text}\n")
        sys.stdout.flush()
        return elapsed, self.token_count

    def put(self, value):
        try:
            # Hugging Face passes the prompt as the first put() call. We skip it so we only see new text.
            if not self.prompt_processed:
                self.prompt_processed = True
                return
            
            # CRITICAL FIX: Flatten any tensor shape into a 1D list so the tokenizer never crashes
            if torch.is_tensor(value):
                value = value.view(-1).tolist()
            elif isinstance(value, int):
                value = [value]
                
            self.token_count += len(value) # NEW: Count tokens exactly as they come in!
                
            text = self.tokenizer.decode(value, skip_special_tokens=True)
            self.generated_text += text
        except Exception:
            pass

    def end(self):
        pass

def chat(mode_name):
    print(f"Initializing {mode_name.upper()} Mode...")
    
    if mode_name == "quantized":
        # THE GPT4ALL METHOD: Shrink weights to 4-bit to fit perfectly in RAM
        print("Loading 4-bit Quantized Model (BitsAndBytes)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
        
        # MODERN HUGGINGFACE API FIX: Use a dedicated configuration object
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        
        model = OPTForCausalLM.from_pretrained(
            MODEL_ID, 
            quantization_config=bnb_config, # <--- The updated 4-bit config
            device_map="auto",
            cache_dir=CACHE_PATH,
            local_files_only=True
        )
        engine_ptr = None

    elif mode_name == "standard":
        # STANDARD MODE: No C++, no tricks. Just pure PyTorch memory limits.
        print("Loading Vanilla Hugging Face (Warning: This will use Python file swapping)")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
        model = OPTForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="auto", # Default HF behavior
            cache_dir=CACHE_PATH,
            offload_folder=OFFLOAD_PATH,
            local_files_only=True
        )
        engine_ptr = None
        
    else:
        # APPLE ARCHITECTURES: Oracle, Predictor, or Naive
        engine_ptr = lib.init_engine(FFN_BIN_PATH)
        mode_map = {"predictor": 0, "oracle": 1, "naive": 2}
        mode_int = mode_map[mode_name]

        custom_device_map = {
            "model.decoder.embed_tokens": "cuda:0",
            "model.decoder.embed_positions": "cuda:0",
            "model.decoder.final_layer_norm": "cuda:0",
            "lm_head": "cuda:0"
        }
        for i in range(32):
            prefix = f"model.decoder.layers.{i}"
            custom_device_map[f"{prefix}.self_attn"] = "cuda:0"
            custom_device_map[f"{prefix}.self_attn_layer_norm"] = "cuda:0"
            custom_device_map[f"{prefix}.final_layer_norm"] = "cuda:0"
            custom_device_map[f"{prefix}.fc1"] = "disk"
            custom_device_map[f"{prefix}.fc2"] = "disk"
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH, local_files_only=True)
        model = OPTForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map=custom_device_map,
            cache_dir=CACHE_PATH, offload_folder=OFFLOAD_PATH, local_files_only=True
        )

        print("Extracting Biases, Stripping Ghost Hooks, and Monkey-patching...")
        with torch.no_grad():
            for i, layer in enumerate(model.model.decoder.layers):
                zero_fc1 = torch.zeros(1, HIDDEN_SIZE, dtype=model.dtype, device=model.device)
                fc1_bias = layer.fc1(zero_fc1).squeeze().float().cpu().contiguous()
                zero_fc2 = torch.zeros(1, 16384, dtype=model.dtype, device=model.device)
                fc2_bias = layer.fc2(zero_fc2).squeeze().to(model.device)
                
                # Pass the specific architecture mode to the C++ Patch
                layer.fc1 = FlashFFN(i, engine_ptr, fc1_bias, fc2_bias, mode_int)
                layer.activation_fn = nn.Identity()
                layer.fc2 = nn.Identity()
                remove_hook_from_module(layer, recurse=True)

        print("Purging unused weights from system RAM...")
        gc.collect()

    print("\n" + "="*50)
    print(f"LLM IN A FLASH - [{mode_name.upper()}] READY")
    print("="*50 + "\n")

    # Initialize the Hybrid Live Text Streamer
    timer_streamer = StreamAndTimer(tokenizer, mode_name)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
            
            timer_streamer.start()
            
            try:
                # We pass the streamer directly into the generate function!
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=15, 
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id, # REVERTED: Let the model stop on its natural <EOS> token!
                    streamer=timer_streamer # Streams live text AND the spinning timer
                )
                elapsed_time, gen_tokens = timer_streamer.stop()
                
            except KeyboardInterrupt:
                # Graceful interrupt catches early stops AND retrieves current metrics
                elapsed_time, gen_tokens = timer_streamer.stop()
                print("\n[Generation stopped early by user]")
                
            except Exception as e:
                timer_streamer.stop()
                print(f"\n[Generation Error]: HuggingFace crashed with error: {e}\n")
                continue
            
            # Calculate exactly how fast the hardware ran, even if interrupted!
            if elapsed_time > 0 and gen_tokens > 0:
                tps = gen_tokens / elapsed_time
                sec_per_token = elapsed_time / gen_tokens
                print(f"\n[Generated {gen_tokens} tokens in {elapsed_time:.1f}s | Speed: {tps:.2f} tokens/sec | {sec_per_token:.1f} sec/token]\n")
            else:
                print(f"\n[Generation stopped before tokens were produced]\n")
            
        except KeyboardInterrupt:
            # Safely stop the timer if user hits Ctrl+C
            timer_streamer.stop()
            break

    print("\nShutting down engine...")
    if engine_ptr:
        lib.destroy_engine(engine_ptr)

if __name__ == "__main__":
    # Custom help formatter to make the choices obvious
    parser = argparse.ArgumentParser(
        description="LLM in a Flash Runner",
        epilog="Available Modes:\n"
               "  predictor : Apple's ML Sparse Streaming (Fast, 16-bit)\n"
               "  oracle    : C++ Sparse Streaming with Exact Math (Accurate, slow)\n"
               "  naive     : C++ reading 100% of SSD (Apple's Baseline)\n"
               "  standard  : Vanilla Hugging Face (Python disk swapping)\n"
               "  quantized : 4-Bit RAM Compression (GPT4All style)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    valid_modes = ['predictor', 'oracle', 'naive', 'standard', 'quantized']
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=valid_modes, 
        default='oracle',
        help=f"Select the hardware architecture to test. Must be one of: {', '.join(valid_modes)}"
    )
    
    # If the user passes an invalid argument, argparse handles it automatically 
    # and explicitly lists the valid choices defined above!
    args = parser.parse_args()
    chat(args.mode)