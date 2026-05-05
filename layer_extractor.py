import torch
import zipfile
import pickle
import os
import glob
import numpy as np
import gc

# Configuration
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OUT_DIR = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_layers"
os.makedirs(OUT_DIR, exist_ok=True)

def layer_wise_extraction():
    print(f"Starting Layer-wise extraction to {OUT_DIR}...")
    
    snap_dir = sorted(glob.glob(os.path.join(CACHE_PATH, "models--facebook--opt-6.7b/snapshots/*")))[-1]
    shards = sorted(glob.glob(os.path.join(snap_dir, "pytorch_model-*.bin")))
    
    # Store global weights (embeddings, etc.) separately
    globals_sd = {}
    
    for shard in shards:
        print(f"Processing shard {os.path.basename(shard)}...")
        with zipfile.ZipFile(shard, 'r') as z:
            with z.open('archive/data.pkl') as f:
                class SimpleUnpickler(pickle.Unpickler):
                    def persistent_load(self, pid): return pid
                    def find_class(self, module, name): return lambda *args: args
                metadata = SimpleUnpickler(f).load()
            
            for key, val in metadata.items():
                if "fc1.weight" in key or "fc2.weight" in key:
                    continue
                
                try:
                    storage_info = val[0]
                    file_idx = storage_info[2]
                    data_file = f"archive/data/{file_idx}"
                    
                    with z.open(data_file) as df:
                        raw_data = df.read()
                    
                    shape = val[2]
                    num_elements = np.prod(shape)
                    np_dtype = np.float16 if len(raw_data) == num_elements * 2 else np.float32
                    
                    tensor = torch.from_numpy(np.frombuffer(raw_data, dtype=np_dtype).copy()).view(shape)
                    
                    if "decoder.layers." in key:
                        parts = key.split(".")
                        l_idx = parts[2]
                        layer_file = os.path.join(OUT_DIR, f"layer_{l_idx}.pt")
                        
                        # Load existing or create new
                        if os.path.exists(layer_file):
                            layer_sd = torch.load(layer_file)
                        else:
                            layer_sd = {}
                        
                        layer_sd[key] = tensor
                        torch.save(layer_sd, layer_file)
                        del layer_sd
                    else:
                        globals_sd[key] = tensor
                except Exception as e:
                    print(f"  Failed {key}: {e}")
        gc.collect()

    print(f"Saving global weights...")
    torch.save(globals_sd, os.path.join(OUT_DIR, "globals.pt"))
    print("Layer-wise extraction complete!")

if __name__ == "__main__":
    layer_wise_extraction()
