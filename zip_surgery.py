import torch
import zipfile
import pickle
import os
import glob
from safetensors.torch import save_file
import numpy as np
import gc

# Configuration
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
OUT_FILE_1 = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_non_ffn_1.safetensors"
OUT_FILE_2 = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_non_ffn_2.safetensors"

def zip_surgery():
    print("Starting robust ZIP Surgery with sharding...")
    
    snap_dir = sorted(glob.glob(os.path.join(CACHE_PATH, "models--facebook--opt-6.7b/snapshots/*")))[-1]
    shards = sorted(glob.glob(os.path.join(snap_dir, "pytorch_model-*.bin")))
    
    # Process shards one by one and save immediately
    for i, shard in enumerate(shards):
        print(f"Surgically extracting from {os.path.basename(shard)}...")
        shard_state_dict = {}
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
                    shard_state_dict[key] = tensor
                except Exception as e:
                    print(f"  Failed {key}: {e}")
        
        out_file = OUT_FILE_1 if i == 0 else OUT_FILE_2
        print(f"Saving {len(shard_state_dict)} tensors to {out_file}...")
        save_file(shard_state_dict, out_file)
        del shard_state_dict
        gc.collect()
        
    print("Surgery Successful!")

if __name__ == "__main__":
    zip_surgery()
