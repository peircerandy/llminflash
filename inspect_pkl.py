import pickle
import zipfile
import os
import glob

shard = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/pytorch_model-00001-of-00002.bin"

with zipfile.ZipFile(shard, 'r') as z:
    with z.open('archive/data.pkl') as f:
        # Custom unpickler to avoid importing torch
        class SimpleUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                return pid # Just return the persistent id
            def find_class(self, module, name):
                if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                    return lambda *args: args
                return lambda *args: args

        metadata = SimpleUnpickler(f).load()

# Print first 5 items to understand the structure
for i, (k, v) in enumerate(metadata.items()):
    print(f"Key: {k}")
    print(f"Val: {v}")
    if i >= 5: break
