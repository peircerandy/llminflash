import pickle
import zipfile
import os
import glob

shard = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/pytorch_model-00001-of-00002.bin"

with zipfile.ZipFile(shard, 'r') as z:
    with z.open('archive/data.pkl') as f:
        class SimpleUnpickler(pickle.Unpickler):
            def persistent_load(self, pid): return pid
            def find_class(self, module, name): return lambda *args: args

        metadata = SimpleUnpickler(f).load()

for k, v in metadata.items():
    if "fc1.weight" in k:
        print(f"Key: {k}")
        print(f"Storage Info: {v[0]}")
        print(f"Shape: {v[2]}")
        break
