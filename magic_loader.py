import torch
import io
import pickle
import os

class SelectiveUnpickler(pickle.Unpickler):
    def __init__(self, file, excluded_keys):
        super().__init__(file)
        self.excluded_keys = excluded_keys

    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return self.rebuild_tensor
        return super().find_class(module, name)

    def rebuild_tensor(self, storage, storage_offset, size, stride, requires_grad, backward_hooks):
        # We don't know the key here yet, but we can check the size
        # OPT-6.7B FFN weights are [16384, 4096] or [4096, 16384]
        # Size = 16384 * 4096 = 67,108,864 elements
        if size == (16384, 4096) or size == (4096, 16384):
            # Return a dummy small tensor instead of the massive one
            return torch.zeros(1, dtype=torch.float16)
        return torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)

def selective_load(path, excluded_keys):
    # This is a hacky way to use a custom unpickler with torch.load
    # Note: torch.load uses a more complex format for zip-based saves (.bin)
    # For .bin (zip), we need to extract the data file first.
    import zipfile
    with zipfile.ZipFile(path, 'r') as z:
        with z.open('archive/data.pkl') as f:
            unpickler = SelectiveUnpickler(f, excluded_keys)
            return unpickler.load()

# Test the concept
if __name__ == "__main__":
    shard = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/pytorch_model-00001-of-00002.bin"
    if os.path.exists(shard):
        print(f"Testing selective load on {shard}...")
        try:
            sd = selective_load(shard, ["fc1.weight", "fc2.weight"])
            print(f"Loaded {len(sd)} keys.")
            for k, v in sd.items():
                if "weight" in k:
                    print(f"  {k}: {v.shape}")
                    break
        except Exception as e:
            print(f"Failed: {e}")
