import torch
import os
import glob
import json

# OPT-6.7B Constants
MODEL_ID = "facebook/opt-6.7b"
FFN_FILE = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_bundled_ffn.bin"
PERSISTENT_FILE = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_persistent.bin"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"

def get_snapshot_dir():
    snapshot_dir = glob.glob(os.path.join(CACHE_PATH, "models--facebook--opt-6.7b/snapshots/*"))
    return sorted(snapshot_dir)[-1]

def bundle_weights():
    print(f"Re-bundling with Biases (Split-Shard handling)...")
    snap_dir = get_snapshot_dir()
    index_path = os.path.join(snap_dir, "pytorch_model.bin.index.json")
    with open(index_path, "r") as f: index_data = json.load(f)
    weight_map = index_data["weight_map"]

    # Part 1: Persistent
    print("Processing Persistent Weights...")
    persistent_keys = ["decoder.embed_tokens.weight", "decoder.embed_positions.weight", "decoder.final_layer_norm.weight", "decoder.final_layer_norm.bias"]
    for i in range(32):
        persistent_keys += [f"decoder.layers.{i}.self_attn.q_proj.weight", f"decoder.layers.{i}.self_attn.k_proj.weight", f"decoder.layers.{i}.self_attn.v_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight", f"decoder.layers.{i}.self_attn_layer_norm.bias", f"decoder.layers.{i}.final_layer_norm.weight", f"decoder.layers.{i}.final_layer_norm.bias", f"decoder.layers.{i}.fc2.bias"]

    shard_cache = {}
    def get_weight(key):
        s_name = weight_map[key]
        if s_name not in shard_cache:
            shard_cache.clear() # Keep only 1 shard in RAM
            shard_cache[s_name] = torch.load(os.path.join(snap_dir, s_name), map_location="cpu", weights_only=True, mmap=True)
        return shard_cache[s_name][key].to(torch.float16)

    with open(PERSISTENT_FILE, "wb") as f_pers:
        for key in persistent_keys:
            f_pers.write(get_weight(key).numpy().tobytes())

    # Part 2: Bundled FFN
    print("Processing FFN Weights...")
    with open(FFN_FILE, "wb") as f_ffn:
        for i in range(32):
            print(f"  Layer {i}...")
            fc1_w = get_weight(f"decoder.layers.{i}.fc1.weight")
            fc2_w = get_weight(f"decoder.layers.{i}.fc2.weight")
            fc1_b = get_weight(f"decoder.layers.{i}.fc1.bias")
            ffn_dim = fc1_w.shape[0]
            for n in range(ffn_dim):
                f_ffn.write(fc1_w[n, :].numpy().tobytes())
                f_ffn.write(fc2_w[:, n].numpy().tobytes())
                f_ffn.write(fc1_b[n].numpy().tobytes())

    print(f"\nDone! Weights updated.")

if __name__ == "__main__":
    bundle_weights()
