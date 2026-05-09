import os
import shutil
from transformers import AutoTokenizer, AutoConfig

def bundle_cache(model_id="facebook/opt-6.7b", target_dir="edge_deployment/hf_cache"):
    print(f"📦 Bundling minimal cache for {model_id}...")
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Download/Load tokenizer and config to local dir
    # We save them directly to the target dir so we can point to it
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    
    tokenizer.save_pretrained(target_dir)
    config.save_pretrained(target_dir)
    
    print(f"✅ Minimal cache bundled to {target_dir}")
    print("Next step: scp -r edge_deployment/hf_cache your-pi:~/edge_deployment/")

if __name__ == "__main__":
    bundle_cache()
