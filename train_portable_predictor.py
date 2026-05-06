"""
train_portable_predictor.py: Standalone, High-Performance Predictor Trainer.

This script captures activation sparsity from ANY transformer model (HuggingFace) 
and trains low-rank predictors. Supports LLMs and Vision Transformers (like Clay).

Manual overrides are provided to support custom architectures that Transformers 
might not natively recognize (e.g., Clay's 'geovit+DOFA').
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

class LowRankPredictor(nn.Module):
    def __init__(self, d_model, rank, d_ffn):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_ffn, bias=False)
    def forward(self, x):
        return torch.sigmoid(self.up(torch.relu(self.down(x))))

def train(args):
    print(f"--- Portable Predictor Trainer ---")
    print(f"Target Model: {args.model_id}")
    
    # 1. Setup Architecture Info (Config or Manual)
    d_model, d_ffn, num_layers = args.hidden_size, args.ffn_dim, args.num_layers
    
    if not all([d_model, d_ffn, num_layers]):
        print("Attempting to auto-detect architecture from Transformers...")
        try:
            config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
            d_model = d_model or getattr(config, "hidden_size", getattr(config, "d_model", None))
            d_ffn = d_ffn or getattr(config, "intermediate_size", getattr(config, "ffn_dim", None))
            num_layers = num_layers or getattr(config, "num_hidden_layers", getattr(config, "num_layers", None))
        except Exception as e:
            print(f"Auto-detection failed: {e}")
            print("Please provide --hidden_size, --ffn_dim, and --num_layers manually.")
            return

    if not all([d_model, d_ffn, num_layers]):
        print(f"Error: Incomplete dimensions. Found Hidden={d_model}, FFN={d_ffn}, Layers={num_layers}")
        return

    print(f"Using Arch: Hidden={d_model}, FFN={d_ffn}, Layers={num_layers}")

    # 2. Load Model Weights
    print("Loading model weights...")
    try:
        if args.is_causal:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        else:
            # For Clay/ViT models
            try:
                model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, dtype="auto")
            except ValueError as ve:
                if "geovit+DOFA" in str(ve) and "clay" in args.model_id.lower():
                    print("Registering custom 'geovit+DOFA' via claymodel library...")
                    from claymodel.module import ClayMAEModule
                    
                    # ClayMAEModule expects 'configs/metadata.yaml' relative to the CWD
                    if not os.path.exists("configs/metadata.yaml"):
                        print("Creating missing 'configs/metadata.yaml' for Clay model...")
                        os.makedirs("configs", exist_ok=True)
                        metadata_content = """sentinel-2-l2a:
  band_order: [blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22]
  rgb_indices: [2, 1, 0]
  gsd: 10
  bands:
    mean: {blue: 1105., green: 1355., red: 1552., rededge1: 1887., rededge2: 2422., rededge3: 2630., nir: 2743., nir08: 2785., swir16: 2388., swir22: 1835.}
    std: {blue: 1809., green: 1757., red: 1888., rededge1: 1870., rededge2: 1732., rededge3: 1697., nir: 1742., nir08: 1648., swir16: 1470., swir22: 1379.}
    wavelength: {blue: 0.493, green: 0.56, red: 0.665, rededge1: 0.704, rededge2: 0.74, rededge3: 0.783, nir: 0.842, nir08: 0.865, swir16: 1.61, swir22: 2.19}
sentinel-1-rtc:
  band_order: [vv, vh]
  gsd: 10
  bands:
    mean: {vv: -12.113, vh: -18.673}
    std: {vv: 8.314, vh: 8.017}
    wavelength: {vv: 3.5, vh: 4.0}
"""
                        with open("configs/metadata.yaml", "w") as f:
                            f.write(metadata_content)

                    if args.ckpt_path:
                        model = ClayMAEModule.load_from_checkpoint(args.ckpt_path)
                    else:
                        print("Error: For custom Clay architecture, please provide --ckpt_path to the .ckpt file.")
                        return
                else: raise ve
    except Exception as e:
        print(f"Model load failed: {e}")
        if "401" in str(e) or "gated" in str(e):
            print("\nCRUCIAL: This model requires login. Run 'huggingface-cli login' first.")
        return
        
    model.eval()

    # Auto-detect layers if not manually specified or to verify
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        actual_layers = len(model.model.blocks)
        if actual_layers != num_layers:
            print(f"Adjusting num_layers from {num_layers} to {actual_layers} based on model blocks.")
            num_layers = actual_layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        actual_layers = len(model.model.layers)
        if actual_layers != num_layers:
            print(f"Adjusting num_layers from {num_layers} to {actual_layers} based on model layers.")
            num_layers = actual_layers

    # 3. Initialize Predictors
    predictors = [LowRankPredictor(d_model, args.rank, d_ffn).cuda() for _ in range(num_layers)]
    optimizers = [optim.Adam(p.parameters(), lr=1e-4) for p in predictors]
    
    # 4. Capture Logic
    captured_inputs = {}
    captured_acts = {}
    
    def get_input_hook(idx):
        def hook(m, i, o):
            captured_inputs[idx] = i[0].detach().float()
        return hook
        
    def get_act_hook(idx):
        def hook(m, i, o):
            act_data = o[0] if isinstance(o, tuple) else o
            captured_acts[idx] = (act_data > 0).float().detach()
        return hook

    hooks = []
    # Flexible module search for hooks
    for i in range(num_layers):
        mlp_mod = None
        act_mod = None
        
        # 1. Check Clay / Transformer.layers[i][1] (FeedForward)
        if hasattr(model, "model") and hasattr(model.model, "encoder") and \
           hasattr(model.model.encoder, "transformer") and hasattr(model.model.encoder.transformer, "layers"):
            layers = model.model.encoder.transformer.layers
            if i < len(layers):
                # Clay layers are [Attention, FeedForward]
                mlp_mod = layers[i][1]
                # FeedForward usually has an 'fn' or 'net' or just an act
                # Looking at standard ViT/Clay FeedForward: it has 'net' nn.Sequential
                if hasattr(mlp_mod, "net"):
                    # Find GELU in the sequence
                    for sub in mlp_mod.net:
                        if isinstance(sub, (nn.GELU, nn.ReLU, nn.SiLU)):
                            act_mod = sub
                            break
        
        # 2. Check Standard ViT / Timm
        if not mlp_mod and hasattr(model, "model") and hasattr(model.model, "blocks") and i < len(model.model.blocks):
            mlp_mod = model.model.blocks[i].mlp
            act_mod = getattr(mlp_mod, "act", None)
        
        # 3. Check Direct ViT
        if not mlp_mod and hasattr(model, "blocks") and i < len(model.blocks):
            mlp_mod = model.blocks[i].mlp
            act_mod = getattr(mlp_mod, "act", None)

        # 4. Check Llama / HF standard
        if not mlp_mod and hasattr(model, "model") and hasattr(model.model, "layers") and i < len(model.model.layers):
            mlp_mod = model.model.layers[i].mlp
            act_mod = getattr(mlp_mod, "act_fn", None)
            
        # 5. Check OPT
        if not mlp_mod and hasattr(model, "model") and hasattr(model.model, "decoder") and \
           hasattr(model.model.decoder, "layers") and i < len(model.model.decoder.layers):
            mlp_mod = model.model.decoder.layers[i]
            act_mod = getattr(mlp_mod, "activation_fn", None)

        if mlp_mod and act_mod:
            hooks.append(mlp_mod.register_forward_hook(get_input_hook(i)))
            hooks.append(act_mod.register_forward_hook(get_act_hook(i)))
        else:
            print(f"Warning: MLP or Activation layer {i} not found. Capture will fail.")

    # 5. Training Loop
    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = "allenai/c4" if args.is_causal else "blanchon/EuroSAT_MSI"
    
    print(f"Loading dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, "en" if "c4" in dataset_name else None, split="train", streaming=True)
    except Exception as e:
        print(f"Error: Could not load dataset {dataset_name}: {e}")
        return

    pbar = tqdm(total=args.samples, desc="Training")
    
    tokenizer = None
    if args.is_causal:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        except:
            print("Warning: No tokenizer found for causal model.")

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i, sample in enumerate(dataset):
        if i >= args.samples: break
        
        try:
            captured_inputs.clear()
            captured_acts.clear()
            
            if args.is_causal:
                # --- LLM Path ---
                text = None
                for key in ['text', 'content', 'body']:
                    if key in sample: text = sample[key]; break
                
                if text is None:
                    print(f"\nWarning: Sample {i} has no text field. Skipping.")
                    continue
                
                if tokenizer:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
                    with torch.no_grad(): model(**inputs)
                else:
                    print("Error: No tokenizer found for causal model.")
                    break
                    
            else:
                # --- Vision Path ---
                img_data = None
                for key in ['image', 'img', 'pixels', 'pixel_values', 'MSI']:
                    if key in sample: img_data = sample[key]; break
                
                if img_data is None:
                    print(f"\nWarning: Sample {i} has no recognizable image field. Skipping.")
                    continue

                # Preprocess
                if isinstance(img_data, list): img = torch.tensor(img_data).float()
                elif not isinstance(img_data, torch.Tensor):
                    if hasattr(img_data, "convert"): img = preprocess(img_data.convert("RGB"))
                    else: img = transforms.ToTensor()(img_data)
                else: img = img_data
                
                if img.dim() == 2: img = img.unsqueeze(0)
                if img.shape[-1] != 224 or img.shape[-2] != 224:
                    img = transforms.Resize((224, 224))(img)
                
                img = img.unsqueeze(0).to("cuda").float()
                
                # Channel Match
                expected_channels = 10 if "clay" in args.model_id.lower() else 3
                if img.shape[1] != expected_channels:
                    if img.shape[1] < expected_channels:
                        pad = torch.zeros((img.shape[0], expected_channels - img.shape[1], img.shape[2], img.shape[3]), device="cuda")
                        img = torch.cat([img, pad], dim=1)
                    else: img = img[:, :expected_channels, :, :]

                with torch.no_grad():
                    if "clay" in args.model_id.lower():
                        datacube = {
                            "pixels": img,
                            "time": torch.zeros((img.shape[0], 4), device="cuda"),
                            "latlon": torch.zeros((img.shape[0], 4), device="cuda"),
                            "platform": ["sentinel-2-l2a"] * img.shape[0],
                            "waves": torch.tensor([490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0], device="cuda"),
                            "gsd": torch.tensor([10.0], device="cuda")
                        }
                        # ClayMAE forward()
                        model(datacube)
                    else:
                        model(img)

            # --- Unified Predictor Training ---
            for l_idx in range(num_layers):
                x = captured_inputs.get(l_idx)
                y_true = captured_acts.get(l_idx)
                
                if x is None or y_true is None: continue
                if x.shape[0] != y_true.shape[0]: continue

                optimizers[l_idx].zero_grad()
                y_pred = predictors[l_idx](x)
                loss = nn.BCELoss()(y_pred.view(-1), y_true.view(-1).float())
                loss.backward()
                optimizers[l_idx].step()
            pbar.update(1)
        except Exception as e:
            if i % 100 == 0: print(f"Sample {i} failed: {e}")
            continue
    
    for h in hooks: h.remove()
    pbar.close()

    # 6. Export
    base_name = args.model_id.split("/")[-1]
    bin_path = f"{base_name}_predictors.bin"
    meta_path = f"{base_name}_predictors.json"

    with open(bin_path, "wb") as f:
        for p in predictors:
            f.write(p.down.weight.data.cpu().float().numpy().tobytes())
            f.write(p.up.weight.data.cpu().float().numpy().tobytes())

    metadata = {
        "model_id": args.model_id, "hidden_size": d_model, "ffn_dim": d_ffn,
        "num_layers": num_layers, "rank": args.rank, "dtype": "float32"
    }
    with open(meta_path, "w") as f: json.dump(metadata, f, indent=4)
    print(f"Done! Exported to {bin_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agnostic Predictor Trainer.")
    parser.add_argument("--model_id", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--ckpt_path", type=str, help="Path to .ckpt (for custom models like Clay)")
    parser.add_argument("--hidden_size", type=int, help="Manual Hidden Size override")
    parser.add_argument("--ffn_dim", type=int, help="Manual FFN Dim override")
    parser.add_argument("--num_layers", type=int, help="Manual Layers override")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true", help="Set for LLMs")
    parser.add_argument("--dataset_name", type=str, help="HF Dataset name (e.g., 'allenai/c4' or 'food101')")
    train(parser.parse_args())
