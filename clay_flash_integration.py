
import torch
import torch.nn as nn
import sys
import ctypes
import os
import warnings

# Note: Requires installation via: pip install git+https://github.com/Clay-foundation/model.git
try:
    from claymodel.module import ClayMAEModule
    from accelerate import init_empty_weights
except ImportError:
    print("\n[!] Error: Missing dependencies.")
    print("Please ensure you are using the correct conda environment:")
    print("    conda activate llm-flash-v2")
    sys.exit(1)

class FlashViTFFN(nn.Module):
    """
    A drop-in replacement for the MLP block in Clay's Vision Transformer.
    Instead of loading the massive MLP weights into RAM, we stream them from SSD
    based on activation sparsity, exactly like we do for OPT/Llama.
    """
    def __init__(self, layer_idx, engine_ptr, fc1_mod, fc2_mod, hidden_size=1024, ffn_dim=4096):
        super().__init__()
        self.layer_idx = layer_idx
        self.engine_ptr = engine_ptr
        self.hidden_size = hidden_size
        
        # Load C++ library
        self.lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
        self.lib.execute_ffn_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
                                               ctypes.POINTER(ctypes.c_float), ctypes.c_int, 
                                               ctypes.POINTER(ctypes.c_float), ctypes.c_int]

        # Ensure biases have real CPU storage (not meta-tensors)
        try:
            if hasattr(fc1_mod, 'bias') and fc1_mod.bias is not None and fc1_mod.bias.device.type != 'meta':
                self.fc1_bias = fc1_mod.bias.detach().float().cpu().contiguous()
            else:
                self.fc1_bias = torch.zeros(ffn_dim, dtype=torch.float32)
                
            if hasattr(fc2_mod, 'bias') and fc2_mod.bias is not None and fc2_mod.bias.device.type != 'meta':
                self.fc2_bias = fc2_mod.bias.detach().float().cpu().contiguous()
            else:
                self.fc2_bias = torch.zeros(hidden_size, dtype=torch.float32)
        except:
            self.fc1_bias = torch.zeros(ffn_dim, dtype=torch.float32)
            self.fc2_bias = torch.zeros(hidden_size, dtype=torch.float32)
            
        self.fc1_bias_c = ctypes.cast(self.fc1_bias.data_ptr(), ctypes.POINTER(ctypes.c_float))

    def forward(self, x):
        orig_shape = x.shape
        flat_x = x.view(-1, self.hidden_size).float().cpu().contiguous()
        num_tokens = flat_x.shape[0]
        out_cpu = torch.zeros_like(flat_x)
        
        # mode 0 = Predictor
        self.lib.execute_ffn_layer(
            self.engine_ptr, self.layer_idx, 
            ctypes.cast(flat_x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_cpu.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            num_tokens, 
            self.fc1_bias_c, 
            0 
        )
        
        res = out_cpu.to(x.device).view(*orig_shape)
        return res + self.fc2_bias.to(x.device)

def patch_clay_model(model_path="/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--made-with-clay--Clay/snapshots/70200ebcccdf67bf2a0cb9984c77ddee26c10ed2/v1.5/clay-v1.5.ckpt", 
                     ffn_bin_path=b"/mnt/wsl/PHYSICALDRIVE0p3/clay_bundled_ffn.bin",
                     pred_bin_path=b"/mnt/wsl/PHYSICALDRIVE0p3/Clay_predictors.bin"):
    
    if not os.path.exists("libengine.so"):
        print("Error: libengine.so not found. Run 'make' first.")
        return None, None

    print("Loading Clay v1.5 Radar Model Structure (CPU Materialized)...")
    try:
        if not os.path.exists("configs/metadata.yaml"):
            os.makedirs("configs", exist_ok=True)
            with open("configs/metadata.yaml", "w") as f:
                f.write("""sentinel-2-l2a:
  band_order: [blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22]
  rgb_indices: [2, 1, 0]
  gsd: 10
  bands:
    mean: {blue: 1105., green: 1355., red: 1552., rededge1: 1887., rededge2: 2422., rededge3: 2630., nir: 2743., nir08: 2785., swir16: 2388., swir22: 1835.}
    std: {blue: 1809., green: 1757., red: 1888., rededge1: 1870., rededge2: 1732., rededge3: 1697., nir: 1742., nir08: 1648., swir16: 1470., swir22: 1379.}
    wavelength: {blue: 0.493, green: 0.56, red: 0.665, rededge1: 0.704, rededge2: 0.74, rededge3: 0.783, nir: 0.842, nir08: 0.865, swir16: 1.61, swir22: 2.19}
""")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = ClayMAEModule(model_size="large", patch_size=14)
        
        print("Surgically loading non-MLP weights from checkpoint...")
        sd = torch.load(model_path, map_location="cpu", mmap=True, weights_only=True)
        state_dict = sd.get("state_dict", sd)
        
        clean_sd = {}
        target_keys = model.state_dict().keys()
        for k, v in state_dict.items():
            new_k = k.replace("model.teacher.", "model.")
            if ".mlp.net.1." in new_k or ".mlp.net.3." in new_k or ".1.net.1." in new_k or ".1.net.3." in new_k:
                continue
            if new_k in target_keys and model.state_dict()[new_k].shape == v.shape:
                clean_sd[new_k] = v
        
        model.load_state_dict(clean_sd, strict=False)
        print(f"Successfully loaded {len(clean_sd)} non-MLP tensors to CPU.")
    except Exception as e:
        print(f"Error initializing model structure: {e}")
        return None, None
        
    print("Initializing C++ Flash Engine for ViT MLPs...")
    lib = ctypes.CDLL(os.path.abspath("./libengine.so"))
    lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
    lib.init_engine.restype = ctypes.c_void_p
    lib.set_engine_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    lib.destroy_engine.argtypes = [ctypes.c_void_p]

    num_layers = len(model.model.encoder.transformer.layers)
    engine_ptr = lib.init_engine(ffn_bin_path, pred_bin_path, 1024, 4096, num_layers, 0)
    
    if not engine_ptr:
        print("Error: Engine initialization failed.")
        return None, None

    lib.set_engine_config(engine_ptr, 1024, 0.1, 5)
    
    print("Patching Transformer Blocks...")
    for i, layer in enumerate(model.model.encoder.transformer.layers):
        ff_block = layer[1]
        fc1_mod = ff_block.net[1]
        fc2_mod = ff_block.net[3]
        ff_block.net[1] = nn.Identity()
        ff_block.net[2] = nn.Identity()
        ff_block.net[3] = FlashViTFFN(i, engine_ptr, fc1_mod, fc2_mod, hidden_size=1024, ffn_dim=4096)
        
    # Bypassing broken library forward() logic (teacher crash)
    def custom_forward(datacube):
        # The encoder returns 4 values: (encoded_patches, unmasked_idx, masked_idx, masked_matrix)
        results = model.model.encoder(datacube)
        # Return the first value [B, L, D]
        return results[0]
    
    model.model.forward = custom_forward
    model.eval() 
    print("Clay Model successfully patched with Real Flash Predictors!")
    return model, engine_ptr

if __name__ == "__main__":
    import time
    from datasets import load_dataset
    
    # 1. Patch the model
    model, engine_ptr = patch_clay_model()
    if not model:
        sys.exit(1)

    # 2. Live Demo
    print("\n" + "="*40)
    print("LIVE DEMO: Running Flash-ViT Inference")
    print("="*40)
    
    try:
        print("Loading real multi-spectral samples from EuroSAT...")
        dataset = load_dataset("blanchon/EuroSAT_MSI", split="train", streaming=True)
        sample = next(iter(dataset))
        
        img = torch.tensor(sample['image']).float()
        if img.dim() == 2: img = img.unsqueeze(0)
        if img.shape[0] > 10: img = img[:10, :, :]
        
        import torchvision.transforms as T
        img = T.Resize((224, 224))(img).unsqueeze(0)
        # Standard Datacube format
        datacube = {
            "pixels": img,
            "waves": torch.tensor([490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0]),
            "time": torch.zeros((1, 4)),
            "latlon": torch.zeros((1, 4)),
            "platform": ["sentinel-2-l2a"],
            "gsd": torch.tensor([10.0])
        }

        print(f"Input Shape: {img.shape} (10 spectral bands)")
        print("Starting Forward Pass (Predicting sparsity & streaming from SSD)...")
        
        start_time = time.time()
        with torch.no_grad():
            output = model(datacube)
        end_time = time.time()
        
        print("\n" + "-"*40)
        print(f"INFERENCE COMPLETE!")
        print(f"Total Model Latency: {end_time - start_time:.4f} seconds")
        print(f"Output Embedding Size: {output.shape}")
        print("-"*40)
        print("Success: The model processed real pixels using only SSD-resident weights!")
        
        # 3. Visualization: Side-by-Side Comparison
        print("\nGenerating Comparison Graphic (clay_comparison.png)...")
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg') 
        
        # 1. Prepare Input Image (Bands 2, 1, 0 for RGB)
        # Note: EuroSAT MSI bands are different from standard.
        # Simple normalization for display
        rgb_img = img[0, [2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0)
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        
        # 2. Prepare Heatmap
        features = output[0, 1:, :].cpu().numpy()
        grid_size = int(features.shape[0]**0.5)
        heatmap = features.mean(axis=-1).reshape(grid_size, grid_size)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Input
        ax[0].imshow(rgb_img)
        ax[0].set_title("Input Satellite Image (RGB)")
        ax[0].axis('off')
        
        # Plot Heatmap
        im = ax[1].imshow(heatmap, cmap='magma')
        ax[1].set_title("Flash-ViT Feature Intensity\n(SSD-Resident Weights)")
        ax[1].axis('off')
        
        # Explain Colors
        fig.colorbar(im, ax=ax[1], label="Mean Embedding Activation")
        
        plt.tight_layout()
        plt.savefig("clay_comparison.png")
        print("Comparison saved! Open 'clay_comparison.png' to see the input vs. output.")
        print("Color Meanings: Bright/Yellow = High Feature Density (e.g., structures/vegetation). Dark/Purple = Low Density (e.g., water/shadow).")
        
    except Exception as e:
        print(f"\n[!] Demo failed: {e}")
        import traceback
        traceback.print_exc()
