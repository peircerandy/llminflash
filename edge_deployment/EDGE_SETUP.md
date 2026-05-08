# Edge Deployment for LLM-in-a-Flash

This folder contains the minimum necessary scripts to compile the Flash Engine and run inference using either a Causal LLM (OPT/Llama) or a Vision Transformer (Clay) on constrained edge hardware like a Raspberry Pi 4B or a Galaxy S24 Ultra (via Termux).

## Setup Instructions

### 1. Compile the Engine
The engine is written in C++ and uses OpenMP for multithreading. It also has specific compiler directives for ARM NEON, which makes it particularly fast on Raspberry Pi and modern Android devices.

```bash
make
```

### 2. Transfer Weights
You do not need to transfer the massive PyTorch models (which would OOM your device anyway). You only need the specific extracted resident layers and the bundled FFN binaries. 

Copy these from your main workstation using `scp` or `rsync`:

**For the Vision Transformer (Clay v1.5):**
*   `clay-v1.5.ckpt`
*   `clay_bundled_ffn.bin`
*   `Clay_predictors.bin`

**For the Causal LLM (OPT-6.7b):**
*   `opt_6_7b_resident.pt` (Contains only Attention and Norm layers)
*   `opt_bundled_ffn.bin`
*   `opt_predictors.bin`

Place them in the same directory as the scripts, or update the paths in the scripts.

### 3. Running Edge Inference

**Test Clay Vision Transformer:**
```bash
# Run with the included satellite sample
python edge_clay.py

# Run with your own camera photo
python edge_clay.py --image my_photo.jpg
```

### 📸 Using the Raspberry Pi Camera
You can pipe a fresh photo directly from the Pi Camera into the model:

1. **Capture a photo:**
   ```bash
   libcamera-still -o test_capture.jpg
   ```
2. **Run inference:**
   ```bash
   python edge_clay.py --image test_capture.jpg
   ```
*Note: Since Clay is an Earth Observation model, expect "creative" classifications for ground-level objects! Use the generated `edge_output_heatmap.npy` to see what parts of your photo the AI thinks are important features.*

**Test Causal LLM:**
```bash
python chat.py --mode predictor
```
*Note: Make sure your `chat.py` is configured to point to the correct `.bin` and `.pt` files.*
