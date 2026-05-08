# Edge Deployment for LLM-in-a-Flash

This folder contains the minimum necessary scripts to compile the Flash Engine and run inference using either a Causal LLM (OPT/Llama) or a Vision Transformer (Clay) on constrained edge hardware like a Raspberry Pi 4B or a Galaxy S24 Ultra (via Termux).

## 🛠 Setup Instructions

### 1. Install Dependencies (Raspberry Pi / Termux)
I recommend using **Miniforge** (Conda for ARM) to manage your environment on the Pi.

```bash
# 1. Download and Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh

# 2. Create environment
conda create -n flash-edge python=3.11 -y
conda activate flash-edge

# 3. Install packages
pip install -r requirements_edge.txt
```

### 2. Compile the Engine
The engine is written in C++ and uses OpenMP for multithreading. It also has specific compiler directives for ARM NEON, which makes it particularly fast on Raspberry Pi and modern Android devices.

```bash
make
```

### 3. Transfer Weights
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

### 4. Running Edge Inference

**Test Clay Vision Transformer:**
```bash
# Run Predictor mode (Default)
python edge_clay.py --image sample_satellite.png

# Run Draft mode (Block Skipping)
python edge_clay.py --mode draft --image sample_satellite.png

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

### 📡 Feature Telemetry & Classification
When you run `edge_clay.py`, it generates two small telemetry files:
1. `edge_output_heatmap.npy` (~3KB): Spatial features.
2. `edge_output_cls_token.npy` (~4KB): Semantic embedding.

**To classify your camera photo:**
Copy the `edge_output_cls_token.npy` back to your laptop and run the classification suite. This allows you to perform heavy "accuracy" math on your powerful machine without overloading the Pi.

**Test Causal LLM:**
```bash
python chat.py --mode predictor
```
