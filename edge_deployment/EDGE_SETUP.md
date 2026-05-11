# Edge Deployment for LLM-in-a-Flash

This folder contains the minimum necessary scripts to compile the Flash Engine and run inference using either a Causal LLM (OPT/Llama) or a Vision Transformer (Clay) on constrained edge hardware like a Raspberry Pi 4B or a Galaxy S24 Ultra (via Termux).

## 🛠 Setup Instructions

### 1. Install Dependencies (Raspberry Pi / Termux)
I recommend using **Miniforge** (Conda for ARM) to manage your environment on the Pi.

**System Level (Run this first on Pi):**
```bash
sudo apt-get update && sudo apt-get install libgomp1 -y
```

**Python Level:**
```bash
# 1. Download and Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh

# 2. Create environment
conda create -n flash-edge python=3.11 -y
conda activate flash-edge
```bash
# 3. Install packages
pip install -r requirements_edge.txt
```

### 2. Compile the Engine
**IMPORTANT:** Since you copied these files from your laptop, you **must** clean the old laptop-compiled binary first, or it will fail to load on the Pi.

```bash
make clean
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
*   `opt_6_7b_layers/` directory (From `/mnt/wsl/PHYSICALDRIVE0p3/`)
*   `opt_6_7b_bundled_ffn.bin` (From `/mnt/wsl/PHYSICALDRIVE0p3/`)
*   `opt_gcp_predictors.bin` (GCP trained version - from `/mnt/wsl/PHYSICALDRIVE0p3/`)
*   `opt_gcp_predictors.json` (Required metadata - from `/mnt/wsl/PHYSICALDRIVE0p3/`)
### 4. Running Edge Inference

You can run individual tests or the full automated suite.

**Run Automated Suite:**
```bash
# Run Clay benchmarks only (Default)
bash edge_benchmark.sh

# Run LLM benchmarks only
bash edge_benchmark.sh --llm

# Run BOTH Clay and LLM benchmarks
bash edge_benchmark.sh --all

# Run benchmarks using GCP predictors (for LLM)
bash edge_benchmark.sh --llm --gcp
```

**Individual Modes:**
...

```bash
# Predictor (Standard LLM-in-a-Flash)
python edge_clay.py --mode predictor

# Dense (Standard PyTorch - WARNING: Requires Swap)
python edge_clay.py --mode dense
```

### 🧠 Managing Memory (Swap)
If **Dense Mode** crashes your Pi immediately, you may need to increase your Swap memory so the OS can "offload" the 5GB model to disk (this proves why our method is faster!).

```bash
# 1. Stop swap
sudo dphys-swapfile swapoff
# 2. Edit /etc/dphys-swapfile and set CONF_SWAPSIZE=4096
sudo nano /etc/dphys-swapfile
# 3. Re-init and start
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 📸 Using Satellite Data
You can pipe either a standard photo or raw multispectral data into the model:

1. **Capture a photo (Pi Camera):**
   ```bash
   libcamera-still -o test_capture.jpg
   ```
2. **Use Multispectral Data (.npy):**
   If you have a 10-channel or 13-channel `.npy` array (e.g. from Sentinel-2), our script handles it automatically.
   
3. **Run inference:**
   ```bash
   # For standard images:
   python edge_clay.py --image test_capture.jpg
   
   # For raw multispectral data:
   python edge_clay.py --image my_sentinel_data.npy
   ```

### ⚠️ Troubleshooting the "OpenMP Loop" Hang
If your Pi displays OpenBLAS warnings and hangs, it's due to a "deadlock" between the C++ Engine and PyTorch's math library.

**The Fix:**
Run the script with these specific environment flags:
```bash
OPENBLAS_VERBOSE=0 OMP_WAIT_POLICY=PASSIVE python edge_clay.py --mode predictor
```
This tells the processors to be "polite" and wait their turn, which prevents the hang.

### 📡 Feature Telemetry & Classification
When you run `edge_clay.py`, it generates two small telemetry files:
1. `edge_output_heatmap.npy` (~3KB): Spatial features.
2. `edge_output_cls_token.npy` (~4KB): Semantic embedding.

**To classify your camera photo:**
Copy the `edge_output_cls_token.npy` back to your laptop and run the classification suite. This allows you to perform heavy "accuracy" math on your powerful machine without overloading the Pi.

### 🔄 Round-Trip Workflow

To create a professional comparison for your presentation:

1. **Copy the code to your Pi/Phone:**
   Using `scp` is much easier than git for subfolders:
   ```bash
   scp -r ./edge_deployment rpi@192.168.1.x:~/
   ```

2. **Run your tests on the device:**
   ```bash
   python edge_clay.py --mode predictor
   ```

3. **Bring the results back to your LAPTOP:**
   Copy the telemetry, metrics, and **the input image** into the `edge_results/` folder on your laptop:
   ```bash
   # From your laptop terminal:
   # 1. Metrics and Tokens
   scp rpi@192.168.1.x:~/edge_deployment/*_metrics_*.json ./edge_results/
   scp rpi@192.168.1.x:~/edge_deployment/*_output_*.npy ./edge_results/

   # 2. The input image (REQUIRED for Visual Proof)
   # If you used the default:
   scp rpi@192.168.1.x:~/edge_deployment/sample_satellite.png ./edge_results/
   # If you used a custom camera photo, name it with the device prefix:
   scp rpi@192.168.1.x:~/edge_deployment/test_capture.jpg ./edge_results/pi4_test_capture.jpg
   ```

4. **Analyze on your laptop:**
   
   **To Classify a Specific Token:**
   ```bash
   python classify_edge_result.py edge_results/pi4_output_cls_token.npy
   ```

   **To Graph Multi-Model Results:**
   ```bash
   python graph_edge_results.py
   ```
   This generates:
   - `edge_latency_clay.png`
   - `edge_latency_opt.png`
   - `edge_fidelity_proof.png`
   - `edge_visual_proof_pi4.png` (Reconstructs the edge prediction visually!)

**Test Causal LLM:**
```bash
# Run with standard predictors
python chat.py --mode predictor

# Run with GCP predictors (Recommended for best accuracy)
python chat.py --mode predictor --predictor opt_gcp_predictors.bin --layers ./opt_6_7b_layers --ffn_bin opt_6_7b_bundled_ffn.bin
```
