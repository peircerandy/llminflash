# Google Cloud Training Guide: Sparse Predictors

This guide provides expert recommendations for training your sparsity predictors on Google Cloud Platform (GCP).

## 🚀 Recommended Architecture: Compute Engine vs. Vertex AI
For training individual model predictors (like Clay v1.5 or Llama 3), **Compute Engine** is recommended over Vertex AI Workbench. It gives you raw access to the GPU and avoids the high costs of managed notebook environments.

### 🛠 Recommended Specs
*   **Machine Type:** `n2-standard-8` (8 vCPUs, 30GB RAM). If `n2-standard-8` is unavailable in your zone, use `e2-standard-8`.
*   **GPU:** 
    *   **NVIDIA L4 (Recommended):** Best price-to-performance for training these small linear predictors.
    *   **NVIDIA A100 (Fastest):** Use this if you need to train multiple 70B+ model predictors simultaneously.
*   **Disk:** 200GB Standard Persistent Disk for the OS + **Balanced persistent disk** (or SSD persistent disk if preferred) for weight storage, sized at 500GB to accommodate multiple models (e.g., Clay v1.5, OPT-6.7B, Llama 3).
*   **OS Image:** `Deep Learning VM with PyTorch/CUDA` on Ubuntu 22.04 (standard Google image). Ubuntu 24.04 may work but 22.04 is more stable for existing toolchains.

---

## 🏗 Setup Instructions

### 1. Launch Instance
Use the following `gcloud` commands to spin up the optimal machine with two disks (or use the GCP UI for the same setup):
```bash
# Create the instance with boot disk
gcloud compute instances create predictor-trainer \
    --zone=us-central1-a \
    --machine-type=n2-standard-8 \
    --accelerator=type=nvidia-tesla-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE

# Create and attach the second disk for weights
gcloud compute disks create trainer-disk \
    --zone=us-central1-a \
    --size=500GB \
    --type=pd-balanced

gcloud compute instances attach-disk predictor-trainer \
    --zone=us-central1-a \
    --disk=trainer-disk
```

### 1.5. Mount the Second Disk
After attaching the disk, format and mount it for storing model weights and outputs:
```bash
# Check disk device (look for the 500GB disk, e.g., /dev/nvme0n2)
lsblk

# Format the disk (replace /dev/nvme0n2 with the actual device from lsblk)
sudo mkfs.ext4 /dev/nvme0n2

# Mount the disk
sudo mkdir -p /mnt/trainer-disk
sudo mount /dev/nvme0n2 /mnt/trainer-disk
sudo chown $USER:$USER /mnt/trainer-disk
```

### 1.6 Install the driver (This might take a few minutes)
```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# 2. Install CUDA Toolkit 12.1 (Compatible with the latest PyTorch)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# 3. Reboot the VM to activate the drivers
sudo reboot
```

### 2. Install Project & Dependencies
```bash
# Ensure Git and Python are installed
sudo apt update && sudo apt install -y git python3 python3-pip

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc 

# Create and activate environment (Python 3.11+ required for Clay)
conda create -n llm-flash python=3.11 -y
conda activate llm-flash

git clone https://github.com/peircerandy/llminflash.git
cd llminflash
git checkout device-optimizations
pip install -r requirements.txt
```

### 2.5. Authenticate & Download Models (CRUCIAL)
Models like **Llama 3** are gated and require authentication. Models like **Clay v1.5** require downloading a specific checkpoint file.

```bash
# 1. Login to Hugging Face (Paste your "Read" token when prompted)
hf login

# 2. Download Clay v1.5 Checkpoint
# Note: Using 'hf' (standard alias for huggingface-cli)
hf download made-with-clay/Clay v1.5/clay-v1.5.ckpt --local-dir /mnt/trainer-disk/clay-tmp
mv /mnt/trainer-disk/clay-tmp/v1.5/clay-v1.5.ckpt /mnt/trainer-disk/clay-v1.5.ckpt
rm -rf /mnt/trainer-disk/clay-tmp

# 3. INSTALL CLAY SPECIFIC LIBRARY (Required for geovit+DOFA architecture)
pip install git+https://github.com/Clay-foundation/model.git

# 4. Point the model cache to the mounted large disk
mkdir -p /mnt/trainer-disk/hf_cache
export HF_HOME=/mnt/trainer-disk/hf_cache
```

### 3. Run Agnostic Trainer
Train the predictor for any model. The script auto-detects architecture. Ensure you are in the `/mnt/trainer-disk` folder so the resulting `.bin` and `.json` files are saved there.
```bash
cd /mnt/trainer-disk
conda activate llm-flash
export HF_HOME=/mnt/trainer-disk/hf_cache

# For Llama 3 8B (Note the --is_causal flag for LLMs)
python ~/llminflash/train_portable_predictor.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --samples 5000 \
    --rank 128 \
    --is_causal

# For Clay v1.5 (Vision Transformer)
python ~/llminflash/train_portable_predictor.py \
    --model_id made-with-clay/Clay \
    --ckpt_path /mnt/trainer-disk/clay-v1.5.ckpt \
    --hidden_size 1024 \
    --ffn_dim 4096 \
    --num_layers 24 \
    --samples 5000 \
    --dataset_name "blanchon/EuroSAT_MSI"
```
*Note: Using 'blanchon/EuroSAT_MSI' provides 13 spectral bands. The script will automatically slice these to the 10 bands expected by the Clay Sentinel-2 configuration.*
*Note: Increasing `--samples` to 10000+ will provide higher accuracy on complex datasets.*

### 4. Export to Edge
Once training is complete, your `.bin` and `.json` files will be in `/mnt/trainer-disk/`. Use `gcloud scp` to download them:
```bash
# Example for Clay Radar Model
gcloud compute scp predictor-trainer:/mnt/trainer-disk/Clay_predictors.* .
```

---

## 📡 Special Note: Radar Model (Clay v1.5)
Clay v1.5 uses a custom **`geovit+DOFA`** architecture. By installing the `claymodel` package and using **`trust_remote_code=True`** (built into the updated trainer script), the system will automatically handle the 96 MLP blocks. 

**Recommendation:** For Radar data, train on the **C4 dataset** (default) if you want general language-like feature extraction, or point the trainer to a **SAR image dataset** on HuggingFace for specialized radar accuracy.
