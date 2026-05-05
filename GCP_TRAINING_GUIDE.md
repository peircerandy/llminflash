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

# Create and attach the second disk for weights (name it descriptively, e.g., trainer-disk)
gcloud compute disks create trainer-disk \
    --zone=us-central1-a \
    --size=500GB \
    --type=pd-balanced

gcloud compute instances attach-disk predictor-trainer \
    --zone=us-central1-a \
    --disk=trainer-disk
```
*Note: If using the UI, select "Balanced persistent disk" or "SSD persistent disk" for the second disk, size 500GB, and name it "trainer-disk". "Storage pool" is not applicable in GCP—use disk types instead.*

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

### 2. Install Project
```bash
# Ensure Git and Python are installed
sudo apt update && sudo apt install -y git python3 python3-pip

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc 

# Create and activate environment
conda create -n llm-flash python=3.10 -y
conda activate llm-flash

git clone https://github.com/peircerandy/llminflash.git
cd llminflash
git checkout device-optimizations
pip install -r requirements.txt
```

### 2.5. Authenticate & Configure Cache (CRUCIAL)
Models like **Llama 3** are gated and require authentication. We also want to ensure the massive downloads go to our 500GB disk, not the boot disk.
```bash
# 1. Login to Hugging Face (Paste your "Read" token when prompted)
huggingface-cli login

# 2. Point the model cache to the mounted large disk
mkdir -p /mnt/trainer-disk/hf_cache
export HF_HOME=/mnt/trainer-disk/hf_cache
```

### 3. Run Agnostic Trainer
Train the predictor for any model. The script auto-detects architecture. Ensure you are in the `/mnt/trainer-disk` folder so the resulting `.bin` and `.json` files are saved there.
```bash
cd /mnt/trainer-disk
conda activate llm-flash
export HF_HOME=/mnt/trainer-disk/hf_cache

# For Llama 3 8B
python ~/llminflash/train_portable_predictor.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --samples 5000 \
    --rank 128

# For Clay v1.5 (Radar/SAR Foundation Model)
python ~/llminflash/train_portable_predictor.py \
    --model_id made-with-clay/Clay \
    --samples 5000 \
    --rank 128
```
*Note: Increasing `--samples` to 10000+ will provide higher accuracy on complex datasets.*

### 4. Export to Edge
Once training is complete, your `.bin` and `.json` files will be in `/mnt/trainer-disk/`. Use `gcloud scp` to download them to your local computer or edge device:
```bash
# Example for Llama 3
gcloud compute scp predictor-trainer:/mnt/trainer-disk/Meta-Llama-3-8B_predictors.* .

# Example for Clay Radar Model
gcloud compute scp predictor-trainer:/mnt/trainer-disk/Clay_predictors.* .
```

---

## 📡 Special Note: Radar Model (Clay v1.5)
Since Clay v1.5 is a Vision Transformer (ViT), the **`train_portable_predictor.py`** script will automatically handle its 96 MLP blocks. 

**Recommendation:** For Radar data, train on the **C4 dataset** (default) if you want general language-like feature extraction, or point the trainer to a **SAR image dataset** (e.g., Sentinel-1) on HuggingFace for specialized radar accuracy.
