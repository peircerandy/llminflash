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

# Change to the data directory for training
cd /mnt/trainer-disk
# Note: The directory will be empty initially; trained files will appear here after running the trainer.
```

### 2. Install Project
```bash
# Ensure Git and Python are installed (may not be pre-installed on some images)
sudo apt update && sudo apt install -y git python3 python3-pip

# Install Miniconda for environment management (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc  # Reload shell to apply conda init changes

# Create and activate a conda environment for the project
conda create -n llm-flash python=3.10 -y
conda activate llm-flash

git clone https://github.com/peircerandy/llminflash.git
cd llminflash
git checkout device-optimizations
pip install -r requirements.txt
```
*Note: Using a conda environment isolates dependencies and matches the project's testing setup. If conda is already installed, skip the Miniconda installation steps.*

### 3. Run Agnostic Trainer
Train the predictor for any model (e.g., Clay v1.5, OPT-6.7B, or Llama 3 8B). The script auto-detects architecture. Ensure the conda environment is activated and you're in the data directory:
```bash
cd /mnt/trainer-disk
conda activate llm-flash

# For Clay v1.5 (Vision Transformer)
python ~/llminflash/train_portable_predictor.py \
    --model_id made-with-clay/Clay \
    --samples 5000 \
    --rank 128

# For OPT-6.7B
python ~/llminflash/train_portable_predictor.py \
    --model_id facebook/opt-6.7b \
    --samples 5000 \
    --rank 128

# For Llama 3 8B
python ~/llminflash/train_portable_predictor.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --samples 5000 \
    --rank 128
```
*Note: Increasing `--samples` to 5000-10000 will provide higher accuracy on complex datasets like Radar/SAR. For Clay, use C4 dataset (default) or a SAR dataset from HuggingFace. The predictor files (.bin and .json) will be saved in /mnt/trainer-disk.*

### 4. Export to Edge
Once training is complete, download the `.bin` and `.json` files from the data disk to your local machine (replace `MODEL_NAME` with your model's identifier):
```bash
gcloud compute scp predictor-trainer:/mnt/trainer-disk/MODEL_NAME_predictors.* .
```

---

## 📡 Special Note: Radar Model (Clay v1.5)
Since Clay v1.5 is a Vision Transformer (ViT), ensure you use the **`train_portable_predictor.py`** script. Because it auto-detects architecture, it will handle the 96 MLP blocks in Clay without modification. 

**Recommendation:** For Radar data, train on the **C4 dataset** (default) if you want general language-like feature extraction, or point the trainer to a **SAR image dataset** from HuggingFace for domain-specific accuracy. The 500GB second disk provides ample space for Clay checkpoints, datasets, and predictor outputs alongside other models.
