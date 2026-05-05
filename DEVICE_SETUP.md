# Hardware Setup Guide: Raspberry Pi & Samsung S24 Ultra (Termux)

This guide explains how to compile and run the "LLM in a Flash" engine on highly constrained ARM devices. The C++ core has been heavily optimized with **ARM NEON SIMD intrinsics**, allowing for dramatic speedups on mobile and embedded processors compared to standard Python inferences.

## 🍓 Raspberry Pi (4B / 5)

The Raspberry Pi (especially the 8GB models) is an excellent target. While its DRAM is limited, its PCIe Gen 2/3 (Pi 5) interface allows for fast enough SSD streaming to run 7B-parameter models locally.

### 1. Prerequisites
You must boot from an NVMe SSD for this to work effectively. Running from a MicroSD card will be too slow.
*   **OS:** Raspberry Pi OS (64-bit) or Ubuntu Server 22.04+ (64-bit).
*   **Hardware:** Raspberry Pi 4B (8GB) or Pi 5 (8GB) + NVMe Base/Hat + NVMe SSD.

### 2. Install Dependencies
```bash
sudo apt update
sudo apt install -y build-essential libgomp1 python3-pip python3-dev git
```

### 3. Build the Engine
The `Makefile` will automatically detect the ARM architecture and apply `-march=native`. The C++ code will conditionally compile the `<arm_neon.h>` intrinsics.
```bash
git clone https://github.com/peircerandy/llminflash.git
cd llminflash
git checkout device-optimizations
pip3 install -r requirements.txt
make
```

### 4. Run
Ensure you have downloaded the required models and pre-processed the weights (see main `README.md`).
```bash
python3 chat.py --mode predictor
```

---

## 📱 Samsung S24 Ultra (Android via Termux)

Modern flagship phones like the S24 Ultra have incredibly powerful ARM processors (Snapdragon 8 Gen 3) and fast UFS 4.0 internal storage, making them perfect targets for LLM-in-a-Flash.

### 1. Prerequisites
*   Install **Termux** from F-Droid (do not use the Google Play Store version as it is outdated).

### 2. Termux Setup
Open Termux and run the following to set up a full Linux environment:
```bash
# Update packages
pkg update && pkg upgrade -y

# Install build tools and Python
pkg install -y build-essential clang python git cmake make openmp libgomp

# Grant Termux access to your phone's internal storage
termux-setup-storage
```

### 3. Build the Engine
```bash
git clone https://github.com/peircerandy/llminflash.git
cd llminflash
git checkout device-optimizations

# Install Python dependencies (this might take a while on Android)
pip install torch transformers accelerate numpy

# Build the shared library using clang
CXX=clang++ make
```

### 4. Run
You must move your pre-processed `.bin` weights and `.pt` layers to a folder inside `~/storage/shared/` so Termux can access them. Update the paths in `chat.py` to point to `/storage/emulated/0/YourFolder/`.

```bash
python chat.py --mode predictor
```

*Note: Android aggressively kills background processes that consume too much RAM. Ensure you are using the Layer-wise Loading technique to stay well under the S24 Ultra's 12GB RAM limit.*
