import torchvision
import os  # stdlib: filesystem paths

print("🌍 Downloading EuroSAT satellite dataset (may take a while, ~90MB)...")

# torchvision: download and extract automatically
dataset = torchvision.datasets.EuroSAT(root="CLAY/data", download=True)

print("\n" + "=" * 40)
print(f"✅ Download complete. Total images: {len(dataset)}.")
print("🏷️ 10 classes:")
for i, class_name in enumerate(dataset.classes):
    print(f"   {i}: {class_name}")
print("=" * 40)

# Inspect the first sample
img, label_idx = dataset[0]
class_name = dataset.classes[label_idx]

print(f"\n🖼️ First sample class: {class_name}")
print(f"📐 Image size and mode: {img.size}, {img.mode}")

# --- Output directory ---

# 1. Where to write artifacts
output_dir = "CLAY/outputs"

# 2. Create directory (exist_ok=True: no error if it already exists)
os.makedirs(output_dir, exist_ok=True)

# 3. Save preview image there
img.save(f"{output_dir}/sample_satellite.jpg")

print(f"💾 Saved first image to '{output_dir}/sample_satellite.jpg'.")
