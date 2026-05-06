import os
import shutil
import random
from torchvision.datasets import EuroSAT

print("Randomly extracting 5 test images from the EuroSAT dataset...")

# Load the dataset
dataset = EuroSAT(root="CLAY/data", download=False)
os.makedirs("CLAY/sample_tests", exist_ok=True)

# Randomly select 5 images
sample_indices = random.sample(range(len(dataset)), 5)

for i, idx in enumerate(sample_indices):
    image, label_idx = dataset[idx]
    label_name = dataset.classes[label_idx]

    # Save the image (the filename includes the true label for easy comparison)
    save_path = f"CLAY/sample_tests/test_img_{i+1}_TrueLabel_{label_name}.jpg"
    image.save(save_path)
    print(f"✅ Successfully saved: {save_path}")

print("\n🎉 Images have been saved in the CLAY/sample_tests/ directory!")
print(
    "You can now feed them into your inference script to see how accurately the AI predicts!"
)
