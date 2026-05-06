import torch
import torchvision.transforms as T
from torchvision.datasets import EuroSAT
from claymodel.module import ClayMAEModule

# 1. Environment Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Load the Clay base model
model = ClayMAEModule.load_from_checkpoint(
    "CLAY/clay-v1.5.ckpt", metadata_path="configs/metadata.yaml", strict=False
).to(device)
model.eval()

# 2. Image Preprocessing
# Standardizing the image to match Clay's expected input format
preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 3. Load the Image Dataset
dataset = EuroSAT(root="CLAY/data", download=False)
img, label = dataset[0]
class_name = dataset.classes[label]

# 4. Prepare 10-Channel Input
# Clay expects 10 channels. We take our 3 RGB channels and pad with 7 blank channels.
pixels = preprocess(img).unsqueeze(0).to(device)
padding = torch.zeros((1, 7, 224, 224), device=device)
pixels_10ch = torch.cat([pixels, padding], dim=1)  # Final shape: [1, 10, 224, 224]

# Define wavelengths for the 3 active RGB channels (Sentinel-2 approx.), remaining 7 are zeroed out
waves = torch.tensor([665.0, 560.0, 490.0, 0, 0, 0, 0, 0, 0, 0], device=device)

print(f"🧠 Manually extracting features from: {class_name}...")

# 5. 🚀 Navigating the Transformer Core 🚀
with torch.no_grad():
    # Step 1: Get pure patches fused with wavelength info (ignoring the 128-dim byproduct)
    patches, _ = model.model.encoder.patch_embedding(pixels_10ch, waves)

    # Step 2: Pass the patches directly to the transformer manager.
    # It automatically handles the loop through all transformer layers and applies final normalization.
    features = model.model.encoder.transformer(patches)

print("\n" + "=" * 40)
print("✅ MISSION ACCOMPLISHED! Successfully extracted the Vector!")

# 6. Display Results
if isinstance(features, torch.Tensor):
    # Pool the token features (mean) to get a single 1024-dim identity vector for the whole image
    final_vector = features.mean(dim=1)

    print(f"🎯 Final Identity Vector Shape: {final_vector.shape}")
    print(f"🔢 First five values of the Vector:\n{final_vector[0, :5]}")
print("=" * 40)
