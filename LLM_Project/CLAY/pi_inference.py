import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from claymodel.module import ClayMAEModule
from peft import LoraConfig, get_peft_model
import time
import os
import glob

# 🚀 1. Force CPU mode (Raspberry Pi has no CUDA)
device = torch.device("cpu")
print("Initializing Raspberry Pi CPU inference mode...")

# 🚀 2. Prepare 10 EuroSAT class labels
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# 🚀 3. Load model (remove all CUDA-specific settings)
class ClayClassifierLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.clay_module = ClayMAEModule.load_from_checkpoint(
            "CLAY/clay-v1.5.ckpt",
            metadata_path="configs/metadata.yaml",
            strict=False,
            map_location=device,
        )
        self.clay_encoder = self.clay_module.model.encoder
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules="all-linear", bias="none"
        )
        self.clay_encoder = get_peft_model(self.clay_encoder, lora_config)
        self.classifier_head = nn.Linear(1024, num_classes)

    def forward(self, pixels_10ch, waves):
        patches, _ = self.clay_encoder.base_model.model.patch_embedding(
            pixels_10ch, waves
        )
        features = self.clay_encoder.base_model.model.transformer(patches)
        global_vector = features.mean(dim=1)
        return self.classifier_head(global_vector)


print("📥 Loading model weights (this may take tens of seconds on Raspberry Pi)...")
model = ClayClassifierLoRA(num_classes=10).to(device)
model.clay_encoder.load_adapter("CLAY/saved_lora_best", "default")
model.classifier_head.load_state_dict(
    torch.load(
        "CLAY/saved_lora_best/classifier_head.pth",
        map_location=device,
        weights_only=True,
    )
)
model.eval()  # Switch to evaluation mode

# 🚀 4. Image preprocessing and inference function
preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

WAVES = torch.tensor([665.0, 560.0, 490.0, 0, 0, 0, 0, 0, 0, 0], device=device)


def predict_image(image_path):
    print(f"👀 Analyzing image: {image_path}")
    start_time = time.time()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    tensor_img = preprocess(image).unsqueeze(
        0
    )  # Add batch dimension -> [1, 3, 224, 224]

    # Pad with 7 dummy channels
    padding = torch.zeros((1, 7, 224, 224), device=device)
    pixels_10ch = torch.cat([tensor_img, padding], dim=1)

    # Run inference (Raspberry Pi does not support autocast bfloat16)
    with torch.no_grad():
        logits = model(pixels_10ch, WAVES)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    end_time = time.time()

    print("-" * 30)
    print(f"🎯 Prediction: {CLASS_NAMES[predicted_idx.item()]}")
    print(f"📊 Confidence: {confidence.item() * 100:.2f}%")
    print(f"⏱️ Runtime: {end_time - start_time:.2f} s")
    print("-" * 30)


# ==========================================
# 🎯 Execution block
# ==========================================
if __name__ == "__main__":
    test_folder = "sample_tests"

    # Collect all JPG and PNG images in the folder
    image_paths = glob.glob(os.path.join(test_folder, "*.jpg"))

    # Sort by filename for cleaner output
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"⚠️ No images found in folder '{test_folder}'. Please check the path.")
    else:
        print(
            f"📂 Found {len(image_paths)} image(s). Model is resident in memory; starting batch prediction...\n"
        )

        # Total elapsed time for the entire batch
        total_start_time = time.time()

        for img_path in image_paths:
            try:
                predict_image(img_path)
            except Exception as e:
                print(f"❌ Error while processing {img_path}: {e}")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Print final statistics
        print("=" * 40)
        print("✅ Batch prediction completed.")
        print(f"🖼️ Total images processed: {len(image_paths)}")
        print(f"⏱️ Total prediction time: {total_duration:.2f} s")
        print(f"⚡ Average time per image: {(total_duration / len(image_paths)):.2f} s")
        print("=" * 40)
