import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import EuroSAT
from claymodel.module import ClayMAEModule
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Ensure outputs directory exists
os.makedirs("CLAY/outputs", exist_ok=True)

# Saved figure path (under outputs)
output_filename = "CLAY/outputs/confusion_matrix.png"
plt.savefig(output_filename, dpi=300)

# ==========================================
# 1. Setup & Environment
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing Inference Pipeline on {device}...")

BATCH_SIZE = 16
NUM_CLASSES = 10


# ==========================================
# 2. Load the Fine-Tuned Model (The "Best Brain")
# ==========================================
class ClayClassifierLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. Load base Clay encoder
        self.clay_module = ClayMAEModule.load_from_checkpoint(
            "CLAY/clay-v1.5.ckpt", metadata_path="configs/metadata.yaml", strict=False
        )
        self.clay_encoder = self.clay_module.model.encoder

        # 2. Prepare LoRA adapter slots
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none",
        )
        self.clay_encoder = get_peft_model(self.clay_encoder, lora_config)
        self.classifier_head = nn.Linear(1024, num_classes)

    def forward(self, pixels_10ch, waves):
        patches, _ = self.clay_encoder.base_model.model.patch_embedding(
            pixels_10ch, waves
        )
        features = self.clay_encoder.base_model.model.transformer(patches)
        global_vector = features.mean(dim=1)
        logits = self.classifier_head(global_vector)
        return logits


# Initialize model architecture
model = ClayClassifierLoRA(num_classes=NUM_CLASSES).to(device)

print("📥 Loading saved weights from 'saved_lora_best'...")
# 3. Load trained LoRA weights into adapters
model.clay_encoder.load_adapter("CLAY/saved_lora_best", "default")
# 4. Load trained classifier head weights
model.classifier_head.load_state_dict(
    torch.load("CLAY/saved_lora_best/classifier_head.pth", weights_only=True)
)

# ⚠️ Eval mode: disables dropout and other train-only behavior
model.eval()

# ==========================================
# 3. Load the Validation Data (The "New Questions")
# ==========================================
preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print("🌍 Reconstructing the Validation Dataset...")
full_dataset = EuroSAT(root="CLAY/data", download=False, transform=preprocess)
class_names = full_dataset.classes  # EuroSAT class names

# Same RNG seed as training so the validation split matches
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# No shuffle needed for inference
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
WAVES = torch.tensor([665.0, 560.0, 490.0, 0, 0, 0, 0, 0, 0, 0], device=device)

# ==========================================
# 4. Inference Loop (The Final Exam)
# ==========================================
print("\n" + "=" * 50)
print("🔥 STARTING THE FINAL EXAM (EVALUATION) 🔥")
print("=" * 50)

all_preds = []
all_labels = []
correct_predictions = 0
total_samples = 0

with torch.no_grad():  # No gradient updates during inference
    # Optional AMP for faster inference on CUDA
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pbar = tqdm(val_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            padding = torch.zeros((images.size(0), 7, 224, 224), device=device)
            pixels_10ch = torch.cat([images, padding], dim=1)

            outputs = model(pixels_10ch, WAVES)
            _, preds = torch.max(outputs, 1)

            # Collect predictions and labels for the confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

# Final accuracy
final_acc = (correct_predictions / total_samples) * 100
print(f"\n🎯 FINAL VALIDATION ACCURACY: {final_acc:.2f}%")

# ==========================================
# 5. Plotting the Confusion Matrix
# ==========================================
print("📊 Generating Confusion Matrix plot...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
# Seaborn heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title(f"EuroSAT Confusion Matrix (Accuracy: {final_acc:.2f}%)", fontsize=14)
plt.xticks(rotation=45, ha="right")  # Rotate x labels to avoid overlap
plt.tight_layout()

# Save figure to disk
output_filename = "CLAY/outputs/confusion_matrix.png"
plt.savefig(output_filename, dpi=300)
print(f"✅ Confusion matrix successfully saved to '{output_filename}'!")
