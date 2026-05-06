import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import EuroSAT
from claymodel.module import ClayMAEModule

# ==========================================
# 1. Setup & Hyperparameters
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing training pipeline on {device}...")

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 3
NUM_CLASSES = 10  # EuroSAT has 10 classes


# ==========================================
# 2. Custom Model: Clay + Classification Head
# ==========================================
class ClayClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load the frozen pre-trained Clay model
        print("📥 Loading Clay base model...")
        self.clay_module = ClayMAEModule.load_from_checkpoint(
            "CLAY/clay-v1.5.ckpt", metadata_path="configs/metadata.yaml", strict=False
        )
        self.clay_encoder = self.clay_module.model.encoder

        # FREEZE the base model (Linear Probing)
        for param in self.clay_encoder.parameters():
            param.requires_grad = False

        # Add a trainable classification head (1024 -> 10)
        self.classifier_head = nn.Linear(1024, num_classes)

    def forward(self, pixels_10ch, waves):
        # 1. Feature Extraction (The manual path we PERFECTED)
        with torch.no_grad():  # Ensure base model doesn't track gradients
            patches, _ = self.clay_encoder.patch_embedding(pixels_10ch, waves)

            # 🚀 THE FIX: Let the transformer handle its own layers!
            features = self.clay_encoder.transformer(patches)

            # Global Average Pooling [Batch, Tokens, 1024] -> [Batch, 1024]
            global_vector = features.mean(dim=1)

        # 2. Classification (This part is trainable!)
        logits = self.classifier_head(global_vector)
        return logits


model = ClayClassifier(num_classes=NUM_CLASSES).to(device)

# ==========================================
# 3. Data Preparation
# ==========================================
preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print("🌍 Loading EuroSAT dataset...")
dataset = EuroSAT(root="CLAY/data", download=False, transform=preprocess)
# Use a DataLoader to feed data in batches
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Standard wavelengths for Sentinel-2 RGB
WAVES = torch.tensor([665.0, 560.0, 490.0, 0, 0, 0, 0, 0, 0, 0], device=device)

# ==========================================
# 4. Training Loop
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier_head.parameters(), lr=LEARNING_RATE)

print("\n" + "=" * 40)
print("🔥 STARTING LINEAR PROBING TRAINING 🔥")
print("=" * 40)

model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # We will only run 10 batches per epoch for this quick test
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 10:
            break

        images, labels = images.to(device), labels.to(device)

        # Pad 3 RGB channels to 10 channels for Clay
        padding = torch.zeros((BATCH_SIZE, 7, 224, 224), device=device)
        pixels_10ch = torch.cat([images, padding], dim=1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(pixels_10ch, WAVES)

        # Calculate Loss & Accuracy
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        # Backward pass & optimize ONLY the classification head
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(
            f"   Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/10] | Loss: {loss.item():.4f}"
        )

    # Epoch Summary
    epoch_acc = (correct_predictions / total_samples) * 100
    print(
        f"📈 Epoch {epoch+1} Summary -> Avg Loss: {total_loss/10:.4f} | Accuracy: {epoch_acc:.2f}%\n"
    )

print("✅ Training test complete! The pipeline is fully operational.")
