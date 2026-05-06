import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split  # 🚀 Added random_split
import torchvision.transforms as T
from torchvision.datasets import EuroSAT
from claymodel.module import ClayMAEModule
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import os

# ==========================================
# 1. Setup & Hyperparameters
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing LoRA Fine-Tuning pipeline on {device}...")

BATCH_SIZE = 8  # Batch size tuned for RTX 5080
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_CLASSES = 10


# ==========================================
# 2. Custom Model (unchanged)
# ==========================================
class ClayClassifierLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print("📥 Loading Clay base model...")
        self.clay_module = ClayMAEModule.load_from_checkpoint(
            "CLAY/clay-v1.5.ckpt", metadata_path="configs/metadata.yaml", strict=False
        )
        self.clay_encoder = self.clay_module.model.encoder

        print("💉 Injecting LoRA adapters into Transformer blocks...")
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


model = ClayClassifierLoRA(num_classes=NUM_CLASSES).to(device)

# ==========================================
# 3. 🚀 Data Preparation (80/20 split)
# ==========================================
preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print("🌍 Loading EuroSAT dataset...")
full_dataset = EuroSAT(root="CLAY/data", download=False, transform=preprocess)

# 🚀 Split dataset into Train (80%) and Val (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
# Fixed RNG seed for a reproducible train/val split
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=generator
)

# Two DataLoaders (num_workers=0 avoids worker hangs on some setups)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0
)

WAVES = torch.tensor([665.0, 560.0, 490.0, 0, 0, 0, 0, 0, 0, 0], device=device)

# ==========================================
# 4. Training & Validation Loop
# ==========================================
criterion = nn.CrossEntropyLoss()
trainable_params = list(model.clay_encoder.parameters()) + list(
    model.classifier_head.parameters()
)
optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
scaler = torch.amp.GradScaler("cuda")

# 🚀 Track best (lowest) validation loss
best_val_loss = float("inf")

print("\n" + "=" * 50)
print("🔥 STARTING LORA FINE-TUNING (WITH VALIDATION) 🔥")
print("=" * 50)

for epoch in range(EPOCHS):
    # -------------------------
    # 🥊 First half: training phase
    # -------------------------
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        padding = torch.zeros((images.size(0), 7, 224, 224), device=device)
        pixels_10ch = torch.cat([images, padding], dim=1)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(pixels_10ch, WAVES)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels).item()
        train_total += labels.size(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # -------------------------
    # 🧪 Second half: validation phase (held-out eval)
    # -------------------------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    # 🚀 No gradients during eval — use no_grad()
    with torch.no_grad():
        val_pbar = tqdm(
            val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ", leave=False
        )
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            padding = torch.zeros((images.size(0), 7, 224, 224), device=device)
            pixels_10ch = torch.cat([images, padding], dim=1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(pixels_10ch, WAVES)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()
            val_total += labels.size(0)
            val_loss += loss.item()

    # Epoch metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = (val_correct / val_total) * 100

    print(
        f"📈 [Epoch {epoch+1} Summary] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    # -------------------------
    # 💾 Save best checkpoint (lowest val loss)
    # -------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"   🌟 New best val loss ({best_val_loss:.4f}) — saving weights...")
        os.makedirs("CLAY/saved_lora_best", exist_ok=True)
        model.clay_encoder.save_pretrained("CLAY/saved_lora_best")
        torch.save(
            model.classifier_head.state_dict(),
            "CLAY/saved_lora_best/classifier_head.pth",
        )
    print("-" * 50)

print("✅ Training finished. Best model saved under 'CLAY/saved_lora_best/'.")
