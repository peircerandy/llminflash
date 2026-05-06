import torch

# Use Clay's official package loader (ClayMAEModule)
from claymodel.module import ClayMAEModule

print("🚀 Starting Clay base model test...")

print("📥 Loading Clay via official claymodel package...")
# Load checkpoint weights from the CLAY directory (e.g. downloaded .ckpt file)
model_path = "CLAY/clay-v1.5.ckpt"

try:
    model = ClayMAEModule.load_from_checkpoint(
        model_path, metadata_path="configs/metadata.yaml", strict=False
    )
    model.eval()

    print("\n" + "=" * 40)
    print("✅ Test passed! Clay loaded from checkpoint (no Hugging Face required).")
    print("=" * 40)

    # Print encoder structure for inspection
    print("\n📊 Model encoder structure:")
    print(model.model.encoder)  # Encoder submodule

except Exception as e:
    print(f"❌ Load failed. Error:\n{e}")
