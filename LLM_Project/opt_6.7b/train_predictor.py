import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


# 1. Optimized Low-Rank Predictor architecture (N -> R -> M)
class LowRankPredictor(nn.Module):
    def __init__(self, d_model, rank, d_ffn):
        super().__init__()
        # Paper Figure 3(b): Bottleneck design to minimize overhead [cite: 142, 933]
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_ffn, bias=False)

    def forward(self, x):
        # Returns raw logits for BCEWithLogitsLoss (numerical stability) [cite: 146]
        return self.up(self.down(x))


def train():
    print("⏳ Initializing Expert Training on RTX 5080...")
    model_id = "facebook/opt-6.7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load 6.7B model in float16 to fit comfortably in 16GB VRAM [cite: 39, 718]
    model = OPTForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()
    model.eval()

    d_model, d_ffn = model.config.hidden_size, model.config.ffn_dim
    num_layers = len(model.model.decoder.layers)

    # Paper config: Rank 128 for layers 0-27, Rank 1024 for layers 28-31 [cite: 705, 706]
    print("🛠️ Creating Predictors with Sensitive Layer upgrades...")
    predictors = nn.ModuleList(
        [
            LowRankPredictor(d_model, 128 if i < 28 else 1024, d_ffn).cuda()
            for i in range(num_layers)
        ]
    )

    optimizer = torch.optim.Adam(predictors.parameters(), lr=1e-4)

    # Prepare dataset: 10,000 samples as per paper requirements [cite: 149]
    print("📚 Preparing Dataset...")
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset_texts = [t for t in raw_dataset["text"] if len(t.strip()) > 100][:10000]
    train_loader = DataLoader(dataset_texts, batch_size=4, shuffle=True)

    ffn_inputs, relu_outputs = {}, {}

    def get_ffn_input_hook(idx):
        def hook(m, i, o):
            ffn_inputs[idx] = i[0].detach()

        return hook

    def get_relu_output_hook(idx):
        def hook(m, i, o):
            relu_outputs[idx] = (o > 0).float().detach()

        return hook

    handles = []
    for i, layer in enumerate(model.model.decoder.layers):
        handles.append(layer.fc1.register_forward_hook(get_ffn_input_hook(i)))
        handles.append(
            layer.activation_fn.register_forward_hook(get_relu_output_hook(i))
        )

    print("🚀 Training (Monitoring Loss and Recall for 97% Sparsity)...")

    for epoch in range(2):  # 2 Epochs as per paper [cite: 149]
        epoch_loss, epoch_recall = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_texts in pbar:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to("cuda")
            optimizer.zero_grad()

            with torch.no_grad():
                model(**inputs)

            batch_total_loss = 0
            batch_total_recall = 0

            for i in range(num_layers):
                x, y_true = ffn_inputs[i].float(), relu_outputs[i]

                # Balanced Loss: pos_weight combats 97% sparsity [cite: 109, 146]
                num_pos = y_true.sum()
                num_neg = y_true.numel() - num_pos
                pos_weight = (num_neg / (num_pos + 1e-5)).clamp(max=100)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                y_pred_logits = predictors[i](x)
                loss = criterion(y_pred_logits, y_true)
                batch_total_loss += loss

                # --- Recall Calculation (Monitoring Predictor Accuracy) [cite: 569, 584] ---
                with torch.no_grad():
                    # Predict active if logit > 0 (Sigmoid > 0.5) [cite: 134, 154]
                    preds_binary = (y_pred_logits > 0).float()
                    true_positives = (preds_binary * y_true).sum()
                    recall = true_positives / (y_true.sum() + 1e-8)
                    batch_total_recall += recall.item()

                # Cleanup layer memory immediately to save VRAM for 5080
                del ffn_inputs[i], relu_outputs[i]

            batch_total_loss.backward()
            optimizer.step()

            avg_loss = batch_total_loss.item() / num_layers
            avg_recall = batch_total_recall / num_layers
            epoch_loss += avg_loss
            epoch_recall += avg_recall

            pbar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "recall": f"{avg_recall * 100:.1f}%"}
            )

            ffn_inputs.clear()
            relu_outputs.clear()
            torch.cuda.empty_cache()
        print(
            f"📊 Epoch {epoch+1} Results: Loss: {epoch_loss/len(train_loader):.4f} | Recall: {(epoch_recall/len(train_loader))*100:.2f}%"
        )

    for h in handles:
        h.remove()
    os.makedirs("predictor_weights", exist_ok=True)
    for i, pred in enumerate(predictors):
        torch.save(pred.state_dict(), f"predictor_weights/layer_{i}.pt")
    print("✅ Training finished. Predictor weights saved successfully.")


if __name__ == "__main__":
    train()
