import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "llama3_8b/predictor_weights"
os.makedirs(SAVE_DIR, exist_ok=True)

D_MODEL, D_FFN, RANK = 4096, 14336, 128
EPOCHS = 3
DATA_SAMPLES = 500


class Llama3Predictor(nn.Module):
    def __init__(self, d_model, rank, d_ffn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, rank, bias=False),
            nn.ReLU(),
            nn.Linear(rank, d_ffn, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def fast_overnight_training():
    print(f"🚀 Launching Turbo Training Mode...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()
    model.eval()

    # Initialize 32 predictors and optimizers
    predictors = [Llama3Predictor(D_MODEL, RANK, D_FFN).cuda() for _ in range(32)]
    optimizers = [optim.Adam(p.parameters(), lr=1e-4) for p in predictors]
    criterion = nn.BCELoss()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_texts = [t for t in dataset["text"] if len(t) > 150][:DATA_SAMPLES]

    for epoch in range(EPOCHS):
        print(f"\n🌟 Epoch {epoch+1}/{EPOCHS}")

        for text in tqdm(train_texts, desc="Training All Layers"):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to("cuda")

            with torch.no_grad():
                # [CRITICAL] Run model ONCE to get ALL hidden states
                outputs = model(**inputs, output_hidden_states=True)

            # Now train each layer's predictor using the captured states
            for i in range(32):
                hidden_in = outputs.hidden_states[i].detach()
                layer = model.model.layers[i]

                with torch.no_grad():
                    # 1. Compute ground-truth SwiGLU activation
                    gate_out = layer.mlp.gate_proj(hidden_in)
                    up_out = layer.mlp.up_proj(hidden_in)
                    act = torch.abs(torch.nn.functional.silu(gate_out) * up_out)

                    # [Key upgrade] 2. Normalize activations to [0.0, 1.0]
                    # Per-token max neuron value
                    act_max = act.max(dim=-1, keepdim=True)[0]
                    # Add 1e-8 to avoid division by zero; yields per-token weight scores
                    target_score = act / (act_max + 1e-8)

                # 3. Train the predictor to match these continuous scores, not binary 0/1
                optimizers[i].zero_grad()
                pred = predictors[i](hidden_in.float())

                # [Key upgrade] 4. Use MSE loss to approximate the ground-truth scores
                loss = nn.MSELoss()(pred, target_score.float())
                loss.backward()
                optimizers[i].step()

        # Save all checkpoints after each epoch
        for i in range(32):
            torch.save(predictors[i].state_dict(), f"{SAVE_DIR}/layer_{i}.pt")
        print(f"✅ Epoch {epoch+1} checkpoints saved.")

    print(f"🎉 Turbo Training Complete! All 32 layers are ready.")


if __name__ == "__main__":
    fast_overnight_training()
