
import torch
import torch.nn as nn
from claymodel.module import ClayMAEModule
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import numpy as np

# Configuration
PREDICTOR_BIN = "Clay_predictors.bin"
PREDICTOR_META = "Clay_predictors.json"
CKPT_PATH = "/mnt/trainer-disk/clay-v1.5.ckpt" # Update if needed
SAMPLES = 100

class LowRankPredictor(nn.Module):
    def __init__(self, d_model, rank, d_ffn):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_ffn, bias=False)
    def forward(self, x):
        return torch.sigmoid(self.up(torch.relu(self.down(x))))

def validate():
    print("--- Clay Predictor Validator ---")
    
    # 1. Load Metadata
    with open(PREDICTOR_META, "r") as f:
        meta = json.load(f)
    d_model, d_ffn, num_layers, rank = meta["hidden_size"], meta["ffn_dim"], meta["num_layers"], meta["rank"]

    # 2. Load Model
    print("Loading Clay Model...")
    model = ClayMAEModule.load_from_checkpoint(CKPT_PATH).cuda()
    model.eval()

    # 3. Load Predictors
    print(f"Loading {num_layers} Predictors from {PREDICTOR_BIN}...")
    predictors = [LowRankPredictor(d_model, rank, d_ffn).cuda() for _ in range(num_layers)]
    
    with open(PREDICTOR_BIN, "rb") as f:
        for p in predictors:
            down_w = np.frombuffer(f.read(d_model * rank * 4), dtype=np.float32).reshape(rank, d_model)
            up_w = np.frombuffer(f.read(rank * d_ffn * 4), dtype=np.float32).reshape(d_ffn, rank)
            p.down.weight.data = torch.from_numpy(down_w).cuda()
            p.up.weight.data = torch.from_numpy(up_w).cuda()

    # 4. Hooks
    captured_inputs = {}
    captured_acts = {}
    
    def get_input_hook(idx):
        def hook(m, i, o): captured_inputs[idx] = i[0].detach().float()
        return hook
        
    def get_act_hook(idx):
        def hook(m, i, o):
            act_data = o[0] if isinstance(o, tuple) else o
            captured_acts[idx] = (act_data > 0).float().detach()
        return hook

    # Hook into FeedForward layers
    layers = model.model.encoder.transformer.layers
    for i in range(num_layers):
        mlp_mod = layers[i][1]
        act_mod = None
        for sub in mlp_mod.net:
            if isinstance(sub, (nn.GELU, nn.ReLU, nn.SiLU)):
                act_mod = sub
                break
        if mlp_mod and act_mod:
            mlp_mod.register_forward_hook(get_input_hook(i))
            act_mod.register_forward_hook(get_act_hook(i))

    # 5. Validation Loop
    dataset = load_dataset("blanchon/EuroSAT_MSI", split="train", streaming=True)
    
    total_recall = [] # How many true activations we found
    total_precision = [] # How many of our guesses were right
    
    print(f"Running validation on {SAMPLES} samples...")
    for i, sample in enumerate(tqdm(dataset, total=SAMPLES)):
        if i >= SAMPLES: break
        
        # Simple preprocessing for validation
        img = torch.tensor(sample['image']).float()
        if img.dim() == 3: img = img.unsqueeze(0)
        if img.shape[1] > 10: img = img[:, :10, :, :]
        if img.shape[-1] != 224:
            import torchvision.transforms as T
            img = T.Resize((224, 224))(img)
        
        img = img.cuda()
        datacube = {
            "pixels": img, "time": torch.zeros((1,4), device="cuda"),
            "latlon": torch.zeros((1,4), device="cuda"), "platform": ["sentinel-2-l2a"]
        }

        with torch.no_grad():
            model(datacube)
            
            for l_idx in range(num_layers):
                x = captured_inputs[l_idx]
                y_true = captured_acts[l_idx].view(-1)
                
                # Predict
                y_pred_prob = predictors[l_idx](x).view(-1)
                y_pred = (y_pred_prob > 0.5).float() # 0.5 threshold
                
                # Metrics
                true_pos = (y_pred * y_true).sum()
                actual_pos = y_true.sum()
                predicted_pos = y_pred.sum()
                
                if actual_pos > 0:
                    total_recall.append((true_pos / actual_pos).item())
                if predicted_pos > 0:
                    total_precision.append((true_pos / predicted_pos).item())
                    
    print("\n--- Final Results ---")
    print(f"Mean Recall (Hit Rate): {np.mean(total_recall):.2%}")
    print(f"Mean Precision: {np.mean(total_precision):.2%}")
    print("----------------------")

if __name__ == "__main__":
    validate()
