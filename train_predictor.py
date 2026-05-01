import torch
import torch.nn as nn
import torch.optim as optim
from transformers import OPTForCausalLM, AutoTokenizer
import os
import struct
from datasets import load_dataset

# Constants tailored to OPT-6.7B
MODEL_ID = "facebook/opt-6.7b"
CACHE_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_cache"
PREDICTOR_OUT = "/mnt/wsl/PHYSICALDRIVE0p3/opt_6_7b_predictors.bin"
OFFLOAD_PATH = "/mnt/wsl/PHYSICALDRIVE0p3/hf_offload"

HIDDEN_SIZE = 4096
FFN_DIM = 16384
RANK = 128  # The compression bottleneck size (R value from the paper)
NUM_LAYERS = 32

class LowRankPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Compresses the 4096-dimensional token down to 128 dimensions
        self.down = nn.Linear(HIDDEN_SIZE, RANK, bias=False)
        # Expands the 128 dimensions to guess the 16384 activations
        self.up = nn.Linear(RANK, FFN_DIM, bias=False)

    def forward(self, x):
        # Sigmoid outputs a probability between 0.0 and 1.0 for each neuron
        return torch.sigmoid(self.up(self.down(x)))

def get_real_hidden_states(model, tokenizer, layer_idx, num_samples=10000):
    """
    Passes real English text from the C4 dataset through the model 
    to gather genuine hidden states for training.
    """
    print(f"  Fetching C4 dataset samples (OVERNIGHT MODE)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # OVERNIGHT HACK: Grab 500 random articles instead of 15 to get a massive variety of English words
    texts = [next(iter(dataset))["text"] for _ in range(500)]
    
    fc1_inputs = []
    fc1_targets = [] # We will store targets directly to save RAM
    total_tokens = 0
    
    # Use a forward hook to capture the EXACT input and output of the FC1 layer.
    def hook_fn(module, args, output):
        # Cast input to float16 to save 50% RAM
        fc1_inputs.append(args[0].detach().cpu().half())
        # Immediately convert to 0/1 integers to save 75% RAM
        fc1_targets.append((output.detach().cpu() > 0).to(torch.int8))
        
    # Attach the wiretap
    fc1_layer = model.model.decoder.layers[layer_idx].fc1
    hook_handle = fc1_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        # MEMORY FIX: We cannot pass 500 articles into a 6GB GPU all at once!
        # We must process them in small batches to prevent an OOM crash.
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
            
            # Let the model run the small batch
            model(**inputs)
            
            # Count how many tokens we just processed
            total_tokens += inputs["input_ids"].numel()
            
            # Immediately clear the GPU memory before the next loop
            del inputs
            torch.cuda.empty_cache()
            
            # MEMORY FIX: Stop passing articles if we already have the 10,000 tokens we need!
            if total_tokens >= num_samples:
                break
            
    # Remove the wiretap so it doesn't leak memory on the next loop
    hook_handle.remove()
    
    # Flatten across batch and sequence length
    hidden = torch.cat(fc1_inputs, dim=0).view(-1, HIDDEN_SIZE)
    targets = torch.cat(fc1_targets, dim=0).view(-1, FFN_DIM)
    
    if hidden.shape[0] > num_samples:
        hidden = hidden[:num_samples]
        targets = targets[:num_samples]
        
    # Move the extracted data to the GPU for the predictor's training loop
    return hidden.float().to("cuda:0"), targets.float().to("cuda:0")

def train_predictors():
    print("Loading OPT-6.7B weights (distributing across RAM and VRAM)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FIX: Use device_map="auto" so HuggingFace handles the CPU/GPU splitting safely!
    model = OPTForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        cache_dir=CACHE_PATH,
        offload_folder=OFFLOAD_PATH,
        output_hidden_states=True
    )
    
    # --- ENTERPRISE AUTO-RESUME LOGIC ---
    # Calculate exact bytes per layer (Down weight + Up weight in float32 = ~10.4 MB)
    BYTES_PER_LAYER = (HIDDEN_SIZE * RANK * 4) + (RANK * FFN_DIM * 4)
    
    start_layer = 0
    file_mode = "wb"
    
    if os.path.exists(PREDICTOR_OUT):
        current_size = os.path.getsize(PREDICTOR_OUT)
        start_layer = current_size // BYTES_PER_LAYER
        if 0 < start_layer < NUM_LAYERS:
            print(f"\n[AUTO-RESUME] Found existing file. Protecting data and resuming at Layer {start_layer}/{NUM_LAYERS}...")
            file_mode = "ab" # Append binary so we don't overwrite!
        elif start_layer >= NUM_LAYERS:
            print("\nAll predictor layers are already fully trained!")
            return
            
    # Open safely in append mode if resuming
    out_file = open(PREDICTOR_OUT, file_mode)
    
    criterion = nn.BCELoss() # Binary Cross Entropy (Perfect for Yes/No predictions)

    for i in range(start_layer, NUM_LAYERS):
        print(f"\n--- Training Predictor for Layer {i}/{NUM_LAYERS} ---")
        
        predictor = LowRankPredictor().to("cuda:0")
        optimizer = optim.Adam(predictor.parameters(), lr=0.001)
        
        # Gather the ground-truth training data
        x_train, y_train = get_real_hidden_states(model, tokenizer, i)
        
        # OVERNIGHT HACK: Train the lightweight predictor for 1000 epochs to force it to learn!
        epochs = 1000
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = predictor(x_train)
            
            # Penalize the predictor if it guesses a neuron will fire when it shouldn't
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
                
        # Export the trained predictor weights for the C++ engine
        print(f"Exporting Predictor weights for Layer {i} to SSD...")
        down_bytes = predictor.down.weight.data.cpu().float().numpy().tobytes()
        up_bytes = predictor.up.weight.data.cpu().float().numpy().tobytes()
        
        out_file.write(down_bytes)
        out_file.write(up_bytes)
        
    out_file.close()
    print("\nAll predictors trained and exported successfully!")

if __name__ == "__main__":
    train_predictors()