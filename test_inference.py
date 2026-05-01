import torch
from chat import model, tokenizer
import sys

print("\n--- Running Coherent Inference Test (Slow Pass) ---")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

input_ids = inputs.input_ids
past_key_values = None

print(f"Prompt: {prompt}")
print("Generating 10 tokens...", flush=True)

generated_tokens = []

for i in range(10):
    print(f"\n[Python] Generating Token {i+1}/10...", flush=True)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
    
    word = tokenizer.decode(next_token_id[0])
    print(f"[Python] Token: '{word}'", flush=True)
    
    generated_tokens.append(word)
    input_ids = next_token_id
    past_key_values = outputs.past_key_values

print("\n" + "="*30)
print("Final Response:", prompt + "".join(generated_tokens))
print("="*30)
