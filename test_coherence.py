import torch
from chat import model, tokenizer

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print(f"\n--- Testing Predictor Coherence (Fixed Engine) ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=15, 
        do_sample=False
    )

print("Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))
