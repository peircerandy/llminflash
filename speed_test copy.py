import torch
import time
from chat import model, tokenizer

prompt = "The weather today is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print("Generating 1 token...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1)
end = time.time()

print(f"Time for 1 token: {end - start:.2f} seconds")
print("Response:", tokenizer.decode(outputs[0]))
