import numpy as np
import os

files = [f for f in os.listdir("benchmark_results") if f.endswith(".npy")]
for f in files:
    data = np.load(os.path.join("benchmark_results", f))
    print(f"{f}: shape={data.shape}, dtype={data.dtype}, size={data.nbytes}")
    if data.ndim == 2:
        print(f"  Sample values: {data.flatten()[:5]}")
