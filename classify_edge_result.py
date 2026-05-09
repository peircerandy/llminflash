import torch
import numpy as np
import os
import sys

def classify_edge(token_path, proto_path="benchmark_results/class_prototypes.pt"):
    CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVeg", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    
    if not os.path.exists(token_path):
        print(f"Error: Telemetry file {token_path} not found.")
        return
    if not os.path.exists(proto_path):
        print(f"Error: Prototypes {proto_path} not found. Run main benchmark first.")
        return
        
    # Load Telemetry from Edge
    emb = torch.from_numpy(np.load(token_path))
    # Load Prototypes from Laptop
    centroids = torch.load(proto_path)
    
    # Cosine Similarity
    sims = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), centroids)
    probs = torch.softmax(sims * 15, dim=0)
    conf, pred = torch.max(probs, dim=0)
    
    print("\n--- Edge Telemetry Classification ---")
    print(f"File: {token_path}")
    print(f"Prediction: {CLASS_NAMES[pred.item()]}")
    print(f"Confidence: {conf.item()*100:.2f}%")
    print("--------------------------------------\n")
    
    # Save results for graphing
    res_path = token_path.replace(".npy", "_classification.json")
    import json
    with open(res_path, "w") as f:
        json.dump({
            "prediction": CLASS_NAMES[pred.item()],
            "confidence": float(conf.item()),
            "timestamp": os.path.getmtime(token_path)
        }, f, indent=4)
    print(f"Results saved to {res_path} for graphing.")

if __name__ == "__main__":
    path = "edge_deployment/edge_output_cls_token.npy"
    if len(sys.argv) > 1: path = sys.argv[1]
    classify_edge(path)
