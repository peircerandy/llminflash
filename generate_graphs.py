import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_graphs():
    """
    Generates graphs comparing performance (Tokens/Sec) and Accuracy
    across different architectures and execution modes.
    """
    # 1. Speed Comparison Graph (Tokens per Second)
    modes = ['Standard\n(Swap)', 'Quantized\n(4-bit)', 'Naive\n(100% SSD)', 'Oracle\n(Exact Math)', 'Predictor\n(Apple ML)', 'Speculative\n(+Draft)']
    # Simulated TPS based on OPT-6.7B on constrained hardware (e.g. 8GB RAM, PCIe Gen 4)
    tps = [0.8, 12.5, 1.2, 6.5, 9.8, 14.2] 
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#c2c2f0', '#ffb3e6']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(modes, tps, color=colors)
    plt.title('LLM Inference Speeds (OPT-6.7B on Constrained RAM)', fontsize=14, pad=20)
    plt.ylabel('Tokens Per Second (Higher is Better)', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    print("Generated performance_comparison.png")
    plt.close()

    # 2. Accuracy / Coherence Comparison Graph
    # Comparing how much performance degradation occurs due to the method
    acc_modes = ['Full Precision\n(Baseline)', 'Quantized\n(4-bit)', 'Predictor\n(Top-K only)', 'Predictor\n(+Threshold)']
    # Simulated percentage of coherent responses / benchmark accuracy
    accuracy = [100.0, 92.5, 89.0, 98.5]
    acc_colors = ['#cccccc', '#ffcc99', '#ff9999', '#99ff99']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(acc_modes, accuracy, color=acc_colors)
    plt.title('Model Accuracy vs Execution Mode', fontsize=14, pad=20)
    plt.ylabel('Relative Accuracy / Coherence (%)', fontsize=12)
    plt.ylim(80, 105)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300)
    print("Generated accuracy_comparison.png")
    plt.close()

if __name__ == "__main__":
    generate_comparison_graphs()
