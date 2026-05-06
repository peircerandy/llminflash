
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Clay Data Exploration: Earth Observation Datasets\n"
                                 "This notebook explores the datasets used to train and validate the Clay Foundation Model."),
        
        nbf.v4.new_code_cell("import torch\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
                             "from datasets import load_dataset\nfrom PIL import Image"),
        
        nbf.v4.new_markdown_cell("## 1. EuroSAT MSI (Sentinel-2)\n"
                                 "EuroSAT MSI consists of 27,000 labeled images with 13 spectral bands. "
                                 "Clay uses 10 of these bands (Sentinel-2 L2A format)."),
        
        nbf.v4.new_code_cell("ds_msi = load_dataset('blanchon/EuroSAT_MSI', split='train', streaming=True)\n"
                             "sample = next(iter(ds_msi))\n"
                             "img_raw = np.array(sample['image'])\n"
                             "print(f'Shape: {img_raw.shape}')\n"
                             "print(f'Max Value: {img_raw.max()}') # Usually 16-bit"),
        
        nbf.v4.new_code_cell("# Visualize RGB (Bands 4, 3, 2)\n"
                             "rgb = img_raw[[3, 2, 1], :, :].transpose(1, 2, 0).astype(float)\n"
                             "p2, p98 = np.percentile(rgb, (2, 98))\n"
                             "rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)\n\n"
                             "plt.figure(figsize=(6, 6))\n"
                             "plt.imshow(rgb)\n"
                             "plt.title(f'EuroSAT MSI RGB (B4-B3-B2) - Class: {sample[\"label\"]}')\n"
                             "plt.axis('off')\n"
                             "plt.show()"),
        
        nbf.v4.new_markdown_cell("## 2. EuroSAT SAR (Sentinel-1 Radar)\n"
                                 "SAR data uses radar backscatter (VV and VH polarizations). Clay can handle "
                                 "these by padding or using specific band embeddings."),
        
        nbf.v4.new_code_cell("ds_sar = load_dataset('blanchon/EuroSAT_SAR', split='train', streaming=True)\n"
                             "sample_sar = next(iter(ds_sar))\n"
                             "img_sar = np.array(sample_sar['image'])\n"
                             "print(f'SAR Shape: {img_sar.shape}')"),
        
        nbf.v4.new_code_cell("# Visualize SAR (VV Polarization)\n"
                             "vv = img_sar[0, :, :].astype(float)\n"
                             "p2, p98 = np.percentile(vv, (2, 98))\n"
                             "vv = np.clip((vv - p2) / (p98 - p2), 0, 1)\n\n"
                             "plt.figure(figsize=(6, 6))\n"
                             "plt.imshow(vv, cmap='gray')\n"
                             "plt.title('EuroSAT SAR (VV Polarization)')\n"
                             "plt.axis('off')\n"
                             "plt.show()")
    ]
    
    nb['cells'] = cells
    with open('clay_data_exploration.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Generated 'clay_data_exploration.ipynb'.")

if __name__ == "__main__":
    create_notebook()
