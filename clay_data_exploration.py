
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Clay Data Exploration: Authentic Earth Observation\n"
                                 "This notebook explores the datasets used for validating the Clay Foundation Model. "
                                 "We compare standard RGB satellite imagery with Synthetic Aperture Radar (SAR)."),
        
        nbf.v4.new_code_cell("import torch\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
                             "import torchvision\nfrom PIL import Image\nfrom datasets import load_dataset"),
        
        nbf.v4.new_markdown_cell("## 1. EuroSAT RGB (Standard Visual Satellite)\n"
                                 "Used for the primary benchmark visualization. Clean 3-channel imagery."),
        
        nbf.v4.new_code_cell("dataset_rgb = torchvision.datasets.EuroSAT(root='CLAY/data', download=True)\n"
                             "img, label = dataset_rgb[4] # Industrial example\n"
                             "plt.imshow(img)\n"
                             "plt.title(f'EuroSAT RGB - Class: Industrial')\n"
                             "plt.axis('off')\n"
                             "plt.show()"),
        
        nbf.v4.new_markdown_cell("## 2. EuroSAT SAR (Sentinel-1 Radar)\n"
                                 "SAR captures physical structure and moisture. It works through clouds and at night."),
        
        nbf.v4.new_code_cell("# Using the verified HuggingFace path for SAR\n"
                             "ds_sar = load_dataset('wangyi111/EuroSAT-SAR', split='train', streaming=True)\n"
                             "sample_sar = next(iter(ds_sar))\n"
                             "img_sar = np.array(sample_sar['image'])\n"
                             "print(f'SAR Shape: {img_sar.shape}')"),
        
        nbf.v4.new_code_cell("# Visualize SAR VV and VH polarizations\n"
                             "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))\n"
                             "vv = img_sar[0, :, :].astype(float)\n"
                             "vh = img_sar[1, :, :].astype(float)\n\n"
                             "ax1.imshow(vv, cmap='gray')\n"
                             "ax1.set_title('VV Polarization')\n"
                             "ax2.imshow(vh, cmap='gray')\n"
                             "ax2.set_title('VH Polarization')\n"
                             "plt.show()"),
        
        nbf.v4.new_markdown_cell("## 3. Clay 10-Channel Preparation\n"
                                 "Clay expects 10 channels. We pad the SAR/RGB data to fit.")
    ]
    
    nb['cells'] = cells
    with open('clay_data_exploration.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Generated 'clay_data_exploration.ipynb'.")

if __name__ == "__main__":
    create_notebook()
