
import torch
import torch.nn as nn
import timm
import yaml
from box import Box

class MockTeacher(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_features = 512

timm.create_model = lambda *args, **kwargs: MockTeacher()

from claymodel.model import clay_mae_large

def check():
    meta = Box(yaml.safe_load('sentinel-2-l2a: {gsd: 10, rgb_indices: [2, 1, 0], bands: {mean: {}, std: {}, wavelength: {}}}'))
    mae_args = {'mask_ratio': 0.75, 'patch_size': 14, 'norm_pix_loss': False, 'shuffle': False, 'teacher': 'vit_large', 'dolls': [], 'doll_weights': []}
    
    model = clay_mae_large(metadata=meta, **mae_args)
    # Check if layers[0][0] has norm
    # Architecture: Encoder.transformer.layers[i] is [AttentionBlock, MLPBlock]
    # AttentionBlock is a Sequential/ModuleList containing norm and attn?
    
    l0 = model.encoder.transformer.layers[0][0]
    print(f"Layer 0[0] type: {type(l0)}")
    # In Clay, transformer.layers[i][0] is AttentionBlock
    # AttentionBlock has .norm and .fn (Attention)
    print(f"Norm weight sample: {l0.norm.weight[:3]}")

    model.to_empty(device="cpu")
    print(f"After to_empty, Norm weight sample: {l0.norm.weight[:3]}")

if __name__ == "__main__":
    check()
