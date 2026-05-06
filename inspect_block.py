
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

def inspect():
    meta = Box(yaml.safe_load('sentinel-2-l2a: {gsd: 10, rgb_indices: [2, 1, 0], bands: {mean: {}, std: {}, wavelength: {}}}'))
    mae_args = {'mask_ratio': 0.75, 'patch_size': 14, 'norm_pix_loss': False, 'shuffle': False, 'teacher': 'vit_large_patch16_224', 'dolls': [], 'doll_weights': []}
    model = clay_mae_large(metadata=meta, **mae_args)
    block = model.encoder.transformer.layers[0][1]
    print(f"BLOCK TYPE: {type(block)}")
    print(f"BLOCK CONTENT: {block}")

if __name__ == "__main__":
    inspect()
