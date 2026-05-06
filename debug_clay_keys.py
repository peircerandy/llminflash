
import torch
import torch.nn as nn
import timm
import yaml
from box import Box
from accelerate import init_empty_weights

class MockTeacher(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_features = 512

timm.create_model = lambda *args, **kwargs: MockTeacher()

from claymodel.model import clay_mae_large

def debug_keys():
    meta = Box(yaml.safe_load('sentinel-2-l2a: {gsd: 10, rgb_indices: [2, 1, 0], bands: {mean: {}, std: {}, wavelength: {}}}'))
    mae_args = {'mask_ratio': 0.75, 'norm_pix_loss': False, 'shuffle': False, 'teacher': 'vit_large_patch14_224', 'dolls': [], 'doll_weights': []}
    
    with init_empty_weights():
        model = clay_mae_large(metadata=meta, patch_size=14, **mae_args)
    
    path = '/mnt/wsl/PHYSICALDRIVE0p3/hf_cache/models--made-with-clay--Clay/snapshots/70200ebcccdf67bf2a0cb9984c77ddee26c10ed2/v1.5/clay-v1.5.ckpt'
    # Use mmap and weights_only for safety
    ckpt = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    sd = ckpt.get('state_dict', ckpt)
    
    m_keys = [k for k in model.state_dict().keys() if 'transformer.layers.0.' in k]
    c_keys = [k for k in sd.keys() if 'transformer.layers.0.' in k]
    
    print("\n--- MODEL KEYS (layer 0) ---")
    for k in m_keys[:5]: print(k)
    
    print("\n--- CKPT KEYS (layer 0) ---")
    for k in c_keys[:5]: print(k)

if __name__ == "__main__":
    debug_keys()
