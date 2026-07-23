import torch.nn as nn


class _ExtModule:
    def ms_deform_attn_forward(self, *args, **kwargs):
        raise RuntimeError('mmcv CUDA extension not available (shim)')
    def ms_deform_attn_backward(self, *args, **kwargs):
        raise RuntimeError('mmcv CUDA extension not available (shim)')


ext_module = _ExtModule()


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_dims = kwargs.get('embed_dims', 256)

    def forward(self, *args, **kwargs):
        raise RuntimeError('MultiScaleDeformableAttention CUDA op not available (shim)')
