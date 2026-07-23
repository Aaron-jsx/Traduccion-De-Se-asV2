import warnings

try:
    from torchvision.ops import roi_align as _tv_roi_align
except ImportError:
    _tv_roi_align = None


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=0, mode='avg', aligned=False):
    if _tv_roi_align is None:
        raise ImportError('torchvision is required for roi_align')
    if mode != 'avg':
        warnings.warn(f'torchvision.ops.roi_align only supports "avg" mode, got "{mode}"')
    return _tv_roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
