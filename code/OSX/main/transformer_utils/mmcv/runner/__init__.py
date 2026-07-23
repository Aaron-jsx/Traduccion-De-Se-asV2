import functools
import torch.nn as nn
from ..utils import Registry
from .base_module import BaseModule


def get_dist_info():
    return 0, False


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    import torch
    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


class OptimizerHook:
    pass


class DistEvalHook:
    pass


class EvalHook:
    pass


class Hook:
    pass


HOOKS = Registry('hooks')


def build_optimizer(*args, **kwargs):
    return None


def auto_fp16(func=None, apply_to=None):
    if func is not None:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return lambda f: auto_fp16(f, apply_to=apply_to)


def force_fp32(func=None, apply_to=None):
    if func is not None:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return lambda f: force_fp32(f, apply_to=apply_to)


def _stub(*args, **kwargs):
    return None


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    import warnings as _w
    _w.warn(f'mmcv.runner.{name} is not implemented (shim stub)')
    return _stub


class DefaultOptimizerConstructor:
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        pass
