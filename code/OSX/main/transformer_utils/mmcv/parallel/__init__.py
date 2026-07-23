from ..utils import Registry


MODULE_WRAPPERS = Registry('module wrappers')


class MMDistributedDataParallel:
    pass


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    import warnings as _w
    _w.warn(f'mmcv.parallel.{name} is not implemented (shim stub)')
    return _stub


class _stub:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return None
    def __getattr__(self, name): return self
