import inspect
import functools


class Registry:
    def __init__(self, name, build_func=None, parent=None):
        self._name = name
        self._module_dict = {}
        self._build_func = build_func
        self._parent = parent

    def get(self, key):
        cls = self._module_dict.get(key)
        if cls is None and self._parent is not None:
            cls = self._parent.get(key)
        if cls is None:
            raise KeyError(f'{key} is not registered in registry {self._name}')
        return cls

    def register_module(self, cls=None, name=None, force=False):
        if cls is None:
            return lambda c: self.register_module(c, name=name, force=force)
        names = name if isinstance(name, (list, tuple)) else [name or cls.__name__]
        for n in names:
            self._module_dict[n] = cls
        return cls

    def build(self, cfg, *args, **kwargs):
        if self._build_func is not None:
            return self._build_func(cfg, *args, **kwargs, registry=self)
        return build_from_cfg(cfg, self, *args, **kwargs)

    def __contains__(self, key):
        if key in self._module_dict:
            return True
        if self._parent is not None:
            return key in self._parent
        return False


def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        return cfg
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    else:
        obj_cls = obj_type
    if default_args:
        for key, val in default_args.items():
            args.setdefault(key, val)
    return obj_cls(**args)


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


class ext_loader:
    @staticmethod
    def load_ext(name, funcs):
        import warnings
        warnings.warn(f'mmcv CUDA extension "{name}" not available (shim)')
        class _DummyExt:
            pass
        return _DummyExt()


def get_logger(name):
    import logging
    return logging.getLogger(name)


def get_git_hash():
    return 'unknown'


def collect_env():
    return {}


_BatchNorm = None


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    import warnings as _w
    _w.warn(f'mmcv.utils.{name} is not implemented (shim stub)')
    return _stub


class _stub:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return None
    def __getattr__(self, name): return self
