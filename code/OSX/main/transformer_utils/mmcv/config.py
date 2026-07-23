import os.path as osp
import sys
import tempfile


class Config:
    def __init__(self, cfg_dict):
        self._cfg_dict = cfg_dict

    def __getattr__(self, name):
        try:
            return self._cfg_dict[name]
        except KeyError:
            raise AttributeError(f'Config has no attribute "{name}"')

    @classmethod
    def fromfile(cls, filename):
        import importlib.util as iu
        spec = iu.spec_from_file_location('config', filename)
        mod = iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg_dict = {k: v for k, v in mod.__dict__.items() if not k.startswith('_')}
        return cls(cfg_dict)
