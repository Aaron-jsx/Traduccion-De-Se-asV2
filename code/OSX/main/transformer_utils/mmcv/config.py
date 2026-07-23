import os.path as osp
import sys
import tempfile
import re
import copy


class Config:
    def __init__(self, cfg_dict):
        self._cfg_dict = cfg_dict

    def __getattr__(self, name):
        try:
            return self._cfg_dict[name]
        except KeyError:
            raise AttributeError(f'Config has no attribute "{name}"')

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __contains__(self, name):
        return name in self._cfg_dict

    @classmethod
    def fromfile(cls, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace {{_base_.xxx}} with "{{_base_.xxx}}" to make it valid Python
        content = re.sub(r'\{\{\s*(_base_\.[^}]+?)\s*\}\}', r'"{{\1}}"', content)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', encoding='utf-8', delete=False) as f:
            f.write(content)
            temp_filename = f.name
            
        try:
            import importlib.util as iu
            spec = iu.spec_from_file_location('config_module', temp_filename)
            mod = iu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cfg_dict = {k: v for k, v in mod.__dict__.items() if not k.startswith('_') or k == '_base_'}
        finally:
            import os
            os.remove(temp_filename)

        base_dict = {}
        if '_base_' in cfg_dict:
            bases = cfg_dict.pop('_base_')
            if isinstance(bases, str):
                bases = [bases]
            for base in bases:
                base_path = osp.join(osp.dirname(filename), base)
                base_cfg = Config.fromfile(base_path)
                cls._merge_dicts(base_dict, base_cfg._cfg_dict)

        if base_dict:
            cls._substitute_vars(cfg_dict, base_dict)
            cls._merge_dicts(base_dict, cfg_dict)
            cfg_dict = base_dict

        return cls(cfg_dict)

    @classmethod
    def _merge_dicts(cls, base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                cls._merge_dicts(base[k], v)
            else:
                base[k] = v
        return base

    @classmethod
    def _substitute_vars(cls, cfg_dict, base_dict):
        if isinstance(cfg_dict, dict):
            for k, v in cfg_dict.items():
                if isinstance(v, str) and v.startswith('{{_base_.') and v.endswith('}}'):
                    keys = v[9:-2].split('.')
                    val = base_dict
                    for key in keys:
                        val = val[key]
                    cfg_dict[k] = copy.deepcopy(val)
                else:
                    cls._substitute_vars(v, base_dict)
        elif isinstance(cfg_dict, list):
            for i in range(len(cfg_dict)):
                v = cfg_dict[i]
                if isinstance(v, str) and v.startswith('{{_base_.') and v.endswith('}}'):
                    keys = v[9:-2].split('.')
                    val = base_dict
                    for key in keys:
                        val = val[key]
                    cfg_dict[i] = copy.deepcopy(val)
                else:
                    cls._substitute_vars(v, base_dict)
