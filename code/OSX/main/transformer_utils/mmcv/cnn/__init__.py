import math
import torch
import torch.nn as nn
from ..utils import Registry


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    if distribution == 'normal':
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    if distribution == 'normal':
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module, mean=0, std=1, bias=0, a=-2, b=2):
    import warnings
    try:
        from torch.nn.init import trunc_normal_
        trunc_normal_(module.weight, std=std, a=a, b=b)
    except (ImportError, AttributeError):
        warnings.warn('trunc_normal_ not available, using normal_')
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    return float(-math.log((1 - prior_prob) / prior_prob))


def is_norm(module):
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d,
                               nn.InstanceNorm2d, nn.InstanceNorm3d))


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias='auto', conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU'), inplace=True, order=('conv', 'norm', 'act')):
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.order = order
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.conv = conv
        if self.with_norm:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = nn.Identity()
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)
        else:
            self.activate = nn.Identity()

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias='auto', norm_cfg=None, act_cfg=dict(type='ReLU'),
                 dw_norm_cfg=None, pw_norm_cfg=None, inplace=True, order=('dwconv', 'dwact', 'pwconv', 'pwact')):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                         padding, dilation, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if norm_cfg:
            self.dw_norm = build_norm_layer(dw_norm_cfg or norm_cfg, in_channels)[1]
            self.pw_norm = build_norm_layer(pw_norm_cfg or norm_cfg, out_channels)[1]
            self.with_norm = True
        else:
            self.dw_norm = nn.Identity()
            self.pw_norm = nn.Identity()
            self.with_norm = False
        self.dw_act = build_activation_layer(act_cfg)
        self.pw_act = build_activation_layer(act_cfg) if act_cfg else nn.Identity()

    def forward(self, x):
        for layer_name in self.order:
            if layer_name == 'dwconv':
                x = self.depthwise_conv(x)
            elif layer_name == 'dwact':
                x = self.dw_act(x)
            elif layer_name == 'pwconv':
                x = self.pointwise_conv(x)
            elif layer_name == 'pwact':
                x = self.pw_act(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.linear(x)


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


_norm_cfgs = {
    'BN': lambda n: nn.BatchNorm2d(n),
    'BN1d': lambda n: nn.BatchNorm1d(n),
    'BN2d': lambda n: nn.BatchNorm2d(n),
    'BN3d': lambda n: nn.BatchNorm3d(n),
    'GN': lambda n: nn.GroupNorm(num_groups=32, num_channels=n),
    'LN': lambda n: nn.LayerNorm(n),
    'IN': lambda n: nn.InstanceNorm2d(n),
    'SyncBN': lambda n: nn.BatchNorm2d(n),
}


def build_norm_layer(cfg, num_features, postfix=''):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    norm_type = cfg.pop('type', 'BN')
    norm_class = _norm_cfgs.get(norm_type, _norm_cfgs['BN'])
    layer = norm_class(num_features)
    name = 'bn' + str(postfix) if postfix else 'bn'
    cfg.update({'type': norm_type})
    return name, layer


_act_cfgs = {
    'ReLU': lambda: nn.ReLU(inplace=True),
    'LeakyReLU': lambda: nn.LeakyReLU(inplace=True),
    'PReLU': lambda: nn.PReLU(),
    'ELU': lambda: nn.ELU(),
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'Softmax': nn.Softmax,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'HSigmoid': nn.Hardsigmoid,
    'HSwish': nn.Hardswish,
}


def build_activation_layer(cfg):
    if cfg is None:
        return nn.Identity()
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    cfg = cfg.copy()
    act_type = cfg.pop('type')
    act_class = _act_cfgs.get(act_type)
    if act_class is None:
        return nn.ReLU(inplace=True)
    return act_class(**cfg)


_upsample_cfgs = {
    'deconv': nn.ConvTranspose2d,
    'deconv_t': nn.ConvTranspose2d,
    'nearest': lambda **kwargs: nn.Sequential(nn.Upsample(scale_factor=kwargs.pop('scale_factor', 2), mode='nearest'), nn.Conv2d(**kwargs)),
    'bilinear': lambda **kwargs: nn.Sequential(nn.Upsample(scale_factor=kwargs.pop('scale_factor', 2), mode='bilinear', align_corners=False), nn.Conv2d(**kwargs)),
}


def build_upsample_layer(cfg, *args, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    cfg = cfg.copy() if cfg else {}
    up_type = cfg.pop('type', 'deconv')
    factory = _upsample_cfgs.get(up_type, _upsample_cfgs['deconv'])
    return factory(*args, **cfg, **kwargs)


_conv_cfgs = {
    'Conv2d': nn.Conv2d,
    'Conv': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv1d': nn.Conv1d,
}


def build_conv_layer(cfg, *args, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    cfg = cfg.copy() if cfg else {}
    conv_type = cfg.pop('type', 'Conv2d')
    factory = _conv_cfgs.get(conv_type, _conv_cfgs['Conv2d'])
    return factory(*args, **cfg, **kwargs)


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    import warnings as _w
    _w.warn(f'mmcv.cnn.{name} is not implemented (shim stub)')
    return _Stub


class _Stub:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return None
    def __getattr__(self, name): return self


MODELS = Registry('models')


def build_model_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        return cfg
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    else:
        obj_cls = obj_type
    pass
    if default_args:
        for key, val in default_args.items():
            args.setdefault(key, val)
    return obj_cls(**args)
