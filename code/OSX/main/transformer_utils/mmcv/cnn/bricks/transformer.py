import copy
import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.linear(x)


class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2,
                 act_cfg=dict(type='ReLU'), ffn_drop=0., dropout_layer=None,
                 add_identity=True, init_cfg=None, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(self.activate)
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(ffn_drop) if dropout_layer else nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class BaseTransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attentions = nn.ModuleList()
        self.ffns = nn.ModuleList()

    def forward(self, x, *args, **kwargs):
        return x


class TransformerLayerSequence(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x, *args, **kwargs):
        return x


def build_transformer_layer_sequence(cfg, default_args=None):
    return TransformerLayerSequence()


def build_dropout(cfg, default_args=None):
    if cfg is None:
        return nn.Identity()
    cfg = copy.deepcopy(cfg)
    dropout_type = cfg.pop('type', 'Dropout')
    if dropout_type == 'Dropout':
        return nn.Dropout(cfg.get('dropout_ratio', 0.1))
    elif dropout_type == 'DropPath':
        return nn.Identity()
    return nn.Identity()


def build_positional_encoding(cfg, default_args=None):
    if cfg is None:
        return nn.Identity()
    cfg = copy.deepcopy(cfg)
    encoding_type = cfg.pop('type')
    if encoding_type == 'SinePositionalEncoding':
        num_feats = cfg.get('num_feats', 128)
        return SinePositionalEncoding(num_feats)
    elif encoding_type == 'LearnedPositionalEncoding':
        num_feats = cfg.get('num_feats', 256)
        row_num_embed = cfg.get('row_num_embed', 50)
        col_num_embed = cfg.get('col_num_embed', 50)
        return LearnedPositionalEncoding(num_feats, row_num_embed, col_num_embed)
    return nn.Identity()


class SinePositionalEncoding(nn.Module):
    def __init__(self, num_feats, temperature=10000, normalize=False, scale=None, eps=1e-6):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi
        self.eps = eps

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_feats=256, row_num_embed=50, col_num_embed=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)

    def forward(self, mask):
        h, w = mask.shape[-2:]
        i = torch.arange(w)
        j = torch.arange(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1),
                         y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos
