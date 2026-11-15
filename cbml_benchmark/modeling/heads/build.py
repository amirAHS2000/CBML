from cbml_benchmark.modeling.registry import HEADS

from .linear_norm import LinearNorm


BACKBONE_OUT_CHANNELS = {
    'bninception': 1024,
    'googlenet': 1024,
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
}


def build_head(cfg):
    head_name = cfg.MODEL.HEAD.NAME
    backbone_name = cfg.MODEL.BACKBONE.NAME
    assert head_name in HEADS, f'head {head_name} is not defined'
    if backbone_name not in BACKBONE_OUT_CHANNELS:
        raise ValueError(f'Unknown output dimension for backbone {backbone_name}')
    return HEADS[head_name](cfg, in_channels=BACKBONE_OUT_CHANNELS[backbone_name])
