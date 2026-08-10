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
    'mobilenet_v2': 1280,
    'mobilenet_v3_small': 576,
}


def build_head(cfg):
    assert cfg.MODEL.HEAD.NAME in HEADS, f"head {cfg.MODEL.HEAD.NAME} is not defined"
    assert cfg.MODEL.BACKBONE.NAME in BACKBONE_OUT_CHANNELS, \
        f"unknown output channels for backbone {cfg.MODEL.BACKBONE.NAME}"
    in_channels = BACKBONE_OUT_CHANNELS[cfg.MODEL.BACKBONE.NAME]
    return HEADS[cfg.MODEL.HEAD.NAME](cfg, in_channels=in_channels)