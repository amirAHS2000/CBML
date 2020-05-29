from dbml_benchmark.modeling.registry import BACKBONES

from .bninception import BNInception
from .resnet import ResNet50
from .googlenet import GoogLeNet


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME in BACKBONES, \
        f"backbone {cfg.MODEL.BACKBONE} is not registered in registry : {BACKBONES.keys()}"
    return BACKBONES[cfg.MODEL.BACKBONE.NAME]()
