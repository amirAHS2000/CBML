"""Offline smoke test: real ResNet-18 backbone, 64-D head and proxy loss.

The pretrained-weight download is replaced with a local synthetic state dict;
this tests wiring and dimensions, not the quality of ImageNet weights.
"""
from types import SimpleNamespace as NS

import torch
import torchvision.models as models

from cbml_benchmark.modeling import build_model
from cbml_benchmark.losses import build_loss


def test_resnet18_64_model_and_proxy(tmp_path, monkeypatch):
    original = models.resnet18
    weight_file = tmp_path / 'resnet18.pth'
    torch.save(original(weights=None).state_dict(), weight_file)
    monkeypatch.setattr(models, 'resnet18', lambda **kwargs: original(weights=None))
    cfg = NS(
        MODEL=NS(BACKBONE=NS(NAME='resnet18'), HEAD=NS(NAME='linear_norm', DIM=64),
                 PRETRAIN='imagenet', PRETRIANED_PATH={'resnet18': str(weight_file)}),
        LOSSES=NS(NAME='proxy_cbml_loss', PROXY_CBML_LOSS=NS(
            NUM_CLASSES=100, K=3, POS_A=.5, POS_B=.1, NEG_A=.2, NEG_B=.1,
            DETACH_NEGATIVE=True, PROXY_LR=.01, INIT_METHOD='kmeans')),
    )
    model = build_model(cfg).eval()
    criterion = build_loss(cfg)
    images = torch.randn(2, 3, 227, 227)
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        backbone = model.backbone(images)
        embeddings = model.headembedding(backbone)
        loss = criterion(embeddings, labels)
    assert backbone.shape == (2, 512)
    assert embeddings.shape == (2, 64)
    assert criterion.proxies.shape == (100, 3, 64)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(loss)
