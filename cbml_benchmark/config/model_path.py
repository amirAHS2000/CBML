
from yacs.config import CfgNode as CN

MODEL_PATH = {
    'bninception': "~/.cache/torch/checkpoints/bn_inception-52deb4733.pth",
    'resnet50': "~/.cache/torch/checkpoints/resnet50-19c8e357.pth",
    'googlenet': "~/.cache/torch/checkpoints/googlenet-1378be20.pth",
    'resnet18': "~/.cache/torch/checkpoints/resnet18-f37072fd.pth"
}

MODEL_PATH = CN(MODEL_PATH)
