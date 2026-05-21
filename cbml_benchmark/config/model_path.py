
from yacs.config import CfgNode as CN

MODEL_PATH = {
    'bninception': "~/.cache/torch/hub/checkpoints/bn_inception-52deb4733.pth",
    'resnet50': '~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth',
    'resnet18': '~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth',
    'googlenet': "~/.cache/torch/hub/checkpoints/googlenet-1378be20.pth"
}

MODEL_PATH = CN(MODEL_PATH)