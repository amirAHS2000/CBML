
from yacs.config import CfgNode as CN

MODEL_PATH = {
    'bninception': "~/.cache/torch/checkpoints/bn_inception-52deb4733.pth",
    'resnet18': "/content/CBML/resnet18-f37072fd.pth",
    'googlenet': "~/.cache/torch/checkpoints/googlenet-1378be20.pth"
}

MODEL_PATH = CN(MODEL_PATH)
