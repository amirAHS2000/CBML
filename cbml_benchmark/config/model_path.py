
from yacs.config import CfgNode as CN

MODEL_PATH = {
    'bninception': "resource/models/bn_inception-52deb4733.pth",
    'resnet50': "resource/models/resnet50-19c8e357.pth",
    'resnet18': "resource/models/resnet18-f37072fd.pth",
    'googlenet': "resource/models/googlenet-1378be20.pth",
    'mobilenet_v3_small': "resource/models/mobilenet_v3_small.pth",
}

MODEL_PATH = CN(MODEL_PATH)