import os
from yacs.config import CfgNode as CN


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = {
    'bninception': os.path.join(_PROJECT_ROOT, "resource/models/bn_inception-52deb4733.pth"),
    'resnet50': os.path.join(_PROJECT_ROOT, "resource/models/resnet50-19c8e357.pth"),
    'resnet18': os.path.join(_PROJECT_ROOT, "resource/models/resnet18-f37072fd.pth"),
    'googlenet': os.path.join(_PROJECT_ROOT, "resource/models/googlenet-1378be20.pth"),
    'mobilenet_v3_small': os.path.join(_PROJECT_ROOT, "resource/models/mobilenet_v3_small.pth"),
}

MODEL_PATH = CN(MODEL_PATH)