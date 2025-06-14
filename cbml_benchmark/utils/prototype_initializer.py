import os
import re

import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import torchvision.transforms as T

from cbml_benchmark.utils.img_reader import read_image


def initialize_prototypes_random(num_classes, 
                                prototype_per_class, 
                                embed_dim, 
                                device):
    # random initialization
    prototypes = torch.randn(
        num_classes,
        prototype_per_class,
        embed_dim,
        device=device
    )
    return prototypes

def initialize_prototypes_mean(train_loader, 
                               model, 
                               num_classes, 
                               prototype_per_class, 
                               device):
    """
    compute the mean feature for each class from a subset of the training data.
    If multiple prototypes are desired per class, add small noise around the mean.
    """
    features_dict = {cls: [] for cls in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            feats = model(images) # backbone + head
            for feat, label in zip(feats, labels):
                features_dict[label.item()].append(feat.cpu())
    embed_dim = next(iter(features_dict.values()))[0].shape[0]
    prototypes = torch.zeros(num_classes, prototype_per_class, embed_dim)
    for cls in range(num_classes):
        if features_dict[cls]:
            class_feats = torch.stack(features_dict[cls])
            class_mean = class_feats.mean(dim=0)
            for k in range(prototype_per_class):
                noise = torch.randn(embed_dim) * 0.01 # noise
                prototypes[cls, k] = class_mean + noise
        else:
            prototypes[cls] = torch.zeros(prototype_per_class, embed_dim)
    return prototypes.to(device)

def initialize_prototypes_kmeans(model, cfg):
    
    model.eval()

    # build transforms using training configs
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                                      std=cfg.INPUT.PIXEL_STD)
    transforms = T.Compose([
        T.Resize(size=cfg.INPUT.ORIGIN_SIZE),
        T.RandomResizedCrop(
            scale=cfg.INPUT.CROP_SCALE,
            size=cfg.INPUT.CROP_SIZE
        ),
        T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
        T.ToTensor(),
        normalize_transform,
    ])

    # a dictionary contains each class data based on class label
    img_class_dict = {cls: [] for cls in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES)}
    
    with open(cfg.DATA.TRAIN_IMG_SOURCE, 'r') as f:
        for line in f:
            try:
                _path, _label = re.split(r",| ", line.strip())
                base_dir = os.path.dirname(cfg.DATA.TRAIN_IMG_SOURCE)
                actual_path = os.path.join(base_dir, _path)
                img = read_image(actual_path, mode=cfg.INPUT.MODE)
                img_class_dict[int(_label)].append(transforms(img))
                # clean up the loaded image immediately
                del img
            except Exception as e:
                print(f"Error loading image {_path}: {e}")

    prototypes = torch.zeros(
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES,
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
        cfg.MODEL.HEAD.DIM,
        device=cfg.MODEL.DEVICE
    )

    for cls in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES):
        if not img_class_dict[cls]:
            # random initialization if no images for this class
            prototypes[cls] = torch.randn(
                cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
                cfg.MODEL.HEAD.DIM,
                device=cfg.MODEL.DEVICE
            )
            continue
        
        # stack all images for current class
        images = torch.stack(img_class_dict[cls]).to(cfg.MODEL.DEVICE)
        # clear the loaded images for this class
        img_class_dict[cls] = []

        # extract features
        with torch.no_grad():
            feats = model(images)
            feats_np = feats.cpu().numpy()
            # clear the features tensor
            del feats
        
        # clear the stacked images
        del images

        if len(feats_np) >= cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS:
            # use KMeans if we have enough samples
            kmeans = KMeans(
                n_clusters=cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
                random_state=0,
                n_init=10 # multiple initialization for better results
            ).fit(feats_np)
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
            prototypes[cls] = centers
            # clear kmeans and centers
            del kmeans
            del centers
        else:
            # if we don't have enough samples, use mean with noise
            mean_feat = torch.tensor(feats_np.mean(axis=0), dtype=torch.float)
            for k in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS):
                noise = torch.randn(cfg.MODEL.HEAD.DIM, device=cfg.MODEL.DEVICE) * 0.01
                prototypes[cls, k] = mean_feat + noise
            
            # clear mean_feat
            del mean_feat

        # clear the numpy features array
        del feats_np

    # clear the image dictionary
    del img_class_dict

    return prototypes
