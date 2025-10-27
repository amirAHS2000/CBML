import os
import re
import gc

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

def initialize_prototypes_mean(model, cfg):
    """
    compute the mean feature for each class from a subset of the training data.
    If multiple prototypes are desired per class, add small noise around the mean.
    """
    model.eval()

    # build transforms using training configs
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD    
    )
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

    img_path_cls = {cls: [] for cls in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES)}
    BASE_DIR = os.path.dirname(cfg.DATA.TRAIN_IMG_SOURCE)

    prototypes = torch.zeros(
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES,
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
        cfg.MODEL.HEAD.DIM,
        device=cfg.MODEL.DEVICE
    )

    with open(cfg.DATA.TRAIN_IMG_SOURCE, 'r') as f:
        for line in f:
            try:
                path, label = re.split(r",| ", line.strip())
                actual_path = os.path.join(BASE_DIR, path)
                img_path_cls[int(label)].append(actual_path)
            except Exception as e:
                print(f"Error loading image {path}: {e}")

    for cls in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES):
        if not img_path_cls[cls]:
            # random initialization if no images for this class
            prototypes[cls] = torch.randn(
                cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
                cfg.MODEL.HEAD.DIM,
                device=cfg.MODEL.DEVICE
            )
            continue

        imgs = []

        for img_path in img_path_cls[cls]:
            img = read_image(img_path, mode=cfg.INPUT.MODE)
            transformed_img = transforms(img)
            imgs.append(transformed_img)

        # stack all images for current class
        images = torch.stack(imgs).to(cfg.MODEL.DEVICE)

        # extract features
        with torch.no_grad():
            feats = model(images)
            feats_np = feats.cpu().numpy()
            # clear the features tensor
            del feats
            torch.cuda.empty_cache()

        # clear the stacked images
        del images
        torch.cuda.empty_cache()

        class_mean_np = feats_np.mean(axis=0)

        class_mean = torch.tensor(class_mean_np, dtype=torch.float, device=cfg.MODEL.DEVICE)
        
        for k in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS):
            noise = torch.randn(cfg.MODEL.HEAD.DIM, device=cfg.MODEL.DEVICE) * 0.01 # noise
            prototypes[cls, k] = class_mean + noise

        del class_mean_np
        del class_mean

        del feats_np
        gc.collect()

    return prototypes

def initialize_prototypes_kmeans(model, cfg):
    
    model.eval()

    # build transforms using training configs
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                                      std=cfg.INPUT.PIXEL_STD)
    transforms = T.Compose([
        T.Resize(size=cfg.INPUT.CROP_SIZE), # TODO: should it CROP_SIZE or ORIGIN_SIZE
        T.ToTensor(),
        normalize_transform,
    ])

    img_path_cls = {cls: [] for cls in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES)}
    BASE_DIR = os.path.dirname(cfg.DATA.TRAIN_IMG_SOURCE)

    prototypes = torch.zeros(
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES,
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
        cfg.MODEL.HEAD.DIM,
        device=cfg.MODEL.DEVICE
    )

    cluster_sizes = torch.zeros(
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES,
        cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
        device=cfg.MODEL.DEVICE
    )

    with open(cfg.DATA.TRAIN_IMG_SOURCE, 'r') as f:
        for line in f:
            try:
                path, label = re.split(r",| ", line.strip())
                actual_path = os.path.join(BASE_DIR, path)
                img_path_cls[int(label)].append(actual_path)
            except Exception as e:
                print(f"Error loading image {path}: {e}")

    for cls in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.N_CLASSES):
        if not img_path_cls[cls]:
            # Random initialization if no images for this class
            prototypes[cls] = torch.randn(
                cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
                cfg.MODEL.HEAD.DIM,
                device=cfg.MODEL.DEVICE
            )
            continue
        
        imgs = []
        for img_path in img_path_cls[cls]:
            img = read_image(img_path, mode=cfg.INPUT.MODE)
            transformed_img = transforms(img)
            imgs.append(transformed_img)
        
        images = torch.stack(imgs).to(cfg.MODEL.DEVICE)

        with torch.no_grad():
            feats = model(images)
            feats_np = feats.cpu().numpy()
            del feats
            torch.cuda.empty_cache()
        
        del images
        torch.cuda.empty_cache()

        if len(feats_np) >= cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS:
            kmeans = KMeans(
                n_clusters=cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS,
                random_state=0,
                n_init=10
            ).fit(feats_np)
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
            prototypes[cls] = centers.to(cfg.MODEL.DEVICE)
            cluster_sizes[cls] = torch.tensor(np.bincount(kmeans.labels_, minlength=cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS), device=cfg.MODEL.DEVICE).float()
            del kmeans
            del centers
            torch.cuda.empty_cache()
        else:
            mean_feat = torch.tensor(feats_np.mean(axis=0), dtype=torch.float)
            for k in range(cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS):
                noise = torch.randn(cfg.MODEL.HEAD.DIM, device=cfg.MODEL.DEVICE) * 0.01
                prototypes[cls, k] = mean_feat.to(cfg.MODEL.DEVICE) + noise
                del noise
                torch.cuda.empty_cache()
            cluster_sizes[cls] = torch.ones(cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS, device=cfg.MODEL.DEVICE) / cfg.LOSSES.MULTI_PROTOTYPE_CBML.PROTOTYPE_PER_CLASS
            del mean_feat
            torch.cuda.empty_cache()

        del feats_np
        gc.collect()

    gc.collect()
    torch.cuda.empty_cache()

    return prototypes, cluster_sizes