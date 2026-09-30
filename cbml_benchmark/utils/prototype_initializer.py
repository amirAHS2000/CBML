import os
import re
import gc
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

from cbml_benchmark.utils.img_reader import read_image


def build_eval_transform(cfg):
    """Same pipeline the model sees at eval time.
    Prefer passing the repo's own build_transforms(cfg, is_train=False) to
    initialize_prototypes_kmeans(transform=...), so nothing can drift out of sync."""
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    return T.Compose([
        T.Resize(size=cfg.INPUT.ORIGIN_SIZE),
        T.CenterCrop(cfg.INPUT.CROP_SIZE),
        T.ToTensor(),
        normalize_transform,
    ])

class _ImageList(Dataset):
    def __init__(self, items, transform, mode):
        self.items, self.transform, self.mode = items, transform, mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(read_image(path, mode=self.mode)), label

def _parse_list(cfg):
    """Returns [(abs_path, label)]. Fails loudly instead of silently dropping lines."""
    src = cfg.DATA.TRAIN_IMG_SOURCE
    base = os.path.dirname(src)
    n_cls = cfg.LOSSES.CBML_LOSS.N_CLASSES
    items = []
    with open(src, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(.*)[,\s]+(-?\d+)$", line)
            if m is None:
                raise ValueError(f"{src}:{ln}: cannot parse '{line}'")
            path, label = m.group(1), int(m.group(2))
            if not 0 <= label < n_cls:
                raise ValueError(
                    f"{src}:{ln}: label {label} outside [0, {n_cls - 1}]. "
                    f"Check N_CLASSES and whether labels are 0-based/contiguous.")
            items.append((os.path.join(base, path), label))
    if not items:
        raise ValueError(f"No images found in {src}")
    return items

def _snapshot_modes(model):
    return {m: m.training for m in model.modules()}

def _restore_modes(states):
    for m, t in states.items():
        m.training = t

@torch.no_grad()
def _extract(model, loader, device):
    feats, labels = [], []
    for x, y in loader:
        f = model(x.to(device, non_blocking=True))
        feats.append(F.normalize(f.float(), dim=1).cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


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

    img_path_cls = {cls: [] for cls in range(cfg.LOSSES.CBML_LOSS.N_CLASSES)}
    BASE_DIR = os.path.dirname(cfg.DATA.TRAIN_IMG_SOURCE)

    prototypes = torch.zeros(
        cfg.LOSSES.CBML_LOSS.N_CLASSES,
        cfg.LOSSES.CBML_LOSS.PROTOTYPE_PER_CLASS,
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

    for cls in range(cfg.LOSSES.CBML_LOSS.N_CLASSES):
        if not img_path_cls[cls]:
            # random initialization if no images for this class
            prototypes[cls] = torch.randn(
                cfg.LOSSES.CBML_LOSS.PROTOTYPE_PER_CLASS,
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
        
        for k in range(cfg.LOSSES.CBML_LOSS.PROTOTYPE_PER_CLASS):
            noise = torch.randn(cfg.MODEL.HEAD.DIM, device=cfg.MODEL.DEVICE) * 0.01 # noise
            prototypes[cls, k] = class_mean + noise

        del class_mean_np
        del class_mean

        del feats_np
        gc.collect()

    return prototypes

def initialize_prototypes_kmeans(model, cfg, transform=None, batch_size=128,
                                 num_workers=4, seed=0):
    """Returns (prototypes [C,K,D] unit-norm on cfg.MODEL.DEVICE, cluster_sizes [C,K])."""
    C = cfg.LOSSES.CBML_LOSS.N_CLASSES
    K = cfg.LOSSES.CBML_LOSS.PROTOTYPE_PER_CLASS
    D = cfg.MODEL.HEAD.DIM
    device = cfg.MODEL.DEVICE

    items = _parse_list(cfg)
    transform = transform if transform is not None else build_eval_transform(cfg)
    loader = DataLoader(_ImageList(items, transform, cfg.INPUT.MODE),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    states = _snapshot_modes(model)
    model.eval()
    try:
        feats, labels = _extract(model, loader, device)
    finally:
        _restore_modes(states)          # BN freezing etc. is preserved

    assert feats.shape == (len(items), D), f"feature shape {tuple(feats.shape)} != ({len(items)}, {D})"

    protos = torch.zeros(C, K, D)
    sizes = torch.zeros(C, K)
    g = torch.Generator().manual_seed(seed)
    n_small, n_empty = 0, 0

    for c in range(C):
        fc = feats[labels == c]
        n = fc.size(0)
        if n == 0:
            n_empty += 1
            centers = torch.randn(K, D, generator=g)
            sizes[c] = torch.ones(K) / K
        elif n >= K:
            km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(fc.numpy())
            centers = torch.from_numpy(km.cluster_centers_).float()
            sizes[c] = torch.from_numpy(np.bincount(km.labels_, minlength=K)).float()
        else:
            n_small += 1
            centers = fc.mean(0, keepdim=True) + 0.01 * torch.randn(K, D, generator=g)
            sizes[c] = torch.ones(K) / K
        protos[c] = F.normalize(centers, dim=1)   # cosine loss -> unit-norm start

    if n_empty:
        warnings.warn(f"{n_empty} classes have no images (random prototypes). Check N_CLASSES / label file.")
    print(f"[proto-init] {len(items)} images, {C - n_empty - n_small} classes clustered, "
          f"{n_small} classes with < K={K} images (mean+noise), {n_empty} empty")
    return protos.to(device), sizes.to(device)


def reinitialize_prototypes(model, criterion, optimizer, cfg, **kw):
    """Call once after the warm-up. Re-runs k-means with the current head,
    resets Adam moments of the prototypes and the loss module's usage counters."""
    protos, sizes = initialize_prototypes_kmeans(model, cfg, **kw)
    criterion.set_prototypes(protos)
    for p in criterion.parameters():
        optimizer.state.pop(p, None)          # fresh Adam state for the prototypes
    criterion.pos_proto_counts.zero_()
    criterion.neg_proto_counts.zero_()
    return sizes