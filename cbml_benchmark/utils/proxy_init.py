import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cbml_benchmark.data.datasets import BaseDataSet
from cbml_benchmark.data.collate_batch import collate_fn
from cbml_benchmark.data.transforms import build_transforms


def configure_proxy_classes(cfg, dataset):
    """Infer C and reject noncontiguous labels (the CUB split is contiguous)."""
    labels = sorted({int(y) for y in dataset.label_list})
    if len(labels) < 2 or labels != list(range(len(labels))):
        raise ValueError("Training labels must be contiguous 0..C-1 with C >= 2.")
    opt = (cfg.LOSSES.GMM_PROXY_LOSS if cfg.LOSSES.NAME == "gmm_proxy_loss"
           else cfg.LOSSES.PROXY_CBML_LOSS)
    if opt.NUM_CLASSES not in (0, len(labels)):
        raise ValueError("Configured NUM_CLASSES disagrees with the dataset.")
    opt.NUM_CLASSES = len(labels)
    return labels


@torch.no_grad()
def spherical_kmeans(x, k, iterations=25):
    """Class-local cosine clustering; empty clusters reseeded within class."""
    if x.shape[0] < k:
        raise ValueError("A training class has fewer samples than K.")
    x = F.normalize(x, dim=-1)
    centers = x[torch.randperm(len(x), device=x.device)[:k]].clone()
    for _ in range(iterations):
        assignments = (x @ centers.t()).argmax(dim=1)
        updated = centers.clone()
        for j in range(k):
            members = x[assignments == j]
            if len(members):
                mean = members.mean(dim=0)
                if mean.norm() > 1e-12:
                    updated[j] = F.normalize(mean, dim=0)
                    continue
            updated[j] = x[torch.randint(len(x), (1,), device=x.device).item()]
        if torch.allclose(updated, centers, atol=1e-5, rtol=0):
            centers = updated
            break
        centers = updated
    return centers


@torch.no_grad()
def initialize_proxies(cfg, model, criterion, device, diagnostics=None):
    """Use training images only; one deterministic-transform pass, no sampler."""
    method = (cfg.LOSSES.GMM_PROXY_LOSS.INIT_METHOD if cfg.LOSSES.NAME == "gmm_proxy_loss"
              else cfg.LOSSES.PROXY_CBML_LOSS.INIT_METHOD)
    if method == "random":
        if diagnostics:
            diagnostics.event("proxy_init", method=method, total_s=0.0)
        return
    if method != "kmeans":
        raise ValueError("INIT_METHOD must be random or kmeans.")
    start = time.perf_counter()
    dataset = BaseDataSet(cfg.DATA.TRAIN_IMG_SOURCE,
                          transforms=build_transforms(cfg, is_train=False),
                          mode=cfg.INPUT.MODE)
    loader = DataLoader(dataset, batch_size=cfg.DATA.TEST_BATCHSIZE,
                        shuffle=False, num_workers=cfg.DATA.NUM_WORKERS,
                        collate_fn=collate_fn)
    setup_s = time.perf_counter() - start
    was_training = model.training
    model.eval()
    features, labels = [], []
    try:
        extract_start = time.perf_counter()
        for images, targets in loader:
            features.append(model(images.to(device)).cpu())
            labels.append(targets.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
        extract_s = time.perf_counter() - extract_start
        cluster_start = time.perf_counter()
        for c in range(criterion.num_classes):
            centers = spherical_kmeans(features[labels == c], criterion.k)
            criterion.proxies[c].copy_(centers.to(device))
        if diagnostics:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            diagnostics.event("proxy_init", method=method, n_images=len(dataset),
                              loader_setup_s=setup_s, extract_s=extract_s,
                              cluster_and_copy_s=time.perf_counter()-cluster_start,
                              total_s=time.perf_counter()-start)
    finally:
        model.train(was_training)
