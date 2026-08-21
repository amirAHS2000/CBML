import os
import hashlib
import json
import torch


def _get_prototype_cache_path(cfg):
    """
    Build a unique cache file path based on everything that affects the
    computed prototypes: backbone, pretrain source, dataset, and the
    init hyperparameters. This way different configs never collide.
    """

    cache_dir = getattr(cfg.LOSSES.MPCBML_LOSS, "PROTOTYPE_CACHE_DIR", "resource/prototype_cache")
    os.makedirs(cache_dir, exist_ok=True)

    key_dict = {
        "backbone": cfg.MODEL.BACKBONE.NAME,
        "pretrain": cfg.MODEL.PRETRAIN,
        "train_source": cfg.DATA.TRAIN_IMG_SOURCE,
        "n_classes": cfg.LOSSES.MPCBML_LOSS.N_CLASSES,
        "proto_per_class": cfg.LOSSES.MPCBML_LOSS.PROTOTYPE_PER_CLASS,
        "init_method": cfg.LOSSES.MPCBML_LOSS.INIT_METHOD,
        "head_dim": cfg.MODEL.HEAD.DIM,
        "crop_size": cfg.INPUT.CROP_SIZE,
        "rng_seed": cfg.SOLVER.RNG_SEED,
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:10]

    fname = f"prototypes_{cfg.MODEL.BACKBONE.NAME}_{cfg.LOSSES.MPCBML_LOSS.INIT_METHOD}_{key_hash}.pt"
    return os.path.join(cache_dir, fname), key_dict

def load_cached_prototypes(cfg):
    path, key_dict = _get_prototype_cache_path(cfg)
    if os.path.exists(path):
        data = torch.load(path, map_location="cpu")
        # sanity check the cache actually matches this config
        if data.get("key") == key_dict:
            return data["prototypes"], data["cluster_sizes"], path
        else:
            print(f"Cache file {path} exists but key mismatch, recomputing.")
    return None, None, path

def save_prototype_cache(path, key_dict, prototypes, cluster_sizes):
    torch.save({
        "key": key_dict,
        "prototypes": prototypes.cpu(),
        "cluster_sizes": cluster_sizes.cpu() if cluster_sizes is not None else None,
    }, path)
