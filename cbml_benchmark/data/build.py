from torch.utils.data import DataLoader, Subset
import numpy as np

from cbml_benchmark.utils.reproducibility import seed_worker

from .collate_batch import collate_fn
from .datasets import BaseDataSet
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def _stratified_subsample_indices(label_index_dict, target_size, seed=0, min_per_class=2):
    """
    Selects a class-balanced subset of indices, guaranteeing at least
    `min_per_class` samples per retained class (needed so recall@k's
    self-retrieval threshold, which relies on a second positive match,
    stays valid). Classes with fewer than `min_per_class` images total
    are skipped entirely.
    """
    rng = np.random.RandomState(seed)
    labels = list(label_index_dict.keys())

    usable_labels = [l for l in labels if len(label_index_dict[l]) >= min_per_class]
    skipped = len(labels) - len(usable_labels)

    per_class = max(min_per_class, target_size // max(len(usable_labels), 1))

    selected = []
    for label in usable_labels:
        idxs = label_index_dict[label]
        n_take = min(len(idxs), per_class)
        chosen = rng.choice(idxs, size=n_take, replace=False)
        selected.extend(int(i) for i in chosen)

    rng.shuffle(selected)
    return selected, skipped

def build_data(cfg, is_train=True, is_eval=False):
    transforms = build_transforms(cfg, is_train=is_train and not is_eval)
    if is_train and not is_eval:
        dataset = BaseDataSet(cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        sampler = RandomIdentitySampler(dataset=dataset,
                                       batch_size=cfg.DATA.TRAIN_BATCHSIZE,
                                       num_instances=cfg.DATA.NUM_INSTANCES,
                                       max_iters=cfg.SOLVER.MAX_ITERS,
                                       seed=cfg.SOLVER.RNG_SEED)
        data_loader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=sampler,
                                num_workers=cfg.DATA.NUM_WORKERS,
                                worker_init_fn=seed_worker,
                                pin_memory=True)
    elif is_eval:  # Evaluation mode for train set
        dataset = BaseDataSet(cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)

        subsample_size = getattr(cfg.DATA, "EVAL_TRAIN_SUBSAMPLE_SIZE", None)
        if subsample_size is not None and subsample_size < len(dataset):
            seed = getattr(cfg.DATA, "EVAL_TRAIN_SUBSAMPLE_SEED", 0)
            indices, skipped = _stratified_subsample_indices(
                dataset.label_index_dict, subsample_size, seed=seed
            )
            full_label_list = dataset.label_list
            dataset = Subset(dataset, indices)
            # do_train.py reads eval_train_loader.dataset.label_list directly,
            # so attach the matching subset labels onto the Subset object.
            dataset.label_list = [full_label_list[i] for i in indices]
            print(f"[eval_train subsample] using {len(indices)} images "
                  f"({skipped} classes skipped for having <2 images)")

        data_loader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                shuffle=False,
                                batch_size=cfg.DATA.TEST_BATCHSIZE,
                                num_workers=cfg.DATA.NUM_WORKERS,
                                worker_init_fn=seed_worker)
    else:  # Validation/test
        dataset = BaseDataSet(cfg.DATA.TEST_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        data_loader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                shuffle=False,
                                batch_size=cfg.DATA.TEST_BATCHSIZE,
                                num_workers=cfg.DATA.NUM_WORKERS,
                                worker_init_fn=seed_worker)
    return data_loader