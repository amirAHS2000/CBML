
from torch.utils.data import DataLoader

from .collate_batch import collate_fn
from .datasets import BaseDataSet
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def build_data(cfg, is_train=True, is_eval=False):
    transforms = build_transforms(cfg, is_train=is_train and not is_eval)
    if is_train and not is_eval:
        dataset = BaseDataSet(cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        sampler = RandomIdentitySampler(dataset=dataset,
                                       batch_size=cfg.DATA.TRAIN_BATCHSIZE,
                                       num_instances=cfg.DATA.NUM_INSTANCES,
                                       max_iters=cfg.SOLVER.MAX_ITERS)
        data_loader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=sampler,
                                num_workers=cfg.DATA.NUM_WORKERS,
                                pin_memory=True)
    elif is_eval:  # Evaluation mode for train set
        dataset = BaseDataSet(cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        data_loader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                shuffle=False,
                                batch_size=cfg.DATA.TEST_BATCHSIZE,  # Use test batch size for consistency
                                num_workers=cfg.DATA.NUM_WORKERS)
    else:  # Validation/test
        dataset = BaseDataSet(cfg.DATA.TEST_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        data_loader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                shuffle=False,
                                batch_size=cfg.DATA.TEST_BATCHSIZE,
                                num_workers=cfg.DATA.NUM_WORKERS)
    return data_loader