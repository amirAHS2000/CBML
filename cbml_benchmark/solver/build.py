import torch

from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model, loss_param=None):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        mul = 0.1 if "backbone" in key else 1.0
        params.append({"params": [value], "lr": cfg.SOLVER.BASE_LR * mul})
    if loss_param is not None:
        params.append({"params": list(loss_param.parameters()),
                    "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.PROTO_LR_MUL,
                    "weight_decay": 0.0})
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params,
                                                                lr=cfg.SOLVER.BASE_LR,
                                                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
