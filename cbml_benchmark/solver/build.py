import torch
from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model, criterion=None, loss_param=None):
    """
    Returns:
        optimizer_main: Adam for model + most loss params
        optimizer_weights: SGD for MP-CBML weights, or None otherwise
    """
    params = []
    
    # Add model parameters with lr multiplier
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr_mul = 0.1 if "backbone" in key else 1.0
        params.append({"params": [value], "lr_mul": lr_mul})
    
    is_mpcbml = (cfg.LOSSES.NAME == 'mpcbml_loss')
    
    if is_mpcbml and criterion is not None:
        # MP-CBML: add all loss params except weights to Adam
        for name, param in criterion.named_parameters():
            if not param.requires_grad:
                continue
            if name == 'weights':
                continue
            params.append({"params": [param], "lr_mul": 1.0})
    
    elif loss_param is not None:
        # Other losses: add all loss parameters to Adam
        for p in loss_param.parameters():
            params.append({"params": [p], "lr_mul": 1.0})
    
    # Build optimizer - pass lr directly, let PyTorch handle lr_mul
    optimizer_main = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    
    # Build separate SGD optimizer for weights (MP-CBML only)
    optimizer_weights = None
    if is_mpcbml and criterion is not None and hasattr(criterion, 'weights'):
        optimizer_weights = torch.optim.SGD(
            [{"params": [criterion.weights]}],
            lr=getattr(cfg.SOLVER, 'WEIGHT_LR', 1e-4),
            momentum=getattr(cfg.SOLVER, 'WEIGHT_MOMENTUM', 0.0),
            weight_decay=0.0
        )
    
    return optimizer_main, optimizer_weights


def build_lr_scheduler(cfg, optimizer_main, optimizer_weights=None):
    scheduler_main = WarmupMultiStepLR(
        optimizer_main,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    
    scheduler_weights = None
    if optimizer_weights is not None:
        scheduler_weights = WarmupMultiStepLR(
            optimizer_weights,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    
    return scheduler_main, scheduler_weights

