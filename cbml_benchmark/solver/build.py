import torch
from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model, criterion=None, loss_param=None):
    """
    Returns:
        optimizer_main: Adam for model + most loss params
        optimizer_weights: SGD for MP-CBML weights, or None otherwise
    """
    params = []
    base_lr = getattr(cfg.SOLVER, 'BASE_LR', 0.00003)
    
    # Add model parameters with lr multiplier
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        if key.startswith('backbone.'):
            lr_mul = 0.3                     # backbone (slightly higher)
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
        elif key.startswith('headembedding.'):
            lr_mul = 8.0                      # head learns fast
            weight_decay = 0.0
        else:
            lr_mul = 1.0                      # fallback (rare)
            weight_decay = 0.0

        params.append({
            'params': [value],
            'lr': base_lr * lr_mul,
            'weight_decay': weight_decay
        })


    is_mpcbml = (cfg.LOSSES.NAME == 'mpcbml_loss')
   
    if is_mpcbml and criterion is not None:
        # MP-CBML: add all loss params except weights to Adam
        for name, param in criterion.named_parameters():
            if not param.requires_grad:
                continue
            if name == 'weights':
                continue
            if 'prototypes' in name:
                # Prototypes need to move fast to catch data clusters
                current_lr_mul = 30.0
            elif 'theta' in name:
                current_lr_mul = 1.0
            else:
                current_lr_mul = 1.0
            
            params.append({
                'params': [param],
                'lr': base_lr * current_lr_mul,
                'weight_decay': 0.0
            })

    elif loss_param is not None:
        # Other losses: add all loss parameters to Adam
        for p in loss_param.parameters():
            params.append({
                'params': [p],
                'lr': base_lr
            })
    
    # Build optimizer - pass lr directly, let PyTorch handle lr_mul
    optimizer_main = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        params
    )

    for i, g in enumerate(optimizer_main.param_groups):
        print(
            f"group {i}: lr={g['lr']}, weight_decay={g.get('weight_decay', 'default')}"
        )
    
    # Build separate SGD optimizer for weights (MP-CBML only)
    optimizer_weights = None
    if is_mpcbml and criterion is not None and hasattr(criterion, 'weights'):
        optimizer_weights = torch.optim.SGD(
            [{"params": [criterion.weights]}],
            lr=getattr(cfg.SOLVER, 'WEIGHT_LR', 0.00003),
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

