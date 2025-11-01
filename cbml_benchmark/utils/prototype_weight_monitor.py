import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_weight_stats(weights: torch.Tensor):
    """
    Compute statistics over class-prototype weights.
    Args:
        weights: [C, K] raw weight tensor (before softmax)
    Returns:
        dict with entropy, variance, and dominance statistics
    """

    w = F.softmax(weights, dim=1)
    entropy = (-w * (w.clamp_min(1e-9)).log()).sum(dim=1)
    mean_entropy = entropy.mean().item()
    var_entropy = entropy.var(unbiased=False).item()
    weight_var = w.var(dim=1).mean().item()
    max_weight_mean = w.max(dim=1)[0].mean().item()

    return {
        "mean_entropy": round(mean_entropy, 5),
        "var_entropy": round(var_entropy, 5),
        "mean_weight_var": round(weight_var, 5),
        "mean_max_weight": round(max_weight_mean, 5),
    }

@torch.no_grad()
def compute_proto_stats(prototypes: torch.Tensor, initial_prototypes: torch.Tensor = None):
    """
    Compute intra/inter distances, norms, and displacement of prototypes.
    Args:
        prototypes: [C, K, D] tensor of current prototypes
        initial_prototypes: [C, K, D] optional tensor for displacement measurement
    Returns:
        dict with intra, inter, norm, and (optional) displacement stats
    """
    protos = F.normalize(prototypes, p=2, dim=-1)
    C, K, D = protos.shape

    # --- Intra-class distances ---
    intra = torch.stack([
        torch.pdist(protos[c], p=2).mean()
        if K > 1 else torch.tensor(0., device=protos.device)
        for c in range(C)
    ])
    mean_intra = intra.mean().item()

    # --- Inter-class distances ---
    flat = protos.view(C * K, D)
    dist = torch.cdist(flat, flat, p=2)
    mask = torch.ones_like(dist, dtype=torch.bool)
    for c in range(C):
        mask[c*K:(c+1)*K, c*K:(c+1)*K] = False
    inter = dist[mask].mean().item()

    # --- Norms ---
    norms = protos.norm(dim=-1).mean().item()

    stats = {
        "mean_intra_dist": round(mean_intra, 5),
        "mean_inter_dist": round(inter, 5),
        "mean_proto_norm": round(norms, 5)
    }

    if initial_prototypes is not None:
        disp = torch.norm(protos - F.normalize(initial_prototypes, p=2, dim=-1), dim=-1).mean().item()
        stats["mean_displacement"] = round(disp, 5)

    return stats