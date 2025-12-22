"""
Standalone utility for computing loss statistics.
Can be imported and used from any file without dependency on MpcbmlLoss class.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_statistics(
    embeddings,
    targets,
    z,
    sims,
    r_scores,
    r_pos,
    r_neg,
    s_pos,
    s_neg,
    beta,
    xi,
    loss_main,
    loss_reg,
    # Additional parameters needed from loss class
    gamma_reg,
    global_s_pos,
    global_s_neg,
    theta,
    prototypes,
    weights,
    device,
    initial_prototypes=None,
):
    """
    Compute comprehensive statistics for MPCBML loss monitoring.

    This is a refactored standalone version of MpcbmlLoss._compute_statistics()
    that can be called from any file.

    Args:
        embeddings: [B, D] - Original embeddings
        targets: [B] - Target class indices
        z: [B, D] - Normalized embeddings
        sims: [B, C, K] - Similarity scores
        r_scores: [B, C, K] - Log contribution scores
        r_pos: [B] - Best positive score
        r_neg: [B] - Best negative score
        s_pos: [B] - Positive similarity
        s_neg: [B] - Negative similarity
        beta: scalar - Temperature parameter
        xi: scalar - Regularization threshold
        loss_main: scalar - Main loss value
        loss_reg: scalar - Regularization loss value
        gamma_reg: float - Regularization weight for positive mean
        global_s_pos: scalar tensor - Global positive similarity EMA
        global_s_neg: scalar tensor - Global negative similarity EMA
        theta: scalar tensor - Temperature parameter (learnable)
        prototypes: [C, K, D] - Prototype tensors
        weights: [C, K] - Mixture weights
        device: torch.device - Device to use
        initial_prototypes: [C, K, D] - Initial prototypes (optional, for movement tracking)

    Returns:
        stats: dict - Dictionary containing all computed statistics
    """
    stats = {}
    B, C, K = sims.shape
    eps = 1e-9

    with torch.no_grad():
        # Loss components
        stats['loss_total'] = (loss_main + gamma_reg * loss_reg).item()
        stats['loss_main'] = loss_main.item()
        stats['loss_reg'] = loss_reg.item()

        # Core similarity metrics
        stats['s_pos_mean'] = s_pos.mean().item()
        stats['s_pos_std'] = s_pos.std().item()
        stats['s_neg_mean'] = s_neg.mean().item()
        stats['s_neg_std'] = s_neg.std().item()
        stats['similarity_margin_mean'] = (s_pos - s_neg).mean().item()
        stats['similarity_margin_min'] = (s_pos - s_neg).min().item()

        # Temperature
        stats['beta'] = beta.item()
        stats['theta'] = theta.item()

        # Regularization
        stats['xi_threshold'] = xi.item()
        stats['global_s_pos'] = global_s_pos.item()
        stats['global_s_neg'] = global_s_neg.item()
        stats['reg_violation_ratio'] = (s_neg > xi).float().mean().item()

        # Critical overfitting indicators
        stats['perfect_separation_ratio'] = (s_pos > 0.99).float().mean().item()
        stats['margin_violated_ratio'] = ((s_pos - s_neg) < 0).float().mean().item()

        # Embedding quality
        embed_norms = embeddings.norm(p=2, dim=1)
        stats['embed_norm_mean'] = embed_norms.mean().item()
        stats['embed_norm_std'] = embed_norms.std().item()

        # Embedding collapse detection
        embed_sim_matrix = torch.matmul(z, z.t())
        mask = ~torch.eye(B, device=z.device).bool()
        if mask.any():
            inter_sample_sims = embed_sim_matrix[mask]
            stats['embed_inter_sample_sim_mean'] = inter_sample_sims.mean().item()
            stats['embed_collapse_ratio'] = (inter_sample_sims > 0.9).float().mean().item()

        # Intra-class vs inter-class similarity
        same_class_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)) & mask
        diff_class_mask = (targets.unsqueeze(0) != targets.unsqueeze(1))

        if same_class_mask.any():
            intra_class_sims = embed_sim_matrix[same_class_mask]
            stats['embed_intra_class_sim_mean'] = intra_class_sims.mean().item()

        if diff_class_mask.any():
            inter_class_sims = embed_sim_matrix[diff_class_mask]
            stats['embed_inter_class_sim_mean'] = inter_class_sims.mean().item()

        if same_class_mask.any() and diff_class_mask.any():
            stats['embed_class_separation_margin'] = (
                stats['embed_intra_class_sim_mean'] -
                stats['embed_inter_class_sim_mean']
            )

        # Prototype statistics
        proto_norms = prototypes.norm(p=2, dim=2)
        stats['proto_norm_mean'] = proto_norms.mean().item()
        stats['proto_norm_std'] = proto_norms.std().item()

        if initial_prototypes is not None:
            proto_movement = (prototypes - initial_prototypes).norm(p=2, dim=2)
            stats['proto_movement_mean'] = proto_movement.mean().item()
            stats['proto_movement_max'] = proto_movement.max().item()

        # Intra-class prototype similarity
        intra_class_proto_sims = []
        for c in range(C):
            if K > 1:
                proto_c = prototypes[c]
                sim_matrix = torch.matmul(proto_c, proto_c.t())
                mask_k = ~torch.eye(K, device=proto_c.device).bool()
                intra_class_proto_sims.append(sim_matrix[mask_k].mean().item())

        if intra_class_proto_sims:
            stats['proto_intra_class_sim_mean'] = float(np.mean(intra_class_proto_sims))
            stats['proto_intra_class_collapse_ratio'] = float(
                np.mean([s > 0.95 for s in intra_class_proto_sims])
            )

        # Inter-class prototype separation
        flat_protos = prototypes.view(C * K, -1)
        proto_sim_matrix = torch.matmul(flat_protos, flat_protos.t())
        class_labels = torch.arange(C, device=device).repeat_interleave(K)
        inter_class_proto_mask = (class_labels.unsqueeze(0) != class_labels.unsqueeze(1))

        if inter_class_proto_mask.any():
            inter_class_proto_sims = proto_sim_matrix[inter_class_proto_mask]
            stats['proto_inter_class_sim_mean'] = inter_class_proto_sims.mean().item()
            stats['proto_inter_class_confusion_ratio'] = (
                (inter_class_proto_sims > 0.8).float().mean().item()
            )

        # Weight statistics
        weights_val = weights
        stats['weight_mean'] = weights_val.mean().item()
        stats['weight_std'] = weights_val.std().item()
        stats['weight_min'] = weights_val.min().item()
        stats['weight_max'] = weights_val.max().item()

        # Weight entropy
        weight_entropy = -(weights_val * torch.log(weights_val + eps)).sum(dim=1)
        stats['weight_entropy_mean'] = weight_entropy.mean().item()
        stats['weight_max_entropy'] = float(np.log(K))
        stats['weight_entropy_ratio'] = (
            weight_entropy.mean() / np.log(K)
        ).item()

        # Weight dominance
        max_weights, _ = weights_val.max(dim=1)
        stats['weight_dominance_mean'] = max_weights.mean().item()
        stats['weight_single_prototype_ratio'] = (
            (max_weights > 0.7).float().mean().item()
        )

        # Weight constraint violations
        weight_sums = weights_val.sum(dim=1)
        stats['weight_sum_violation_max'] = torch.abs(weight_sums - 1.0).max().item()

        # Prototype utilization
        target_mask = F.one_hot(targets, num_classes=C).bool()
        pos_scores = r_scores[target_mask].view(B, K)
        _, best_pos_k = pos_scores.max(dim=1)

        proto_usage = torch.zeros(C, K, device=z.device)
        for target, k_idx in zip(targets, best_pos_k):
            proto_usage[target, k_idx] += 1

        unique_targets, target_counts = targets.unique(return_counts=True)
        for i, t in enumerate(unique_targets):
            if target_counts[i] > 0:
                proto_usage[t] = proto_usage[t] / target_counts[i]

        proto_usage_nonzero = proto_usage[proto_usage > 0]
        if len(proto_usage_nonzero) > 0:
            stats['proto_usage_entropy_mean'] = (
                -(proto_usage * torch.log(proto_usage + eps))
                .sum(dim=1)
                .mean()
                .item()
            )
        stats['proto_usage_max_mean'] = proto_usage.max(dim=1)[0].mean().item()
        stats['proto_unused_ratio'] = (
            (proto_usage.max(dim=1)[0] < 0.01).float().mean().item()
        )

        # Score statistics
        stats['r_pos_mean'] = r_pos.mean().item()
        stats['r_neg_mean'] = r_neg.mean().item()
        stats['r_margin_mean'] = (r_pos - r_neg).mean().item()

        # Per-class analysis (sample only a few classes to avoid overhead)
        unique_classes = targets.unique()
        if len(unique_classes) > 5:
            sampled_classes = unique_classes[torch.randperm(len(unique_classes))[:5]]
        else:
            sampled_classes = unique_classes

        class_margins = []
        for c in sampled_classes:
            class_mask = targets == c
            if class_mask.any():
                class_margin = (s_pos[class_mask] - s_neg[class_mask]).mean().item()
                class_margins.append(class_margin)

        if class_margins:
            stats['class_margin_imbalance'] = float(np.std(class_margins))
            stats['class_worst_margin'] = float(np.min(class_margins))

    return stats
