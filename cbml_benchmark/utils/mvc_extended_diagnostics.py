"""Read-only Proxy-CBML diagnostics. No extra forward pass or optimizer updates."""
import math
import numpy as np
import torch


def component_gradients(model, pos, neg, mvc, weight):
    """Full trainable-network parameter gradients, excluding proxy parameters.

    Separate reverse passes on the existing graph; never write .grad. MVC is
    unweighted here (also counterfactual when disabled); scale its norm by the
    actual training coefficient. Cosines are undefined for zero-norm vectors.
    """
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if not params:
        return {}
    def gradients(loss):
        if not loss.requires_grad:
            return (None,) * len(params)
        return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    gp, gn, gv = (gradients(loss) for loss in (pos, neg, mvc))
    # Dot products in double precision, one parameter at a time to limit memory.
    dots = torch.zeros(6, device=params[0].device, dtype=torch.float64)
    with torch.no_grad():
        for triplet in zip(gp, gn, gv):
            a, b, c = (g.detach().reshape(-1).double() if g is not None else None for g in triplet)
            for j, (u, v) in enumerate(((a,a),(b,b),(c,c),(a,b),(a,c),(b,c))):
                if u is not None and v is not None:
                    dots[j] += torch.dot(u, v)
    pp, nn, vv, pn, pv, nv = dots.cpu().tolist()
    p, n, v = (math.sqrt(max(0., s)) for s in (pp, nn, vv))
    main = math.sqrt(max(0., pp + nn + 2*pn))
    total = math.sqrt(max(0., main**2 + weight**2*vv + 2*weight*(pv+nv)))
    def cosine(dot, a, b):
        return max(-1., min(1., dot/(a*b))) if a > 0 and b > 0 else None
    return dict(component_pos_grad_norm=p, component_neg_grad_norm=n,
                component_main_grad_norm=main, component_mvc_grad_norm=v,
                component_weighted_mvc_grad_norm=weight*v,
                component_total_grad_norm=total,
                component_mvc_to_main_ratio=weight*v/main if main > 0 else None,
                component_mvc_main_cosine=cosine(pv+nv, v, main),
                component_mvc_pos_cosine=cosine(pv, v, p),
                component_mvc_neg_cosine=cosine(nv, v, n),
                component_pos_neg_cosine=cosine(pn, p, n))


def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def embedding_geometry(feats, labels):
    """Population centered covariance and sample-weighted scatter of unit vectors.

    Within + between = total trace. Effective rank uses normalized eigenvalues;
    it is None for zero total variance. Applies to embeddings or proxy vectors.
    """
    x = _normalize(feats)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError('Expected nonempty [N,D] vectors with N labels.')
    mean = x.mean(0)
    centered = x - mean
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(x))[::-1].clip(0.)
    trace = float(eigenvalues.sum())
    prob = eigenvalues[eigenvalues > 0] / trace if trace > 0 else np.array([])
    within = between = 0.
    for cls in np.unique(y):
        members = x[y == cls]
        center = members.mean(0)
        within += float(((members-center)**2).sum()) / len(x)
        between += len(members)*float(((center-mean)**2).sum()) / len(x)
    return dict(n_vectors=len(x), dimension=x.shape[1], n_classes=len(np.unique(y)),
                eigenvalues=eigenvalues.tolist(), covariance_trace=trace,
                effective_rank=float(np.exp(-np.sum(prob*np.log(prob)))) if len(prob) else None,
                participation_ratio=float(1./np.sum(prob**2)) if len(prob) else None,
                top_eigenvalue_fraction=float(prob[0]) if len(prob) else None,
                mean_vector_norm=float(np.linalg.norm(mean)),
                within_class_scatter=within, between_class_scatter=between,
                between_within_ratio=between/within if within > 0 else None,
                between_total_fraction=between/trace if trace > 0 else None)


def proxy_score_geometry(feats, labels, proxies, split, tail_delta=0.0):
    """Exact per-anchor population variance and distributions of cosine scores.

    Train: exclude every own-class proxy. Validation: all training proxies are
    foreign; never interpret validation label IDs as training class indices.
    CPU, blockwise products; store only scores needed for exact quantiles.
    The optional upper-tail fields are read-only diagnostics, measured over
    foreign proxies for training and the entire training bank for validation.
    """
    x = _normalize(feats)
    p = _normalize(proxies)
    c,k,d = p.shape
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(x) != len(y) or not len(x) or split not in ('train','validation'):
        raise ValueError('Expected nonempty features/labels and known split.')
    if split == 'train' and (np.any(y < 0) or np.any(y >= c)):
        raise ValueError('Invalid training labels for proxy bank.')
    flat = p.reshape(c*k,d)
    classes = np.repeat(np.arange(c), k)
    scores = []
    for start in range(0,len(x),512):
        block = x[start:start+512] @ flat.T
        if split == 'train':
            mask = y[start:start+len(block),None] != classes[None,:]
            block = block[mask].reshape(len(block),(c-1)*k)
        scores.append(block)
    scores = np.concatenate(scores,axis=0)
    variances = scores.var(axis=1)
    means = scores.mean(axis=1)
    maxima = scores.max(axis=1)
    # Detached-target analogue evaluated on full-data embeddings: not a
    # training gradient and not a validation-class supervised regularizer.
    centered = scores - means[:, None] - float(tail_delta)
    tail_active = centered > 0.
    tail_excess = np.maximum(centered, 0.)
    tail_energy = np.mean(tail_excess**2, axis=1)
    result = dict(score_scope='foreign_class_proxies' if split=='train' else 'all_training_proxies',
                  n_anchors=len(x), proxies_per_anchor=scores.shape[1],
                  pooled_score_mean=float(scores.mean()), pooled_score_std=float(scores.std()),
                  within_anchor_variance_mean=float(variances.mean()),
                  within_anchor_std_mean=float(np.sqrt(variances).mean()),
                  anchor_mean_variance=float(means.var()),
                  nearest_proxy_score_mean=float(maxima.mean()),
                  tail_delta=float(tail_delta),
                  tail_active_fraction=float(tail_active.mean()),
                  tail_excess_mean=float(tail_excess.mean()),
                  tail_penalty_mean=float(tail_energy.mean()))
    for name, values in (('score',scores),('anchor_variance',variances),('nearest_proxy_score',maxima),('tail_penalty',tail_energy)):
        for q, value in zip((10,50,90,95,99),np.percentile(values,[10,50,90,95,99])):
            result[f'{name}_p{q}'] = float(value)
    return result
