"""Read-only diagnostics for CBML; no new runtime dependencies."""
import csv
import json
import logging
import os
import platform
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def preserve_rng(device):
    """Extra evaluation must not consume the training sampler/augmentation RNG."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = []
    if device.type == 'cuda':
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def recall_from_matrix(sim, labels, ks=(1, 2, 4, 8)):
    """Same self-query second-largest-positive/strict > rule as RetMetric.

    Sort/partition positives once and count outranking negatives once per query.
    No approximation, no change in similarity matrix construction.
    """
    labels = np.asarray(labels)
    ranks = []
    for i, row in enumerate(sim):
        mask = labels == labels[i]
        pos = row[mask]
        if len(pos) < 2:
            raise ValueError('Recall requires at least two images per class.')
        threshold = np.partition(pos, len(pos) - 2)[-2]
        ranks.append(np.count_nonzero(row[~mask] > threshold))
    ranks = np.asarray(ranks)
    return [float(np.mean(ranks < k)) for k in ks]


def instance_geometry(sim, labels):
    """Read-only geometry on the existing retrieval matrix, excluding self.

    Pooled moments count ordered non-self pairs, not class-balanced averages.
    Nearest-gap summaries require both same-class and foreign-class peers.
    This does not modify RetMetric's recall/tie convention.
    """
    labels = np.asarray(labels)
    moments = {'same': [0, 0., 0.], 'different': [0, 0., 0.]}
    near_pos, near_neg = [], []
    for i, row in enumerate(sim):
        same = labels == labels[i]
        other = ~same
        same[i] = False
        pos, neg = row[same], row[other]
        for key, vals in (('same', pos), ('different', neg)):
            v = vals.astype(np.float64, copy=False)
            m = moments[key]
            m[0] += len(v)
            m[1] += float(v.sum())
            m[2] += float(np.dot(v, v))
        if len(pos) and len(neg):
            near_pos.append(float(pos.max()))
            near_neg.append(float(neg.max()))
    result = {'n_images': len(labels), 'valid_queries': len(near_pos)}
    for key, (n, total, squares) in moments.items():
        result[key+'_pair_count'] = n
        result[key+'_pair_mean'] = total/n if n else None
        result[key+'_pair_std'] = float(np.sqrt(max(0., squares/n-(total/n)**2))) if n else None
    a, b = np.asarray(near_pos), np.asarray(near_neg)
    gap = a-b
    for key, v in (('near_pos', a), ('near_neg', b), ('instance_gap', gap)):
        result[key+'_mean'] = float(v.mean()) if len(v) else None
        for q in (10, 50, 90):
            result[key+'_p'+str(q)] = float(np.percentile(v, q)) if len(v) else None
    result['instance_violation_fraction'] = float((gap <= 0).mean()) if len(gap) else None
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class TrainingDiagnostics:
    def __init__(self, cfg, device, logger):
        self.cfg = cfg.DIAGNOSTICS
        if min(self.cfg.STATS_EVERY, self.cfg.TRAIN_RECALL_EVERY) < 1:
            raise ValueError('Diagnostic intervals must be positive.')
        self.device = device
        self.logger = logger
        self.start = time.perf_counter()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.path = Path(cfg.SAVE_DIR) / 'diagnostics' / stamp
        self.path.mkdir(parents=True, exist_ok=False)
        self.events = (self.path / 'events.jsonl').open('w', encoding='utf-8', buffering=1)
        self.steps = (self.path / 'steps.jsonl').open('w', encoding='utf-8', buffering=1)
        self.recall_file = (self.path / 'recall.csv').open('w', newline='', encoding='utf-8')
        self.recall = csv.writer(self.recall_file)
        self.recall.writerow(['iteration', 'split', 'n_images', 'recall_1', 'recall_2', 'recall_4', 'recall_8', 'extract_s', 'similarity_s', 'ranking_s', 'total_s'])
        self.recall_file.flush()
        self.proxy_file = (self.path / 'proxy_snapshots.csv').open('w', newline='', encoding='utf-8')
        self.proxy_csv = csv.writer(self.proxy_file)
        self.proxy_csv.writerow(['iteration', 'class_id', 'proxy_id', 'assigned_count', 'mean_assigned_cosine', 'centroid_cosine', 'raw_norm', 'cosine_to_initial', 'nearest_other_proxy_cosine', 'mean_same_class_proxy_cosine', 'negative_selected_count', 'negative_score_gradient_weight', 'mixture_weight'])
        self.proxy_file.flush()
        self.handler = logging.FileHandler(self.path / 'training.log', encoding='utf-8')
        self.handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(self.handler)
        (self.path / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        self.initial_proxies = None
        self.event('environment', python=platform.python_version(), platform=platform.platform(),
                   torch=torch.__version__, numpy=np.__version__, device=str(device),
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
                   torch_initial_seed=torch.initial_seed(), train_workers=cfg.DATA.NUM_WORKERS,
                   train_batch=cfg.DATA.TRAIN_BATCHSIZE, eval_batch=cfg.DATA.TEST_BATCHSIZE,
                   train_source=cfg.DATA.TRAIN_IMG_SOURCE, test_source=cfg.DATA.TEST_IMG_SOURCE)
        logger.info('Diagnostics directory: %s', self.path)

    def event(self, event_name, **data):
        self.events.write(json.dumps(_jsonable(dict(event=event_name, elapsed_s=time.perf_counter()-self.start, **data)), allow_nan=False)+'\n')

    def step(self, **data):
        self.steps.write(json.dumps(_jsonable(data), allow_nan=False)+'\n')

    def close(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
        for f in (self.events, self.steps, self.recall_file, self.proxy_file):
            f.close()

    def set_initial_proxies(self, criterion):
        if hasattr(criterion, 'proxies'):
            self.initial_proxies = F.normalize(criterion.proxies.detach().cpu(), dim=-1).clone()

    def record_recall(self, iteration, split, n, recalls, extraction, similarity, ranking, total):
        self.recall.writerow([iteration, split, n, *recalls, extraction, similarity, ranking, total])
        self.recall_file.flush()
        self.event('evaluation', iteration=iteration, split=split, recalls=recalls,
                   n_images=n, extract_s=extraction, similarity_s=similarity, ranking_s=ranking, total_s=total)

    @torch.no_grad()
    def proxy_snapshot(self, iteration, feats, labels, criterion):
        """Whole training set assignment counts and centroid alignment; CPU only."""
        if getattr(criterion, 'is_gmm_proxy_loss', False):
            from cbml_benchmark.utils.gmm_diagnostics import gmm_proxy_snapshot
            return gmm_proxy_snapshot(self, iteration, feats, labels, criterion)
        start = time.perf_counter()
        p_raw = criterion.proxies.detach().cpu().numpy()
        p = p_raw / np.maximum(np.linalg.norm(p_raw, axis=-1, keepdims=True), 1e-12)
        c, k, d = p.shape
        x = feats / np.maximum(np.linalg.norm(feats, axis=-1, keepdims=True), 1e-12)
        y = np.asarray(labels, dtype=np.int64)
        counts = np.zeros((c,k), dtype=np.int64)
        sims = np.full((c,k), np.nan)
        centroids = np.full((c,k), np.nan)
        pos_all, neg_all = [], []
        neg_counts = np.zeros(c*k, dtype=np.int64)
        neg_weights = np.zeros(c*k, dtype=np.float64)
        flat = p.reshape(c*k, d)
        class_index = np.repeat(np.arange(c), k)
        for offset in range(0, len(x), 512):
            scores = x[offset:offset+512] @ flat.T
            same = y[offset:offset+512,None] == class_index[None,:]
            pos_all.extend(np.max(np.where(same, scores, -np.inf), axis=1))
            masked = np.where(~same, scores, -np.inf)
            neg_ids = masked.argmax(axis=1)
            neg_scores = masked[np.arange(len(scores)), neg_ids]
            neg_all.extend(neg_scores)
            # Stable sigmoid(z)/beta, i.e. the negative score coefficient.
            z = (neg_scores.astype(np.float64)-criterion.neg_a)/criterion.neg_b
            weights = np.exp(-np.logaddexp(0., -z))/criterion.neg_b
            neg_counts += np.bincount(neg_ids, minlength=c*k)
            neg_weights += np.bincount(neg_ids, weights=weights, minlength=c*k)
        for cls in range(c):
            members = x[y == cls]
            score = members @ p[cls].T
            assignment = score.argmax(axis=1)
            for j in range(k):
                mask = assignment == j
                counts[cls,j] = mask.sum()
                if mask.any():
                    sims[cls,j] = score[mask,j].mean()
                    center = members[mask].mean(0)
                    centroids[cls,j] = (center / max(np.linalg.norm(center), 1e-12)) @ p[cls,j]
        initial = self.initial_proxies.numpy() if self.initial_proxies is not None else p
        alignment = (p * initial).sum(-1)
        nearest = []
        for offset in range(0, c*k, 512):
            block = flat[offset:offset+512] @ flat.T
            mask = class_index[offset:offset+512,None] == class_index[None,:]
            nearest.extend(np.max(np.where(mask, -np.inf, block), axis=1))
        nearest = np.asarray(nearest).reshape(c,k)
        within = (p @ p.transpose(0,2,1)).sum(-1)
        within = (within - (p*p).sum(-1)) / (k-1) if k>1 else np.full((c,k), np.nan)
        norms = np.linalg.norm(p_raw, axis=-1)
        neg_counts = neg_counts.reshape(c,k)
        neg_weights = neg_weights.reshape(c,k)
        unused = counts == 0
        def val(t, a, b):
            z = float(t[a,b]); return z if np.isfinite(z) else ''
        for a in range(c):
            for b in range(k):
                self.proxy_csv.writerow([iteration,a,b,int(counts[a,b]),val(sims,a,b),val(centroids,a,b),val(norms,a,b),val(alignment,a,b),val(nearest,a,b),val(within,a,b),int(neg_counts[a,b]),float(neg_weights[a,b]),''])
        self.proxy_file.flush()
        pos_all, neg_all = np.asarray(pos_all), np.asarray(neg_all)
        gap = pos_all-neg_all
        self.event('proxy_snapshot', iteration=iteration, active_proxies=int((counts>0).sum()),
                   total_proxies=c*k, empty_proxies=int((counts==0).sum()),
                   mean_assigned_cosine=float(np.nansum(sims*counts)/counts.sum()),
                   mean_cosine_to_initial=float(alignment.mean()),
                   train_pos_mean=float(pos_all.mean()), train_neg_mean=float(neg_all.mean()),
                   train_gap_mean=float(gap.mean()), train_violation_fraction=float((gap<=0).mean()),
                   total_negative_selections=int(neg_counts.sum()),
                   unused_negative_selections=int(neg_counts[unused].sum()),
                   unused_negative_fraction=float(neg_counts[unused].sum()/neg_counts.sum()),
                   total_negative_score_weight=float(neg_weights.sum()),
                   unused_negative_score_weight=float(neg_weights[unused].sum()),
                   unused_negative_weight_fraction=float(neg_weights[unused].sum()/neg_weights.sum()) if neg_weights.sum() else None,
                   total_s=time.perf_counter()-start)


def sampled_proxy_stats(details, criterion):
    """All inputs are detached; execute only on sampled iterations."""
    s_pos, s_neg, labels = details['s_pos'], details['s_neg'], details['labels']
    gap = s_pos - s_neg
    p = criterion.proxies.detach()
    grad = criterion.proxies.grad
    active = torch.unique(labels * criterion.k + details['pos_k'])
    w_pos = torch.sigmoid((criterion.pos_a-s_pos)/criterion.pos_b)/criterion.pos_b
    w_neg = torch.sigmoid((s_neg-criterion.neg_a)/criterion.neg_b)/criterion.neg_b
    return dict(pos_mean=float(s_pos.mean()), neg_mean=float(s_neg.mean()),
                pos_min=float(s_pos.min()), neg_max=float(s_neg.max()),
                pos_p10=float(torch.quantile(s_pos.float(),.1)), neg_p90=float(torch.quantile(s_neg.float(),.9)),
                gap_mean=float(gap.mean()), gap_p10=float(torch.quantile(gap.float(),.1)),
                violation_fraction=float((gap<=0).float().mean()),
                pos_below_target=float((s_pos<criterion.pos_a).float().mean()),
                neg_above_target=float((s_neg>criterion.neg_a).float().mean()),
                pos_score_gradient_mean=float(w_pos.mean()), neg_score_gradient_mean=float(w_neg.mean()),
                active_positive_proxies=int(active.numel()),
                proxy_grad_norm=float(grad.norm()) if grad is not None else None,
                proxy_raw_norm_mean=float(p.norm(dim=-1).mean()))


def gradient_norms(model):
    groups = {'backbone': [], 'head': [], 'other': []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        group = 'backbone' if name.startswith('backbone.') else 'head' if name.startswith('headembedding.') else 'other'
        groups[group].append(p.grad.detach().float().norm())
    return {key+'_grad_norm': float(torch.stack(v).norm()) if v else None for key,v in groups.items()}
