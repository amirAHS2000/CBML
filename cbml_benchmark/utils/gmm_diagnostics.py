"""Diagnostics specific to the hard-component Gaussian-mixture posterior.

Read-only: evaluation uses class prior + log mixture weight - beta*distance2/2.
These selections, unlike a cosine nearest-center approximation, match the loss.
"""
import time

import numpy as np
import torch


def sampled_gmm_stats(details, criterion):
    odds, posterior = details['log_odds'], details['posterior']
    grad_means = criterion.proxies.grad
    grad_weights = criterion.mixture_weights.grad
    labels = details['labels']
    foreign = torch.zeros(criterion.num_classes, dtype=torch.bool, device=labels.device)
    foreign[labels] = True
    return {
        'log_odds_mean': float(odds.mean()),
        'log_odds_p10': float(torch.quantile(odds.float(), .10)),
        'log_odds_p90': float(torch.quantile(odds.float(), .90)),
        'posterior_mean': float(posterior.mean()),
        'posterior_below_half_fraction': float((posterior < .5).float().mean()),
        'positive_log_score_mean': float(details['positive_log_score'].mean()),
        'negative_log_score_mean': float(details['negative_log_score'].mean()),
        'positive_distance2_mean': float(details['positive_distance2'].mean()),
        'negative_distance2_mean': float(details['negative_distance2'].mean()),
        'selected_positive_weight_mean': float(details['selected_positive_weight'].mean()),
        'selected_negative_weight_mean': float(details['selected_negative_weight'].mean()),
        'beta': float(criterion.beta.detach()),
        'log_beta_grad': float(criterion.log_beta.grad) if criterion.log_beta.grad is not None else None,
        'proxy_grad_norm': float(grad_means.norm()) if grad_means is not None else None,
        'mixture_weight_grad_norm': float(grad_weights.norm()) if grad_weights is not None else None,
        'foreign_only_proxy_grad_norm': float(grad_means[~foreign].norm()) if grad_means is not None else None,
        'foreign_only_weight_grad_norm': float(grad_weights[~foreign].norm()) if grad_weights is not None else None,
        'active_positive_proxies': int(torch.unique(labels * criterion.k + details['pos_k']).numel()),
    }


@torch.no_grad()
def gmm_proxy_snapshot(diagnostics, iteration, feats, labels, criterion):
    start = time.perf_counter()
    means = criterion.proxies.detach().cpu().numpy().astype(np.float64)
    weights = criterion.mixture_weights.detach().cpu().numpy().astype(np.float64)
    beta = float(criterion.beta.detach())
    priors = criterion.class_log_priors.detach().cpu().numpy().astype(np.float64)
    c, k, d = means.shape
    flat = means.reshape(c*k, d)
    norm = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    normalized = flat / norm
    x = feats.astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = np.asarray(labels, dtype=np.int64)
    component_log_prior = (priors[:,None] + np.log(weights)).reshape(-1)
    mu_sq = (flat * flat).sum(1)
    class_ids = np.repeat(np.arange(c), k)
    counts = np.zeros((c,k),dtype=np.int64)
    cos_sums = np.zeros((c,k),dtype=np.float64)
    centroid_sums = np.zeros((c,k,d),dtype=np.float64)
    neg_counts = np.zeros(c*k,dtype=np.int64)
    neg_weights = np.zeros(c*k,dtype=np.float64)
    log_odds_all=[]
    p_all=[]
    for offset in range(0,len(x),512):
        xb = x[offset:offset+512]
        yb = y[offset:offset+512]
        d2 = np.maximum((xb*xb).sum(1,keepdims=True)+mu_sq[None,:]-2*xb@flat.T,0.)
        scores = component_log_prior[None,:] - 0.5*beta*d2
        same = yb[:,None]==class_ids[None,:]
        positive_masked = np.where(same,scores,-np.inf)
        negative_masked = np.where(~same,scores,-np.inf)
        pos_ids=positive_masked.argmax(1)
        neg_ids=negative_masked.argmax(1)
        logpos=positive_masked[np.arange(len(xb)),pos_ids]
        logneg=negative_masked[np.arange(len(xb)),neg_ids]
        odds=logneg-logpos
        # sigmoid(log_odds) is the NLL derivative w.r.t. log S_negative.
        sig=np.exp(-np.logaddexp(0.,-odds))
        log_odds_all.extend(odds)
        p_all.extend(np.exp(-np.logaddexp(0.,odds)))
        np.add.at(neg_counts,neg_ids,1)
        np.add.at(neg_weights,neg_ids,beta*sig)
        pc=pos_ids//k; pj=pos_ids%k
        for b in range(len(xb)):
            a,j=pc[b],pj[b]
            counts[a,j]+=1
            cos_sums[a,j]+=float(xb[b]@normalized[pos_ids[b]])
            centroid_sums[a,j]+=xb[b]
    cos=np.full((c,k),np.nan)
    align=np.full((c,k),np.nan)
    for i in range(c):
        for j in range(k):
            if counts[i,j]:
                cos[i,j]=cos_sums[i,j]/counts[i,j]
                center=centroid_sums[i,j]
                align[i,j]=float(center@normalized[i*k+j]/max(np.linalg.norm(center),1e-12))
    ref=diagnostics.initial_proxies.numpy().reshape(-1,d) if diagnostics.initial_proxies is not None else flat
    ref=ref/np.maximum(np.linalg.norm(ref,axis=1,keepdims=True),1e-12)
    current_similarity=(normalized*ref).sum(1).reshape(c,k)
    nearest_other=np.empty(c*k)
    same_similarity=np.full((c,k),np.nan)
    for offset in range(0,c*k,512):
        sub=normalized[offset:offset+512]@normalized.T
        blocked=class_ids[offset:offset+512,None]==class_ids[None,:]
        nearest_other[offset:offset+len(sub)]=np.where(blocked,-np.inf,sub).max(axis=1)
    if k>1:
        norm3=normalized.reshape(c,k,d)
        within=norm3@norm3.transpose(0,2,1)
        same_similarity=(within.sum(-1)-1.)/(k-1)
    nc=neg_counts.reshape(c,k)
    nw=neg_weights.reshape(c,k)
    unused=counts==0
    def value(v,i,j):
        z=float(v[i,j]);return z if np.isfinite(z) else ''
    for i in range(c):
        for j in range(k):
            diagnostics.proxy_csv.writerow([
                iteration,i,j,int(counts[i,j]),value(cos,i,j),value(align,i,j),
                float(np.linalg.norm(means[i,j])),value(current_similarity,i,j),
                float(nearest_other[i*k+j]),value(same_similarity,i,j),
                int(nc[i,j]),float(nw[i,j]),float(weights[i,j]),
            ])
    diagnostics.proxy_file.flush()
    odds=np.array(log_odds_all)
    diagnostics.event('proxy_snapshot',iteration=iteration,loss_type='gmm_proxy_loss',
        active_proxies=int((counts>0).sum()),total_proxies=c*k,empty_proxies=int(unused.sum()),
        mean_assigned_cosine=float(np.nansum(cos*counts)/len(y)),
        mean_cosine_to_initial=float(current_similarity.mean()),
        posterior_mean=float(np.mean(p_all)),
        log_odds_mean=float(odds.mean()),
        posterior_below_half_fraction=float((odds>0).mean()),
        beta=beta,mixture_weight_min=float(weights.min()),mixture_weight_max=float(weights.max()),
        mixture_weight_sum_error=float(np.abs(weights.sum(1)-1).max()),
        mixture_entropy_mean=float(-(weights*np.log(weights)).sum(1).mean()),
        negative_selected_count=int(nc.sum()),
        unused_negative_selections=int(nc[unused].sum()),
        unused_negative_fraction=float(nc[unused].sum()/max(nc.sum(),1)),
        total_negative_score_weight=float(nw.sum()),
        unused_negative_score_weight=float(nw[unused].sum()),
        unused_negative_weight_fraction=float(nw[unused].sum()/nw.sum()) if nw.sum() else None,
        total_s=time.perf_counter()-start)
