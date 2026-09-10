import torch
import torch.nn as nn
import torch.nn.functional as F
from cbml_benchmark.losses.registry import LOSS


@LOSS.register('mpcbml_loss')
class MpcbmlLoss(nn.Module):
    def __init__(self, cfg):
        super(MpcbmlLoss, self).__init__()

        self.device_name = getattr(cfg.MODEL, 'DEVICE', 'cuda')
        self.device = torch.device(self.device_name)
        self.embed_dim = getattr(cfg.MODEL.HEAD, 'DIM', 512)
        self.num_classes = getattr(cfg.LOSSES.MPCBML_LOSS, 'N_CLASSES', 100)

        theta_is_learnable = getattr(cfg.LOSSES.MPCBML_LOSS, 'THETA_IS_LEARNABLE', False)
        init_theta = getattr(cfg.LOSSES.MPCBML_LOSS, 'INIT_THETA', 1.0)        
        if theta_is_learnable:
            self.theta = nn.Parameter(
                torch.tensor(init_theta, device=self.device),
                requires_grad=True
            )
        else:
            self.theta = nn.Parameter(
                torch.tensor(init_theta, device=self.device),
                requires_grad=False
            )

        self.prototype_per_class = getattr(cfg.LOSSES.MPCBML_LOSS, 'PROTOTYPE_PER_CLASS', 3)
        # Positive prototypes [C, K, D]: updated only by each class's own data.
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
        )

        # NOTE: there is no learned negative-prototype tensor in this version.
        # The negative side of the loss is sourced directly from real batch
        # embeddings (in-batch hardest-negative mining), matching the base
        # Contrastive Bayesian Analysis (CBA/CBML) construction this loss was
        # extended from. See forward() for the selection logic. A prior
        # iteration of this file used a dedicated `neg_prototypes` parameter
        # (Zarei-Sabzevar et al., TNNLS 2022, +-ED-WTA); that approach is not
        # used here, but is a reasonable alternative to revisit later.

        # Mixture weights [C, K]
        # Be updated with Lagrange Multiplier
        self.weights = nn.Parameter(
            torch.ones(self.num_classes, self.prototype_per_class, device=self.device)
            / self.prototype_per_class,
            requires_grad=True
        )

        # Class priors [C]
        priors_list = getattr(cfg.LOSSES.MPCBML_LOSS, 'CLASS_PRIORS', [1.0/self.num_classes]*self.num_classes)
        self.register_buffer('class_priors',
            torch.tensor(priors_list, device=self.device)
        )

        self.register_buffer('pos_proto_counts', torch.zeros(self.num_classes, self.prototype_per_class, dtype=torch.long, device=self.device))
        # `neg_proto_counts` is kept (unincremented) only so existing
        # trainer.py diagnostic/logging code that reads it doesn't break.
        # There is no (class, k) negative-prototype slot anymore -- negatives
        # are individual data points now, see forward().
        self.register_buffer('neg_proto_counts', torch.zeros(self.num_classes, self.prototype_per_class, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def set_prototypes_and_weights(self, prototypes, cluster_sizes):
        prototypes = prototypes.to(self.device)
        self.prototypes.copy_(prototypes)

        # Save a frozen copy of the initial prototypes for monitoring
        # self.initial_prototypes = prototypes

        # if cluster_sizes is not None and \
        #     cluster_sizes.shape == (self.num_classes, self.prototype_per_class):
        #     cluster_sizes = cluster_sizes.to(self.device).float()
        #     normalized_weights = cluster_sizes / (cluster_sizes.sum(dim=1, keepdim=True) + 1e-9)
        #     self.weights.copy_(normalized_weights)
        # else:
        #     # Fallback to uniform
        #     self.weights.fill_(1.0 / self.prototype_per_class)
        self.weights.fill_(1.0 / self.prototype_per_class)
           
        torch.cuda.empty_cache()

    def constrained_weight_update(self):
        if self.weights.grad is None:
            return
       
        # Get gradients [C, K]
        grad_w = self.weights.grad
       
        # Compute mean gradient per class
        mean_grad = grad_w.mean(dim=1, keepdim=True)  # [C, 1]
       
        grad_w.sub_(mean_grad)
    
    @torch.no_grad()
    def compute_dominant_negative_stats(self, embeddings, targets):
        """
        Compute evaluation-only query-to-prototype/data geometry statistics.

        Positive side: dominant (argmax) prototype from self.prototypes, as
        always.

        Negative side: forward() no longer selects a single hardest negative
        -- it aggregates over every other-class embedding in the batch via a
        prior-weighted log-sum-exp (matching CBML's own set-based negative
        construction). Two different things are reported here to reflect
        that split:
          - neg_mean / neg_min / neg_p05 / neg_p10 / gap_* : distance stats
            to the single *hardest* other-class point, computed purely for
            continuity with earlier runs' norm-shrinkage/monitoring analysis.
            These no longer correspond to what actually drives gradients.
          - gate_mean / gate_p90 : now computed from the true aggregate
            softplus argument (matching forward() exactly), so this is the
            one that reflects real training dynamics under the current loss.

        This function does not participate in gradient computation and does
        not modify prototype-selection counters or any training state.
        """
        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device).long()

        z = embeddings
        P = self.prototypes
        C, K, D = P.shape
        B = z.shape[0]
        beta = torch.exp(self.theta)
        eps = 1e-9

        flat_P = P.view(C * K, D)

        pos_logits = (
            z @ flat_P.T
            - 0.5 * flat_P.pow(2).sum(dim=1)
        ).view(B, C, K)

        weights = torch.clamp(self.weights, min=eps)
        priors = torch.clamp(self.class_priors, min=eps)

        rows = torch.arange(B, device=self.device)

        # ---------------------------------------------------------------
        # Dominant positive prototype (from P) -- unchanged.
        # ---------------------------------------------------------------
        pos_score = (
            pos_logits[rows, targets, :]
            + torch.log(weights[targets])
        )

        best_pos_idx = pos_score.argmax(dim=1)
        best_pos = P[targets, best_pos_idx]
        pos_dist = torch.linalg.vector_norm(z - best_pos, dim=1)

        # ---------------------------------------------------------------
        # All-pairs data similarity, same as forward().
        # ---------------------------------------------------------------
        z_sq_norm = (z * z).sum(dim=1)  # [B]
        data_logits = z @ z.T - 0.5 * z_sq_norm.unsqueeze(0)  # [B, B]
        same_class_mask = targets.unsqueeze(1) == targets.unsqueeze(0)  # [B, B]
        valid_neg = (~same_class_mask).any(dim=1)  # [B]

        # ---------------------------------------------------------------
        # (1) Pure-diagnostic "hardest single negative" distance stats --
        # for continuity with prior runs only, not used by forward() anymore.
        # ---------------------------------------------------------------
        hardest_neg_logits = data_logits.masked_fill(same_class_mask, float('-inf'))
        best_neg_idx = hardest_neg_logits.argmax(dim=1)
        best_neg = z[best_neg_idx]
        neg_dist = torch.linalg.vector_norm(z - best_neg, dim=1)

        pos_dist_v = pos_dist[valid_neg]
        neg_dist_v = neg_dist[valid_neg]
        gap = neg_dist_v - pos_dist_v

        # ---------------------------------------------------------------
        # (2) The aggregate quantity that actually drives forward()'s
        # gradient now: log(sum_j prior_j * exp(beta*logit_ij)), matching
        # forward() exactly.
        # ---------------------------------------------------------------
        log_priors_per_sample = torch.log(priors[targets])  # [B]
        neg_agg_terms = log_priors_per_sample.unsqueeze(0) + beta * data_logits  # [B, B]
        neg_agg_terms = neg_agg_terms.masked_fill(same_class_mask, float('-inf'))
        neg_agg = torch.logsumexp(neg_agg_terms, dim=1)  # [B]

        best_pos_weight = weights[targets, best_pos_idx]
        pos_priors_b = priors[targets]
        pos_term = (
            torch.log(pos_priors_b)
            + torch.log(best_pos_weight)
            + beta * pos_logits[rows, targets, best_pos_idx]
        )  # [B]

        softplus_arg = (neg_agg - pos_term)[valid_neg]
        gate = torch.sigmoid(softplus_arg)

        result = {
            "pos_mean": pos_dist_v.mean(),
            "neg_mean": neg_dist_v.mean(),
            "neg_min": neg_dist_v.min(),
            "neg_p05": torch.quantile(neg_dist_v, 0.05),
            "neg_p10": torch.quantile(neg_dist_v, 0.10),
            "gap_mean": gap.mean(),
            "gap_p10": torch.quantile(gap, 0.10),
            "gate_mean": gate.mean(),
            "gate_p90": torch.quantile(gate, 0.90),
            "proto_norm_mean": P.norm(p=2, dim=2).mean(),
        }

        return {
            key: value.detach()
            for key, value in result.items()
        }

    def forward(self, embeddings, targets):

        assert embeddings.size(0) == targets.size(0), \
            f"feats.size(0): {embeddings.size(0)} is not equal to labels.size(0): {targets.size(0)}"

        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device)

        z = embeddings # [B, D]
        P = self.prototypes # [C, K, D] -- positive prototypes only
        W = self.weights # [C, K]
        B = z.shape[0] # batch size
        C = self.num_classes
        K = self.prototype_per_class
        beta = torch.exp(self.theta)
        eps = 1e-9

        # Positive-side similarities [B, C, K], computed from P only (unchanged).
        flat_P = P.view(C * K, -1)
        pos_logits_full = z @ flat_P.T - 0.5 * (flat_P.norm(p=2, dim=1) ** 2)
        pos_logits_full = pos_logits_full.view(B, C, K)

        rows = torch.arange(B, device=self.device)

        # --------------------------------------------------------------
        # Positive selection: unchanged -- best prototype within the
        # target class, from P.
        # --------------------------------------------------------------
        with torch.no_grad():
            log_W = torch.log(torch.clamp(W.detach(), min=eps))
            pos_score = pos_logits_full.detach()[rows, targets, :] + log_W[targets]  # [B, K]
            best_pos_proto_idx = pos_score.argmax(dim=1)  # [B]

            self.pos_proto_counts.index_put_(
                (targets, best_pos_proto_idx),
                torch.ones_like(targets, dtype=self.pos_proto_counts.dtype),
                accumulate=True,
            )
            # neg_proto_counts is not updated: negatives are no longer a
            # single selected (class, k) slot -- see below.

        best_pos_proto = P[targets, best_pos_proto_idx]      # [B, D], grad -> P only
        best_pos_weight = W[targets, best_pos_proto_idx]     # [B],    grad -> W
        pos_priors = self.class_priors[targets]              # [B]

        pos_term = (
            torch.log(torch.clamp(pos_priors, min=eps))
            + torch.log(torch.clamp(best_pos_weight, min=eps))
            + beta * (
                (embeddings * best_pos_proto).sum(dim=1)
                - 0.5 * (best_pos_proto * best_pos_proto).sum(dim=1)
            )
        )  # [B]; log(prior_pos * w_pos) + beta * pos_logit

        # --------------------------------------------------------------
        # Negative side: matching CBML's own construction (cbml.py) rather
        # than a single "winner take all" hardest negative. CBML aggregates
        # over a *set* of other-class pairs via
        #   neg_loss = log(1 + sum_j exp((sim_ij - neg_a) / neg_b))
        # i.e. a soft log-sum-exp over every confusable candidate, not just
        # the single hardest one. We adopt the same structural idea here:
        # aggregate over every other-class embedding in the batch via a
        # log-sum-exp, additively weighted by each candidate's class prior
        # (log(prior_j) + beta*logit_ij inside the sum). This is the proper
        # marginal-likelihood generalization of the previous argmax (MAP)
        # selection -- if one candidate dominates the sum, logsumexp reduces
        # to (approximately) that single term, so the old behavior is a
        # limiting case of this, not something unrelated to it.
        #
        # Because it's now an aggregate over the whole batch rather than one
        # selected index, there's no single "index into z" to gather live --
        # instead every other-class embedding in the batch contributes to
        # (and receives gradient from) this term directly, similar to how
        # CBML's own neg_pair_ sum works.
        # --------------------------------------------------------------
        z_sq_norm = (z * z).sum(dim=1)  # [B]
        data_logits = z @ z.T - 0.5 * z_sq_norm.unsqueeze(0)  # [B, B]; entry [i,j] = z_i . z_j - 0.5||z_j||^2

        with torch.no_grad():
            same_class_mask = targets.unsqueeze(1) == targets.unsqueeze(0)  # [B, B], True incl. diagonal
            valid_neg = (~same_class_mask).any(dim=1)  # [B]
            log_priors_per_sample = torch.log(torch.clamp(self.class_priors[targets], min=eps))  # [B]

        neg_agg_terms = log_priors_per_sample.unsqueeze(0) + beta * data_logits  # [B, B]
        neg_agg_terms = neg_agg_terms.masked_fill(same_class_mask, float('-inf'))
        neg_agg = torch.logsumexp(neg_agg_terms, dim=1)  # [B]; log( sum_j prior_j * exp(beta*logit_ij) )

        current_loss = F.softplus(neg_agg - pos_term)
        current_loss = current_loss[valid_neg]

        if current_loss.numel() == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)

        main_loss = current_loss.mean()
        return main_loss