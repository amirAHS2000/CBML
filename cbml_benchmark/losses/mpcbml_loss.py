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
        Compute evaluation-only query-to-prototype geometry statistics using
        the same positive/negative prototype-selection rules as forward().
        Positive side uses self.prototypes; negative side uses the dedicated
        self.neg_prototypes.

        This function does not participate in gradient computation and does not
        modify prototype-selection counters or any training state.
        """
        embeddings = embeddings.to(self.device)
        targets = targets.to(self.device).long()

        z = embeddings
        P = self.prototypes
        C, K, D = P.shape
        B = z.shape[0]

        flat_P = P.view(C * K, D)

        pos_logits = (
            z @ flat_P.T
            - 0.5 * flat_P.pow(2).sum(dim=1)
        ).view(B, C, K)

        weights = torch.clamp(self.weights, min=1e-9)
        priors = torch.clamp(self.class_priors, min=1e-9)

        rows = torch.arange(B, device=self.device)

        # ---------------------------------------------------------------
        # Dominant positive prototype (from P)
        # ---------------------------------------------------------------
        pos_score = (
            pos_logits[rows, targets, :]
            + torch.log(weights[targets])
        )

        best_pos_idx = pos_score.argmax(dim=1)
        best_pos = P[targets, best_pos_idx]

        # ---------------------------------------------------------------
        # Dominant negative: hardest other-class sample within this same
        # set of embeddings, mirroring forward()'s in-batch mining exactly
        # (here "batch" is the full fixed train-eval subset, size B ~ a
        # few thousand for CUB -- an [B, B] similarity matrix is a few tens
        # of MB, not a memory concern at this scale).
        # ---------------------------------------------------------------
        z_sq_norm = (z * z).sum(dim=1)  # [B]
        data_logits = z @ z.T - 0.5 * z_sq_norm.unsqueeze(0)  # [B, B]

        neg_score = data_logits + torch.log(priors[targets]).unsqueeze(0)  # [B, B]
        same_class_mask = targets.unsqueeze(1) == targets.unsqueeze(0)  # [B, B]
        neg_score = neg_score.masked_fill(same_class_mask, float('-inf'))

        best_neg_idx = neg_score.argmax(dim=1)  # [B]
        valid_neg = (~same_class_mask).any(dim=1)  # [B]
        best_neg = z[best_neg_idx]
        best_neg_class = targets[best_neg_idx]

        # ---------------------------------------------------------------
        # Distances
        # ---------------------------------------------------------------
        pos_dist = torch.linalg.vector_norm(z - best_pos, dim=1)
        neg_dist = torch.linalg.vector_norm(z - best_neg, dim=1)

        # Restrict all stats below to rows that actually had a valid
        # negative candidate.
        pos_dist = pos_dist[valid_neg]
        neg_dist = neg_dist[valid_neg]

        # Positive-negative distance gap:
        # positive when the dominant negative is farther than the
        # dominant positive.
        gap = neg_dist - pos_dist

        # ---------------------------------------------------------------
        # Full softplus argument and "gate" (sigmoid of that argument),
        # exactly matching forward()'s loss term. gate == softplus'(x),
        # i.e. the multiplier on the gradient magnitude flowing into the
        # winning positive prototype this step. gate near 1 means samples
        # are still far from the margin (large steps); gate near 0 means
        # the margin is basically satisfied (steps have mostly died out).
        # ---------------------------------------------------------------
        beta = torch.exp(self.theta)
        best_pos_weight = weights[targets, best_pos_idx][valid_neg]
        pos_priors_b = priors[targets][valid_neg]
        best_neg_priors_b = priors[best_neg_class][valid_neg]

        softplus_arg = (
            (torch.log(best_neg_priors_b) - torch.log(pos_priors_b))
            + (-torch.log(best_pos_weight))  # negative side has no mixture weight
            + beta * 0.5 * (pos_dist.pow(2) - neg_dist.pow(2))
        )
        gate = torch.sigmoid(softplus_arg)

        result = {
            "pos_mean": pos_dist.mean(),
            "neg_mean": neg_dist.mean(),
            "neg_min": neg_dist.min(),
            "neg_p05": torch.quantile(neg_dist, 0.05),
            "neg_p10": torch.quantile(neg_dist, 0.10),
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
        # Negative side: instead of a learned per-class negative prototype,
        # use the hardest *real* data point in this batch belonging to a
        # different class (in-batch hard-negative mining). This matches the
        # base Contrastive Bayesian Analysis (CBML) construction: no
        # separate negative-prototype parameter exists, so there is nothing
        # for `prototypes` to be "corrupted" by, and gradient flows back
        # into the network through whichever sample gets selected, exactly
        # as with a standard triplet/contrastive hard-negative term.
        #
        # Pairwise similarity uses the same squared-Euclidean expansion as
        # the positive side, just with another embedding z_j in place of a
        # prototype p: z_i . z_j - 0.5||z_j||^2. There is no per-sample
        # mixture weight (a raw data point isn't one of K slots), so the
        # log(weight) term is simply omitted (equivalent to weight=1) on the
        # negative side only.
        # --------------------------------------------------------------
        z_sq_norm = (z * z).sum(dim=1)  # [B], ~1 for each row if z is unit-normalized
        data_logits = z @ z.T - 0.5 * z_sq_norm.unsqueeze(0)  # [B, B]; entry [i,j] = z_i . z_j - 0.5||z_j||^2

        with torch.no_grad():
            log_W = torch.log(torch.clamp(W.detach(), min=eps))
            log_priors = torch.log(torch.clamp(self.class_priors, min=eps))

            # positive selection: best prototype within the target class, from P (unchanged)
            pos_score = pos_logits_full.detach()[rows, targets, :] + log_W[targets]  # [B, K]
            best_pos_proto_idx = pos_score.argmax(dim=1)  # [B]

            # negative selection: hardest other-class data point in the batch.
            # score[i, j] uses sample j's own class prior; no weight term.
            neg_score = data_logits.detach() + log_priors[targets].unsqueeze(0)  # [B, B]
            same_class_mask = targets.unsqueeze(1) == targets.unsqueeze(0)  # [B, B], True incl. diagonal
            neg_score = neg_score.masked_fill(same_class_mask, float('-inf'))

            best_neg_idx = neg_score.argmax(dim=1)  # [B], index into the batch
            # Rows with no valid negative candidate at all (every sample in
            # the batch shares this row's class -- shouldn't happen with a
            # multi-identity batch sampler, but guarded rather than assumed).
            valid_neg = (~same_class_mask).any(dim=1)  # [B]

            self.pos_proto_counts.index_put_(
                (targets, best_pos_proto_idx),
                torch.ones_like(targets, dtype=self.pos_proto_counts.dtype),
                accumulate=True,
            )
            # neg_proto_counts is not updated here: negatives are now
            # individual data points, not (class, k) prototype slots.

        # --------------------------------------------------------------
        # Gather selected components. Positive prototype: grad -> P only,
        # as always. Negative: grad -> z itself (i.e. back into the
        # network), via whichever batch index was selected -- not detached,
        # since the whole point of this design is that the negative side
        # is a real, differentiable data point.
        # --------------------------------------------------------------
        best_pos_proto = P[targets, best_pos_proto_idx]      # [B, D], grad -> P only
        best_pos_weight = W[targets, best_pos_proto_idx]     # [B],    grad -> W
        pos_priors = self.class_priors[targets]              # [B]

        best_neg_embed = z[best_neg_idx]                     # [B, D], grad -> z (network) only
        best_neg_class = targets[best_neg_idx]                # [B]
        best_neg_class_prior = self.class_priors[best_neg_class]  # [B]

        current_loss = F.softplus(
            (torch.log(best_neg_class_prior) - torch.log(pos_priors)) +
            (-torch.log(best_pos_weight)) +  # negative side has no mixture weight (log(1) = 0)
            beta * (
                (embeddings * best_neg_embed).sum(dim=1)
                - 0.5 * (best_neg_embed * best_neg_embed).sum(dim=1)
                - (embeddings * best_pos_proto).sum(dim=1)
                + 0.5 * (best_pos_proto * best_pos_proto).sum(dim=1)
            )
        )

        current_loss = current_loss[valid_neg]

        if current_loss.numel() == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)

        main_loss = current_loss.mean()
        return main_loss