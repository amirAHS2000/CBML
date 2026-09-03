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

        # Dedicated negative prototypes [C, K, D], structurally separate from
        # `self.prototypes`. A class's positive prototypes are shaped only by
        # that class's own data (via positive selection in forward()); a
        # class's negative prototypes are shaped only by whichever *other*
        # classes' data selects them as a hard negative. The two tensors
        # never share gradients, so a class's own representative is never
        # dragged around by its role as someone else's negative anchor.
        # (Zarei-Sabzevar et al., "Prototype-Based Interpretation of the
        # Functionality of Neurons in Winner-Take-All Neural Networks",
        # TNNLS 2022 -- the +-ED-WTA model.)
        #
        # There's no existing KMeans-based initializer for this tensor (only
        # `self.prototypes` is populated externally via
        # set_prototypes_and_weights), so it starts from small random values,
        # scaled so its initial squared norm is ~1 -- the same rough scale as
        # unit-normalized embeddings and KMeans-initialized prototypes.
        self.neg_prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.prototype_per_class, self.embed_dim, device=self.device)
            / (self.embed_dim ** 0.5)
        )
       
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
        Q = self.neg_prototypes
        C, K, D = P.shape
        B = z.shape[0]

        flat_P = P.view(C * K, D)
        flat_Q = Q.view(C * K, D)

        pos_logits = (
            z @ flat_P.T
            - 0.5 * flat_P.pow(2).sum(dim=1)
        ).view(B, C, K)

        neg_logits = (
            z @ flat_Q.T
            - 0.5 * flat_Q.pow(2).sum(dim=1)
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
        # Dominant negative prototype (from Q)
        # Same selection rule as forward(), excluding the target class.
        # ---------------------------------------------------------------
        neg_score = (
            neg_logits
            + torch.log(weights).unsqueeze(0)
            + torch.log(priors).view(1, C, 1)
        )

        neg_score[rows, targets, :] = -torch.inf

        flat_neg_idx = neg_score.view(B, C * K).argmax(dim=1)
        best_neg_class_idx = flat_neg_idx // K
        best_neg_proto_idx = flat_neg_idx % K
        best_neg = flat_Q[flat_neg_idx]

        # ---------------------------------------------------------------
        # Distances
        # ---------------------------------------------------------------
        pos_dist = torch.linalg.vector_norm(z - best_pos, dim=1)
        neg_dist = torch.linalg.vector_norm(z - best_neg, dim=1)

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
        best_pos_weight = weights[targets, best_pos_idx]
        best_neg_weight = weights[best_neg_class_idx, best_neg_proto_idx]
        pos_priors_b = priors[targets]
        best_neg_priors_b = priors[best_neg_class_idx]

        softplus_arg = (
            (torch.log(best_neg_priors_b) - torch.log(pos_priors_b))
            + (torch.log(best_neg_weight) - torch.log(best_pos_weight))
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
        P = self.prototypes     # [C, K, D] -- positive prototypes
        Q = self.neg_prototypes # [C, K, D] -- dedicated negative prototypes
        W = self.weights # [C, K] -- shared mixture weights for both roles
        B = z.shape[0] # batch size
        C = self.num_classes
        K = self.prototype_per_class
        beta = torch.exp(self.theta)
        eps = 1e-9

        # Positive-side similarities [B, C, K], computed from P only (unchanged).
        flat_P = P.view(C * K, -1)
        pos_logits_full = z @ flat_P.T - 0.5 * (flat_P.norm(p=2, dim=1) ** 2)
        pos_logits_full = pos_logits_full.view(B, C, K)

        # Negative-side similarities [B, C, K], computed from the SEPARATE Q
        # tensor. This is the key structural change from the previous
        # detach-based version: negative selection never touches P at all,
        # so a class's own positive prototypes are only ever shaped by that
        # class's own data.
        flat_Q = Q.view(C * K, -1)
        neg_logits_full = z @ flat_Q.T - 0.5 * (flat_Q.norm(p=2, dim=1) ** 2)
        neg_logits_full = neg_logits_full.view(B, C, K)

        rows = torch.arange(B, device=self.device)

        # --------------------------------------------------------------
        # Selection: argmax is non-differentiable regardless, so do it
        # under no_grad on detached copies.
        # --------------------------------------------------------------
        with torch.no_grad():
            log_W = torch.log(torch.clamp(W.detach(), min=eps))
            log_priors = torch.log(torch.clamp(self.class_priors, min=eps))

            # positive selection: best prototype within the target class, from P
            pos_score = pos_logits_full.detach()[rows, targets, :] + log_W[targets]  # [B, K]
            best_pos_proto_idx = pos_score.argmax(dim=1)  # [B]

            # negative selection: best (class, prototype) among all other
            # classes, from Q
            neg_score = (
                neg_logits_full.detach()
                + log_W.unsqueeze(0)
                + log_priors.view(1, C, 1)
            )  # [B, C, K]
            neg_score[rows, targets, :] = -torch.inf

            flat_neg_idx = neg_score.view(B, C * K).argmax(dim=1)  # [B]
            best_neg_class_idx = flat_neg_idx // K
            best_neg_proto_idx = flat_neg_idx % K

            # Accumulate selection counts. Must use accumulate=True: plain
            # fancy-index += silently drops duplicate (class, k) hits when
            # two samples in the same batch pick the same slot.
            # Note: neg_proto_counts now indexes into Q (neg_prototypes),
            # not P -- same buffer, new meaning.
            self.pos_proto_counts.index_put_(
                (targets, best_pos_proto_idx),
                torch.ones_like(targets, dtype=self.pos_proto_counts.dtype),
                accumulate=True,
            )
            self.neg_proto_counts.index_put_(
                (best_neg_class_idx, best_neg_proto_idx),
                torch.ones_like(best_neg_class_idx, dtype=self.neg_proto_counts.dtype),
                accumulate=True,
            )

        # --------------------------------------------------------------
        # Gather selected components. Both sides now stay attached to the
        # live graph with no detaching required: P and Q are separate
        # parameters, so a class's positive prototype is structurally
        # protected from its role as someone else's negative anchor --
        # that role now belongs entirely to Q. Weights (W) are still
        # shared between the two roles (see class-level note above);
        # everything else is unchanged.
        # --------------------------------------------------------------
        best_pos_proto = P[targets, best_pos_proto_idx]      # [B, D], grad -> P only
        best_pos_weight = W[targets, best_pos_proto_idx]     # [B],    grad -> W
        pos_priors = self.class_priors[targets]              # [B]

        best_neg_proto = Q[best_neg_class_idx, best_neg_proto_idx]   # [B, D], grad -> Q only
        best_neg_weight = W[best_neg_class_idx, best_neg_proto_idx]  # [B],    grad -> W
        best_neg_class_prior = self.class_priors[best_neg_class_idx] # [B]

        current_loss = F.softplus(
            (torch.log(best_neg_class_prior) - torch.log(pos_priors)) +
            (torch.log(best_neg_weight) - torch.log(best_pos_weight)) +
            beta * (
                (embeddings * best_neg_proto).sum(dim=1)
                - 0.5 * (best_neg_proto * best_neg_proto).sum(dim=1)
                - (embeddings * best_pos_proto).sum(dim=1)
                + 0.5 * (best_pos_proto * best_pos_proto).sum(dim=1)
            )
        )

        if current_loss.numel() == 0:
            return torch.zeros(1, requires_grad=True, device=self.device)

        main_loss = current_loss.mean()
        return main_loss