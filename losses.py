"""
losses.py
─────────
Contrastive loss functions used in the ablation study.

  - NT-Xent     : primary loss (Chen et al., SimCLR, ICML 2020)
  - SupCon      : supervised variant (Khosla et al., NeurIPS 2020)
  - Triplet     : per-triple margin loss (ablation baseline)

All functions accept L2-normalised embedding tensors and integer class
labels. They assume a single GPU device and operate on full batches.

References
──────────
  Chen, T., et al. (2020). A simple framework for contrastive learning of
      visual representations. ICML 2020.
  Khosla, P., et al. (2020). Supervised contrastive learning. NeurIPS 2020.
  Graf, F., et al. (2021). Dissecting supervised contrastive learning. ICML 2021.
"""

import torch
import torch.nn.functional as F


# ── NT-Xent ───────────────────────────────────────────────────────────────────

def nt_xent_loss(
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss.

    For each anchor, exactly ONE positive is randomly drawn from the same
    class (analogous to one augmented view in SimCLR). All other batch
    members serve as negatives. This single-positive formulation is the key
    distinction from SupCon, which treats ALL same-class instances as
    simultaneous positives.

    Parameters
    ──────────
    embeddings  : (N, D)  — L2-normalised embedding vectors
    labels      : (N,)    — integer class labels
    temperature : float   — sharpness parameter τ (default 0.07)

    Returns
    ───────
    loss : scalar tensor
    """
    N = embeddings.size(0)
    device = embeddings.device

    # Cosine similarity matrix (N, N); diagonal is self-similarity (= 1.0)
    sim = embeddings @ embeddings.T / temperature          # (N, N)

    # Mask out the diagonal (self-comparisons)
    eye      = torch.eye(N, device=device, dtype=torch.bool)
    sim_masked = sim.masked_fill(eye, float("-inf"))

    # Positive mask: same class, excluding self
    label_col = labels.unsqueeze(1)       # (N, 1)
    label_row = labels.unsqueeze(0)       # (1, N)
    pos_mask  = (label_col == label_row) & ~eye          # (N, N)

    # Sample exactly one positive per anchor by drawing a random index from
    # the set of valid positives.  Anchors with no positives are skipped.
    # Uses Gumbel noise trick: argmax( log(U) ) over valid positions is
    # equivalent to uniform random sampling without replacement.
    n_pos = pos_mask.float().sum(dim=1)  # (N,)
    valid = n_pos > 0                    # (N,) anchors that have a positive

    # Replace invalid positions with -inf before Gumbel argmax
    gumbel = -torch.log(-torch.log(
        torch.rand_like(sim) + 1e-10
    ) + 1e-10)                           # (N, N) Gumbel noise
    gumbel_masked = torch.where(pos_mask, gumbel,
                                torch.full_like(gumbel, float("-inf")))
    pos_idx = gumbel_masked.argmax(dim=1)  # (N,) — one positive index per anchor

    # Build a one-hot positive mask for the selected positives
    one_hot_pos = torch.zeros(N, N, device=device, dtype=torch.bool)
    one_hot_pos[torch.arange(N, device=device), pos_idx] = True
    # Zero out the one-hot for invalid anchors so their loss = 0
    one_hot_pos = one_hot_pos & valid.unsqueeze(1)

    # For numerical stability, subtract row-wise max before log-sum-exp
    sim_max, _ = sim_masked.max(dim=1, keepdim=True)
    sim_stable  = sim_masked - sim_max.detach()

    exp_sim   = torch.exp(sim_stable)                     # (N, N)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    # NT-Xent loss = -log_softmax at the single selected positive
    # Mask BEFORE summing to avoid -inf * 0 = NaN (IEEE 754)
    log_num     = (sim_stable - log_denom).masked_fill(~one_hot_pos, 0.0)
    loss_all    = -log_num.sum(dim=1) * valid.float()     # 0 for invalid anchors

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_all.sum() / valid.float().sum()


# ── SupCon ────────────────────────────────────────────────────────────────────

def supcon_loss(
    embeddings:  torch.Tensor,
    labels:      torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Extends NT-Xent to the fully supervised setting by treating ALL same-class
    instances in the batch as positives for each anchor simultaneously,
    rather than a single designated positive pair.

    This objective is theoretically grounded by Graf et al. (2021), who show
    it drives class representations toward the vertices of a regular simplex,
    maximising inter-class angular separation.

    Parameters
    ──────────
    embeddings  : (N, D)  — L2-normalised embedding vectors
    labels      : (N,)    — integer class labels
    temperature : float   — sharpness parameter τ

    Returns
    ───────
    loss : scalar tensor
    """
    N = embeddings.size(0)
    device = embeddings.device

    sim   = embeddings @ embeddings.T / temperature       # (N, N)
    eye   = torch.eye(N, device=device, dtype=torch.bool)
    sim   = sim.masked_fill(eye, float("-inf"))

    label_col = labels.unsqueeze(1)
    label_row = labels.unsqueeze(0)
    pos_mask  = (label_col == label_row) & ~eye           # (N, N)

    # Check that every anchor has at least one positive
    n_pos = pos_mask.sum(dim=1)
    if (n_pos == 0).any():
        # Anchors with no positive contribute zero loss (class singletons in batch)
        valid = n_pos > 0
    else:
        valid = torch.ones(N, device=device, dtype=torch.bool)

    # Log-sum-exp denominator over all non-self entries
    sim_max, _ = sim.masked_fill(~valid.unsqueeze(1), float("-inf")).max(
        dim=1, keepdim=True
    )
    exp_sim   = torch.exp(sim - sim_max.detach())
    exp_sim   = exp_sim.masked_fill(eye, 0.0)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    # Mean log probability over positives per anchor.
    # Mask BEFORE multiplying to avoid -inf * 0 = NaN (IEEE 754).
    log_numerator = (sim - sim_max.detach() - log_denom).masked_fill(~pos_mask, 0.0)
    n_pos_clamped = n_pos.clamp(min=1).float()
    loss_per  = -(log_numerator.sum(dim=1) / n_pos_clamped) * valid.float()

    return loss_per.sum() / valid.float().sum().clamp(min=1)


# ── Triplet ───────────────────────────────────────────────────────────────────

def triplet_loss(
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    margin:     float = 0.5,
    mining:     str   = "semi-hard",
) -> torch.Tensor:
    """
    Triplet margin loss with online mining.

    Triplet loss is the weakest of the three objectives in the ablation
    (see Table 6 of the manuscript). Its per-triple optimisation is less
    data-efficient than batch-level objectives, and naive negative mining
    can produce uninformative triplets.

    Parameters
    ──────────
    embeddings : (N, D)  — L2-normalised embeddings
    labels     : (N,)    — integer class labels
    margin     : float   — margin α (default 0.5)
    mining     : str     — 'semi-hard' (default) or 'hardest'

    Returns
    ───────
    loss : scalar tensor
    """
    N      = embeddings.size(0)
    device = embeddings.device

    # Pairwise squared L2 distance using the identity:
    # ||a - b||^2 = 2 - 2 a·b  (for unit vectors)
    # clamp min=1e-12 before sqrt to avoid infinite gradients at exactly 0
    sim  = embeddings @ embeddings.T                             # (N, N)
    dist = (2 - 2 * sim).clamp(min=1e-12).sqrt()               # (N, N)

    label_col = labels.unsqueeze(1)
    label_row = labels.unsqueeze(0)
    pos_mask  = (label_col == label_row)                        # (N, N)
    neg_mask  = ~pos_mask
    eye       = torch.eye(N, device=device, dtype=torch.bool)

    pos_mask_no_self = pos_mask & ~eye
    neg_mask_no_self = neg_mask & ~eye

    # Vectorised mining (no Python for-loop over anchors)
    has_pos = pos_mask_no_self.any(dim=1)
    has_neg = neg_mask_no_self.any(dim=1)
    valid   = has_pos & has_neg

    # Use a large finite sentinel (not INF) to avoid NaN in backward pass
    # when sentinel values propagate through relu / gradient paths
    LARGE = 1e4

    if mining == "hardest":
        # Hardest positive: maximum d_ap per anchor
        d_ap_sel = torch.where(pos_mask_no_self, dist,
                               torch.zeros_like(dist)).max(dim=1).values
        # Hardest negative: minimum d_an per anchor
        d_an_sel = torch.where(neg_mask_no_self, dist,
                               torch.full_like(dist, LARGE)).min(dim=1).values
    else:
        # Semi-hard: negative farther than mean positive but within margin
        # Mean positive distance per anchor
        pos_cnt   = pos_mask_no_self.sum(dim=1).clamp(min=1).float()
        d_ap_mean = (dist * pos_mask_no_self.float()).sum(dim=1) / pos_cnt

        # Semi-hard condition per (anchor, negative) pair
        sh_mask = (neg_mask_no_self
                   & (dist > d_ap_mean.unsqueeze(1))
                   & (dist < (d_ap_mean + margin).unsqueeze(1)))

        # Hardest negative (finite fallback)
        d_hrd = torch.where(neg_mask_no_self, dist,
                            torch.full_like(dist, LARGE)).min(dim=1).values

        # Minimum semi-hard negative; fall back to hardest negative when none
        # Replace sentinel with d_hrd so no row contains LARGE after selection
        sh_fill = d_hrd.unsqueeze(1).expand_as(dist)
        d_sh    = torch.where(sh_mask, dist, sh_fill).min(dim=1).values
        d_an_sel = torch.where(sh_mask.any(dim=1), d_sh, d_hrd)
        d_ap_sel = d_ap_mean

    # Zero out loss for invalid anchors before summing (avoids nan * 0 = nan)
    loss_all = F.relu(d_ap_sel - d_an_sel + margin) * valid.float()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_all.sum() / valid.float().sum()


# ── Loss factory ──────────────────────────────────────────────────────────────

def get_loss_fn(name: str, temperature: float = 0.07, margin: float = 0.5):
    """
    Return a loss function by name.

    Parameters
    ──────────
    name        : 'nt_xent', 'supcon', or 'triplet'
    temperature : used by nt_xent and supcon
    margin      : used by triplet

    Returns
    ───────
    Callable (embeddings, labels) -> scalar loss tensor
    """
    name = name.lower().replace("-", "_")
    if name == "nt_xent":
        return lambda emb, lab: nt_xent_loss(emb, lab, temperature=temperature)
    elif name == "supcon":
        return lambda emb, lab: supcon_loss(emb, lab, temperature=temperature)
    elif name == "triplet":
        return lambda emb, lab: triplet_loss(emb, lab, margin=margin)
    else:
        raise ValueError(f"Unknown loss: {name!r}. Choose from nt_xent, supcon, triplet.")


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    N, D, K = 64, 32, 6
    emb = F.normalize(torch.randn(N, D), dim=1)
    lab = torch.randint(0, K, (N,))

    for name in ["nt_xent", "supcon", "triplet"]:
        fn   = get_loss_fn(name)
        loss = fn(emb, lab)
        print(f"{name:>8s}  loss = {loss.item():.4f}")
