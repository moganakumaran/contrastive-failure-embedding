"""
encoder.py
──────────
MLP encoder for contrastive failure embeddings.

Architecture
────────────
  Backbone: n_layers × [Linear(hidden_dim, hidden_dim) → BatchNorm → ReLU]
  Output proj: Linear(hidden_dim, embedding_dim) → L2-normalise → e ∈ S^{m-1}
  Proj head (training only): Linear(32, 64) → ReLU → Linear(64, 128) → L2-norm

  Inference path (forward):        32-dim L2-normalised embedding
  Training path (forward_with_projection): 128-dim L2-normalised projection

The L2 normalisation projects embeddings onto the unit hypersphere so that
cosine similarity reduces to a dot product, enabling efficient approximate
nearest-neighbour retrieval.  The projection head is discarded at inference.

Usage
─────
  from encoder import FailureEmbeddingEncoder
  model = FailureEmbeddingEncoder(input_dim=28, hidden_dim=128, embedding_dim=32, n_layers=3)  # default
  e = model(x)                        # shape (batch, 32), L2-normalised  (inference)
  z = model.forward_with_projection(x)  # shape (batch, 128), L2-normalised (training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPBlock(nn.Module):
    """Single hidden layer: Linear → BatchNorm → ReLU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.linear(x)))


class ProjectionHead(nn.Module):
    """
    Two-layer non-linear projection head used during contrastive training.
    Maps the encoder output (embedding_dim) to a higher-dimensional space
    where the contrastive loss is computed. Discarded at inference.

    Architecture: Linear(in_dim, hidden_dim) → ReLU → Linear(hidden_dim, out_dim)
    Default configuration: 32 → 64 → 128  (matches Khosla et al. NeurIPS 2020)
    """

    def __init__(self, in_dim: int = 32, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised projection (for contrastive loss computation)."""
        return F.normalize(self.net(e), p=2, dim=1)


class FailureEmbeddingEncoder(nn.Module):
    """
    Contrastive failure embedding encoder.

    Parameters
    ──────────
    input_dim        : Dimensionality of the input feature vector (default 28).
    hidden_dim       : Width of each hidden MLP layer (default 128).
    embedding_dim    : Output embedding dimensionality (default 32).
    n_layers         : Number of hidden layers before the output projection
                       (default 3; ablated over {1, 2, 3, 4}).
    proj_hidden_dim  : Intermediate dim of the projection head (default 64).
    proj_output_dim  : Output dim of the projection head (default 128).
    """

    def __init__(
        self,
        input_dim:      int = 28,
        hidden_dim:     int = 128,
        embedding_dim:  int = 32,
        n_layers:       int = 3,
        proj_hidden_dim: int = 64,
        proj_output_dim: int = 128,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(_MLPBlock(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.backbone    = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.proj_head   = ProjectionHead(embedding_dim, proj_hidden_dim, proj_output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference path: returns 32-dim L2-normalised embedding.

        Parameters
        ──────────
        x : (batch, input_dim)  — percentile-relative feature vector

        Returns
        ───────
        e : (batch, embedding_dim)  — L2-normalised embedding
        """
        return F.normalize(self.output_proj(self.backbone(x)), p=2, dim=1)

    def forward_with_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training path: returns 128-dim L2-normalised projection head output.

        The projection head is applied on top of the 32-dim encoder output.
        This output is passed to the contrastive loss during training and
        discarded at inference.

        Parameters
        ──────────
        x : (batch, input_dim)  — percentile-relative feature vector

        Returns
        ───────
        z : (batch, proj_output_dim)  — L2-normalised projection
        """
        return self.proj_head(self.output_proj(self.backbone(x)))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward(); useful for clarity in evaluation code."""
        return self.forward(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Registry helper ───────────────────────────────────────────────────────────

class EmbeddingRegistry:
    """
    Stores embeddings and class labels for resolved failure episodes.
    Supports cosine similarity retrieval.

    In a production deployment this would be backed by a vector database
    or approximate nearest-neighbour index (e.g., FAISS, ScaNN).
    Here it operates over dense tensors for evaluation purposes.
    """

    def __init__(self):
        self.embeddings: torch.Tensor = None   # (N, embedding_dim)
        self.labels:     torch.Tensor = None   # (N,)

    def add(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a batch of embeddings with their class labels."""
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if self.embeddings is None:
            self.embeddings = embeddings.detach()
            self.labels     = labels.detach()
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings.detach()])
            self.labels     = torch.cat([self.labels,     labels.detach()])

    def query(
        self,
        query_embedding: torch.Tensor,
        threshold: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the best match for each query embedding.

        Parameters
        ──────────
        query_embedding : (Q, embedding_dim)
        threshold       : cosine similarity threshold for a positive match

        Returns
        ───────
        sim_scores   : (Q,)  — cosine similarity to nearest registry entry
        pred_labels  : (Q,)  — predicted class label (-1 if below threshold)
        is_recurrence: (Q,)  — bool, True if similarity >= threshold
        """
        if self.embeddings is None:
            raise RuntimeError("Registry is empty. Call add() first.")

        # Cosine similarity: (Q, N)
        sims = query_embedding @ self.embeddings.T
        best_sim, best_idx = sims.max(dim=1)
        pred_labels  = torch.where(best_sim >= threshold,
                                   self.labels[best_idx],
                                   torch.full_like(self.labels[best_idx], -1))
        is_recurrence = best_sim >= threshold
        return best_sim, pred_labels, is_recurrence

    def clear(self) -> None:
        self.embeddings = None
        self.labels     = None


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FailureEmbeddingEncoder(input_dim=28, hidden_dim=128,
                                    embedding_dim=32, n_layers=3)
    print(model)
    print(f"Trainable parameters: {model.n_parameters:,}")

    x = torch.randn(16, 28)
    e = model(x)
    z = model.forward_with_projection(x)
    print(f"Input  shape          : {x.shape}")
    print(f"Embedding shape       : {e.shape}  (inference path, 32-dim)")
    print(f"Projection shape      : {z.shape}  (training path, 128-dim)")
    print(f"Embedding L2 norms    : {e.norm(dim=1).min():.4f} – {e.norm(dim=1).max():.4f}  (should be ≈ 1.0)")
    print(f"Projection L2 norms   : {z.norm(dim=1).min():.4f} – {z.norm(dim=1).max():.4f}  (should be ≈ 1.0)")
