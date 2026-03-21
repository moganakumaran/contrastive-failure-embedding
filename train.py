"""
train.py
────────
Training script for the contrastive failure embedding model.

Trains the MLP encoder with the specified contrastive loss (NT-Xent,
SupCon, or Triplet) using Adam with linear warmup and cosine decay.
Saves the best checkpoint based on validation F1.

Usage
─────
  # Primary SupCon model (matches manuscript Table 3 results)
  python train.py --config config.yaml

  # SupCon ablation
  python train.py --config config.yaml --loss supcon --output_dir runs/supcon/

  # Dimensionality ablation
  python train.py --config config.yaml --embedding_dim 16 --output_dir runs/dim16/

  # Depth ablation
  python train.py --config config.yaml --n_layers 2 --output_dir runs/depth2/
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Local modules
sys.path.insert(0, os.path.dirname(__file__))
from encoder import FailureEmbeddingEncoder, EmbeddingRegistry
from losses  import get_loss_fn


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(data_dir: str, split: str, device: torch.device):
    """Load a CSV split and return (X_tensor, y_tensor)."""
    import pandas as pd
    path = os.path.join(data_dir, f"{split}.csv")
    df   = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("label", "is_hard_negative")]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    y = torch.tensor(df["label"].values,      dtype=torch.long).to(device)
    return X, y


# ── Learning rate schedule ────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Validation: registry-based F1 ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_f1(
    model:     FailureEmbeddingEncoder,
    X_train:   torch.Tensor,
    y_train:   torch.Tensor,
    X_val:     torch.Tensor,
    y_val:     torch.Tensor,
    threshold: float = 0.75,
) -> float:
    """
    Build a registry from training embeddings, then compute macro-F1 on
    the validation set using cosine similarity retrieval.
    """
    from sklearn.metrics import f1_score

    model.eval()
    registry = EmbeddingRegistry()

    # Encode training set in batches (registry)
    bs = 512
    for i in range(0, len(X_train), bs):
        e = model(X_train[i : i + bs])
        registry.add(e, y_train[i : i + bs])

    # Encode validation queries
    preds = []
    for i in range(0, len(X_val), bs):
        e = model(X_val[i : i + bs])
        _, pred_labels, _ = registry.query(e, threshold=threshold)
        preds.append(pred_labels.cpu().numpy())

    preds   = np.concatenate(preds)
    targets = y_val.cpu().numpy()
    return f1_score(targets, preds, average="macro", zero_division=0)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, args: argparse.Namespace) -> FailureEmbeddingEncoder:
    # ── Reproducibility seed ──────────────────────────────────────────────────
    seed = getattr(args, "seed", None) or 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg["hardware"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={seed}")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = cfg["data"]["output_dir"]
    X_train, y_train = load_split(data_dir, "train", device)
    X_val,   y_val   = load_split(data_dir, "val",   device)
    print(f"Train: {X_train.shape}   Val: {X_val.shape}")

    # Class-stratified sampling: equal class representation per batch
    # (~256/6 ≈ 42 same-class positives per anchor, as described in §2.4)
    class_counts = torch.bincount(y_train)
    sample_weights = (1.0 / class_counts[y_train].float())
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(y_train), replacement=True
    )
    dataset    = TensorDataset(X_train, y_train)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    embedding_dim = getattr(args, "embedding_dim", None) or cfg["model"]["embedding_dim"]
    n_layers      = getattr(args, "n_layers",      None) or cfg["model"]["n_layers"]
    model = FailureEmbeddingEncoder(
        input_dim=cfg["data"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        proj_hidden_dim=cfg["model"]["projection_hidden_dim"],
        proj_output_dim=cfg["model"]["projection_output_dim"],
    ).to(device)
    print(f"Model: {model.n_parameters:,} parameters  "
          f"(embedding_dim={embedding_dim}, n_layers={n_layers})")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_name = getattr(args, "loss", None) or cfg["training"]["loss_function"]
    temperature = getattr(args, "temperature", None) or cfg["training"]["temperature"]
    loss_fn   = get_loss_fn(
        loss_name,
        temperature=temperature,
        margin=cfg["training"]["triplet_margin"],
    )
    print(f"Loss: {loss_name}  τ={temperature}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    n_epochs      = cfg["training"]["n_epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]
    steps_per_epoch = len(dataloader)
    total_steps     = n_epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    base_lr         = cfg["training"]["learning_rate"]

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = getattr(args, "output_dir", None) or cfg["training"]["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────
    best_f1    = -1.0
    best_ckpt  = os.path.join(out_dir, "best.pt")
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for X_batch, y_batch in dataloader:
            # Update learning rate
            lr = get_lr(global_step, warmup_steps, total_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            embeddings = model.forward_with_projection(X_batch)
            loss       = loss_fn(embeddings, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

        avg_loss = epoch_loss / steps_per_epoch

        # ── Validate every 10 epochs ──────────────────────────────────────────
        if epoch % 10 == 0 or epoch == n_epochs:
            val_f1 = evaluate_f1(model, X_train, y_train, X_val, y_val)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:>3d}/{n_epochs}  "
                  f"loss={avg_loss:.4f}  val_F1={val_f1:.4f}  "
                  f"lr={lr:.2e}  {elapsed:.1f}s")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "val_f1":          val_f1,
                    "loss_fn":         loss_name,
                    "embedding_dim":   embedding_dim,
                    "n_layers":        n_layers,
                    "proj_hidden_dim": cfg["model"]["projection_hidden_dim"],
                    "proj_output_dim": cfg["model"]["projection_output_dim"],
                    "config":          cfg,
                }, best_ckpt)
                print(f"  ✓ New best checkpoint saved (val_F1={val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}  →  {best_ckpt}")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train the contrastive failure embedding model.")
    p.add_argument("--config",        default="config.yaml",   help="Path to config.yaml")
    p.add_argument("--loss",          default=None,
                   choices=["nt_xent", "supcon", "triplet"],
                   help="Override loss function (for ablation)")
    p.add_argument("--embedding_dim", type=int, default=None,
                   help="Override embedding dimensionality (for ablation)")
    p.add_argument("--n_layers",      type=int, default=None,
                   help="Override number of encoder layers (for ablation)")
    p.add_argument("--output_dir",    default=None,
                   help="Override checkpoint output directory")
    p.add_argument("--seed",          type=int, default=0,
                   help="Random seed for reproducibility")
    p.add_argument("--temperature",   type=float, default=None,
                   help="Override temperature (for ablation)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args)


if __name__ == "__main__":
    main()
