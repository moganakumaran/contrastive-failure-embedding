"""
baseline.py
───────────
Hand-engineered recurrence detection baseline.

The baseline applies percentile-relative normalization to the input feature
vector and computes cosine similarity directly in the normalized feature
space — no learned encoder. A similarity threshold θ, tuned on the validation
set to maximise F1, determines whether a query episode is flagged as a
recurrence of a registered failure class.

This baseline is deliberately strong: it uses the same features, normalization,
and similarity function as the contrastive embedding approach, differing only
in the absence of a learned encoder. This design isolates the contribution of
contrastive training to recurrence detection accuracy.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class PercentileNormalizer:
    """
    Fit percentile boundaries on training data; transform any split.

    In production, percentiles are computed per-feature from a rolling
    window of historical executions of the same job. Here they are estimated
    from the training split.
    """

    def __init__(self, lo: float = 5.0, hi: float = 95.0):
        self.lo = lo
        self.hi = hi
        self.p_lo: np.ndarray = None
        self.p_hi: np.ndarray = None

    def fit(self, X: np.ndarray) -> "PercentileNormalizer":
        self.p_lo = np.percentile(X, self.lo, axis=0)
        self.p_hi = np.percentile(X, self.hi, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = np.where(
            (self.p_hi - self.p_lo) > 1e-9,
            self.p_hi - self.p_lo,
            1.0,
        )
        return np.clip((X - self.p_lo) / denom, 0.0, 1.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between rows of A and rows of B.

    Returns
    ───────
    S : (|A|, |B|) — cosine similarities in [-1, 1]
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


class EngineeredBaseline:
    """
    Percentile-relative cosine similarity baseline for failure recurrence
    detection.

    Workflow
    ────────
    1. fit(X_train, y_train) — normalise and store the registry.
    2. tune_threshold(X_val, y_val) — sweep θ ∈ [0, 1] to maximise F1.
    3. predict(X_query) — return predicted labels (−1 = no recurrence found).
    4. score(X_test, y_test) — compute F1, precision, recall, AUROC.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.normalizer: PercentileNormalizer = None
        self._registry_X: np.ndarray = None
        self._registry_y: np.ndarray = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EngineeredBaseline":
        """Normalise and store the training set as the failure registry."""
        self.normalizer = PercentileNormalizer()
        self._registry_X = self.normalizer.fit_transform(X_train)
        self._registry_y = y_train.copy()
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.normalizer.transform(X)

    # ── Threshold tuning ──────────────────────────────────────────────────────

    def tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_steps: int = 100,
    ) -> float:
        """
        Sweep similarity thresholds and select θ* that maximises F1 on the
        validation set. Updates self.threshold in place.

        Returns
        ───────
        best_threshold : float
        """
        X_val_n = self._transform(X_val)
        sim      = cosine_similarity_matrix(X_val_n, self._registry_X)
        best_sim = sim.max(axis=1)         # (N_val,)
        best_idx = sim.argmax(axis=1)
        nearest_labels = self._registry_y[best_idx]

        best_f1, best_thresh = -1.0, 0.75
        for theta in np.linspace(0.0, 1.0, n_steps):
            preds = np.where(best_sim >= theta, nearest_labels, -1)
            # Treat -1 (no match) as a wrong prediction for F1 purposes
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, theta

        self.threshold = best_thresh
        return best_thresh

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict recurrence class for each query episode.

        Returns
        ───────
        pred_labels  : (N,) — predicted class (−1 = below threshold)
        sim_scores   : (N,) — cosine similarity to nearest registry entry
        """
        X_n      = self._transform(X)
        sim      = cosine_similarity_matrix(X_n, self._registry_X)
        best_sim = sim.max(axis=1)
        best_idx = sim.argmax(axis=1)
        nearest_labels = self._registry_y[best_idx]
        pred_labels    = np.where(best_sim >= self.threshold, nearest_labels, -1)
        return pred_labels, best_sim

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_std: float = 0.0,
        rng: np.random.Generator = None,
    ) -> dict:
        """
        Compute recurrence detection metrics.

        Parameters
        ──────────
        X_test    : (N, D) test features (before baseline normalization)
        y_test    : (N,)   true labels
        noise_std : float  — if > 0, add Gaussian noise to X_test features
                            before normalizing (for Experiment 3)
        rng       : random generator (for reproducible noise)

        Returns
        ───────
        dict with keys: f1, precision, recall, auroc, threshold
        """
        X_eval = X_test.copy()
        if noise_std > 0.0:
            if rng is None:
                rng = np.random.default_rng(0)
            X_eval = np.clip(X_eval + rng.normal(0, noise_std, X_eval.shape), 0.0, 1.0)

        preds, scores = self.predict(X_eval)

        # Binary recurrence: 1 if correct class matched, 0 otherwise
        is_correct_recurrence = (preds == y_test).astype(int)

        f1  = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec  = recall_score(y_test, preds, average="macro", zero_division=0)

        try:
            auroc = roc_auc_score(is_correct_recurrence, scores)
        except ValueError:
            auroc = float("nan")

        return {
            "f1":        round(f1,   4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "auroc":     round(auroc, 4),
            "threshold": round(self.threshold, 4),
        }


# ── Supervised MLP Baseline ───────────────────────────────────────────────────

class SupervisedMLPBaseline:
    """
    Supervised MLP classifier baseline.

    Same three-layer MLP backbone as FailureEmbeddingEncoder, but trained with
    cross-entropy loss and a linear classification head. Evaluated via argmax —
    no embedding registry or cosine similarity retrieval. This isolates the
    contribution of the contrastive training objective from the choice of
    architecture and features.

    Uses the same percentile-relative normalisation as the engineered baseline
    and the contrastive model.
    """

    def __init__(
        self,
        input_dim:  int   = 28,
        hidden_dim: int   = 128,
        n_layers:   int   = 3,
        n_classes:  int   = 6,
        lr:         float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs:   int   = 200,
        batch_size: int   = 256,
        seed:       int   = 0,
    ):
        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers
        self.n_classes    = n_classes
        self.lr           = lr
        self.weight_decay = weight_decay
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.seed         = seed
        self.normalizer   = PercentileNormalizer()
        self._model       = None

    def _build_model(self) -> "torch.nn.Module":
        import torch.nn as nn
        layers = []
        in_dim = self.input_dim
        for _ in range(self.n_layers):
            layers += [
                nn.Linear(in_dim, self.hidden_dim, bias=False),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.n_classes))
        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        return net

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray = None,
        y_val:   np.ndarray = None,
    ) -> "SupervisedMLPBaseline":
        import torch, torch.nn as nn, torch.optim as optim, math
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_tr = self.normalizer.fit_transform(X_train)
        X_tensor = torch.tensor(X_tr, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )
        model = self._build_model()
        opt   = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce    = nn.CrossEntropyLoss()

        n_steps     = self.n_epochs * len(loader)
        warmup_steps = 10 * len(loader)

        best_val_f1, best_state = -1.0, None
        step = 0
        for epoch in range(1, self.n_epochs + 1):
            model.train()
            for Xb, yb in loader:
                # linear warmup then cosine decay
                if step < warmup_steps:
                    lr = self.lr * (step + 1) / warmup_steps
                else:
                    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
                    lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in opt.param_groups:
                    pg["lr"] = lr
                opt.zero_grad()
                loss = ce(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1

            if X_val is not None and (epoch % 10 == 0 or epoch == self.n_epochs):
                model.eval()
                with torch.no_grad():
                    X_val_n = torch.tensor(
                        self.normalizer.transform(X_val), dtype=torch.float32
                    )
                    logits = model(X_val_n)
                    preds  = logits.argmax(dim=1).numpy()
                vf1 = f1_score(y_val, preds, average="macro", zero_division=0)
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        X_n  = self.normalizer.transform(X)
        with torch.no_grad():
            logits = self._model(torch.tensor(X_n, dtype=torch.float32))
            probs  = torch.softmax(logits, dim=1)
        preds      = probs.argmax(dim=1).numpy()
        confidence = probs.max(dim=1).values.numpy()
        return preds, confidence

    def score(self, X_test: np.ndarray, y_test: np.ndarray,
              noise_std: float = 0.0, rng: np.random.Generator = None) -> dict:
        X_eval = X_test.copy()
        if noise_std > 0.0:
            if rng is None:
                rng = np.random.default_rng(0)
            X_eval = np.clip(X_eval + rng.normal(0, noise_std, X_eval.shape), 0.0, 1.0)
        preds, scores = self.predict(X_eval)
        is_correct = (preds == y_test).astype(int)
        f1   = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec  = recall_score(y_test, preds, average="macro", zero_division=0)
        try:
            auroc = roc_auc_score(is_correct, scores)
        except ValueError:
            auroc = float("nan")
        return {
            "f1": round(f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "auroc": round(auroc, 4),
        }


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N_train, N_test, D = 400, 100, 28

    X_train = rng.random((N_train, D))
    y_train = rng.integers(0, 6, N_train)
    X_val   = rng.random((N_test // 2, D))
    y_val   = rng.integers(0, 6, N_test // 2)
    X_test  = rng.random((N_test, D))
    y_test  = rng.integers(0, 6, N_test)

    bl = EngineeredBaseline()
    bl.fit(X_train, y_train)
    best_thresh = bl.tune_threshold(X_val, y_val)
    print(f"Best threshold (val): {best_thresh:.3f}")

    metrics = bl.score(X_test, y_test)
    print("Test metrics:", metrics)
