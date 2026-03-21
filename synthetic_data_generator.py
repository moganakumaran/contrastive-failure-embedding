"""
synthetic_data_generator.py
───────────────────────────
Generates the 60,000-record synthetic failure dataset described in:

  "Contrastive Failure Embeddings for Automated Recurrence Detection
   in Data Lakehouse Platforms"

Each record is a 28-dimensional percentile-relative feature vector.
Feature distributions are calibrated against published cloud workload
traces (Reiss et al., SoCC 2012; Di et al., IEEE CLUSTER 2012).

Usage
─────
  python synthetic_data_generator.py --output_dir data/ --seed 42

Outputs
───────
  data/train.csv          42 000 records
  data/val.csv             9 000 records
  data/test.csv            9 000 records  (includes hard negatives)
  data/full_dataset.csv   60 000 records  (unsplit)
  data/generation_report.txt
"""

import argparse
import os
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Feature names ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # Group A: Resource utilization (8)
    "peak_executor_memory",
    "mean_executor_memory",
    "peak_cpu_utilization",
    "mean_cpu_utilization",
    "total_input_bytes",
    "total_output_bytes",
    "total_records_read",
    "total_records_written",
    # Group B: Shuffle and spill (12)
    "shuffle_read_bytes",
    "shuffle_write_bytes",
    "spill_to_disk_bytes",
    "total_partitions",
    "max_partition_bytes",
    "min_partition_bytes",
    "partition_size_std",
    "shuffle_fetch_wait_ms",
    "sort_buffer_utilization",
    "executor_disk_io",
    "network_bytes_transmitted",
    "task_duration_variance",
    # Group C: Error diagnostics (8)
    "error_type_code",
    "affected_subsystem_code",
    "failure_timing_fraction",
    "retry_attempt_count",
    "executor_count_at_failure",
    "stage_id_at_failure",
    "failed_task_count",
    "error_severity_code",
]

CLASS_NAMES = [
    "data_skew",
    "memory_saturation",
    "shuffle_spill",
    "spot_interruption",
    "data_format_cast_error",
    "api_rate_limiting",
]

N_FEATURES = 28
N_CLASSES  = 6


# ── Per-class distribution parameters ────────────────────────────────────────
# All values are percentile-relative scores in [0, 1].
# 'mean'  : class centroid in feature space
# 'std'   : per-feature standard deviation
# 'corr'  : list of (i, j, rho) off-diagonal correlations to inject
# 'error_type_dist'  : probability over error_type_code values [0..5]
# 'subsystem_dist'   : probability over affected_subsystem_code values [0..4]
# 'severity_dist'    : probability over error_severity_code values [0..3]

@dataclass
class ClassParams:
    name: str
    mean: np.ndarray              # shape (28,)
    std:  np.ndarray              # shape (28,)
    corr: List[Tuple[int,int,float]] = field(default_factory=list)
    error_type_dist:  np.ndarray = None   # shape (6,)  for feature 20
    subsystem_dist:   np.ndarray = None   # shape (5,)  for feature 21
    severity_dist:    np.ndarray = None   # shape (4,)  for feature 27


def _make_params() -> List[ClassParams]:
    """
    Build per-class distribution parameters.

    Design philosophy (Option A — realistic overlap)
    ─────────────────────────────────────────────────
    Each class has 4–6 "key" discriminating features with distinctive centroids
    and tight std (0.06–0.08).  All remaining features share a neutral centroid
    (0.50) with high std (0.15), adding 22+ dimensions of noise.

    This intentionally confuses the engineered baseline (equal-weight cosine
    similarity across all 28 features) while still being learnable by the
    SupCon encoder, which discovers the key feature subspace.

    Designed class confusions (realistic, but challenging for baselines)
    ────────────────────────────────────────────────────────────────────
    • Data Skew ↔ Shuffle Spill:   both show high shuffle_read [8]; differ on
      partition_size_std [14] (very high for Skew, moderate for Spill) and
      sort_buffer_utilization [16] (very high for Spill, moderate for Skew).
    • Spot Interruption ↔ API Rate Limiting:  both show elevated retry [23];
      differ on shuffle_fetch_wait_ms [15] (very high for Rate Limiting, low
      for Spot) and executor_count_at_failure [24] (very low for Spot, high
      for Rate Limiting).
    • Memory Saturation ↔ Shuffle Spill (hard negatives):  primary pair for
      _inject_hard_negatives(); differ on peak_memory [0] (very high for
      Memory, moderate for Spill) and spill_to_disk_bytes [10] (very high for
      Spill, very low for Memory).

    Feature index reference
    ───────────────────────
    0  peak_executor_memory       8  shuffle_read_bytes      16 sort_buffer_utilization
    1  mean_executor_memory       9  shuffle_write_bytes     17 executor_disk_io
    2  peak_cpu_utilization      10  spill_to_disk_bytes     18 network_bytes_transmitted
    3  mean_cpu_utilization      11  total_partitions        19 task_duration_variance
    4  total_input_bytes         12  max_partition_bytes     20 error_type_code
    5  total_output_bytes        13  min_partition_bytes     21 affected_subsystem_code
    6  total_records_read        14  partition_size_std      22 failure_timing_fraction
    7  total_records_written     15  shuffle_fetch_wait_ms   23 retry_attempt_count
                                                             24 executor_count_at_failure
                                                             25 stage_id_at_failure
                                                             26 failed_task_count
                                                             27 error_severity_code
    """
    # ── Background parameters ────────────────────────────────────────────────
    # Every class shares the same background distribution for 20 non-key
    # features.  Key features (2–4 per class) deviate from the background.
    # Large background std relative to small centroid shifts creates genuine
    # inter-class confusion that a learned embedding can resolve, but an
    # equal-weight cosine baseline cannot.
    #
    # Background: mean=0.50, std=0.20
    # Key "high" centroid: 0.68  (shift = +0.18)
    # Key "low"  centroid: 0.32  (shift = −0.18)
    # Key feature std: 0.15
    #
    # Designed inter-class confusions (shared key features → hard pairs):
    #   Pair A: Data Skew (0) ↔ Shuffle Spill (2)  — share high shuffle_read [8]
    #   Pair B: Mem. Saturation (1) ↔ Shuffle Spill (2)  — share elevated memory [0]
    #                                                hard-negative pair in Exp. 5
    #   Pair C: Spot Interruption (3) ↔ API Rate Limiting (5)  — share high retry [23]

    BG_MU   = 0.50   # background centroid (all non-key features)
    BG_SIG  = 0.16   # background std — creates inter-class confusion
    KEY_SIG = 0.12   # key-feature std — tighter, but still overlapping with BG
    HI      = 0.74   # key "high" centroid  (delta = +0.24 from BG)
    LO      = 0.26   # key "low"  centroid  (delta = −0.24 from BG)

    def _base_mean():
        return np.full(N_FEATURES, BG_MU, dtype=float)

    def _base_std():
        return np.full(N_FEATURES, BG_SIG, dtype=float)

    params = []

    # ── Class 0: Data Skew ────────────────────────────────────────────────────
    # Key discriminators (distinctive for this class):
    #   [14] partition_size_std     → HI  (highly skewed partition sizes)
    #   [19] task_duration_variance → HI  (long tail from hot partitions)
    #   [12] max_partition_bytes    → HI  (huge max partition)
    #   [13] min_partition_bytes    → LO  (tiny min partition)
    # Shared confusion with Shuffle Spill (2):
    #   [8]  shuffle_read_bytes     → HI  (both classes shuffle a lot)
    m = _base_mean()
    m[8]  = HI    # shared with Shuffle Spill → confusion
    m[12] = HI    # max partition KEY
    m[13] = LO    # min partition KEY
    m[14] = HI    # partition_size_std KEY
    m[19] = HI    # task_duration_var KEY
    s = _base_std()
    s[8]  = KEY_SIG
    s[12] = KEY_SIG
    s[13] = KEY_SIG
    s[14] = KEY_SIG
    s[19] = KEY_SIG
    params.append(ClassParams(
        name="data_skew", mean=m, std=s,
        corr=[(14, 19, 0.65), (12, 14, 0.60), (8, 9, 0.70)],
        error_type_dist=np.array([0.50, 0.10, 0.18, 0.08, 0.08, 0.06]),
        subsystem_dist =np.array([0.15, 0.55, 0.15, 0.10, 0.05]),
        severity_dist  =np.array([0.08, 0.20, 0.58, 0.14]),
    ))

    # ── Class 1: Memory Saturation ────────────────────────────────────────────
    # Key discriminators:
    #   [0]  peak_executor_memory   → HI   (very high — primary key)
    #   [1]  mean_executor_memory   → HI
    #   [10] spill_to_disk_bytes    → LO   (pure OOM, no spill — distinct from Spill)
    # Shared confusion with Shuffle Spill (2) — hard-negative pair:
    #   [16] sort_buffer_utilization→ BG   (neither low nor high — overlapping)
    m = _base_mean()
    m[0]  = HI    # peak memory KEY
    m[1]  = HI    # mean memory KEY
    m[10] = LO    # spill very low (distinguishes from Shuffle Spill)
    # [16] stays at BG to create ambiguity with Spill for hard negatives
    s = _base_std()
    s[0]  = KEY_SIG
    s[1]  = KEY_SIG
    s[10] = KEY_SIG
    params.append(ClassParams(
        name="memory_saturation", mean=m, std=s,
        corr=[(0, 1, 0.82), (0, 10, -0.50)],
        error_type_dist=np.array([0.10, 0.50, 0.22, 0.07, 0.06, 0.05]),
        subsystem_dist =np.array([0.55, 0.18, 0.12, 0.10, 0.05]),
        severity_dist  =np.array([0.04, 0.10, 0.24, 0.62]),
    ))

    # ── Class 2: Shuffle Spill ────────────────────────────────────────────────
    # Key discriminators:
    #   [10] spill_to_disk_bytes    → HI   (very high — primary key)
    #   [16] sort_buffer_utilization→ HI   (sort buffer saturated)
    #   [17] executor_disk_io       → HI   (heavy disk IO from spill)
    # Shared confusions:
    #   [8]  shuffle_read_bytes     → HI   (shared with Data Skew — Pair A)
    #   [0]  peak_executor_memory   → mid  (slightly elevated, shared with MemSat — Pair B)
    m = _base_mean()
    m[0]  = 0.60   # slightly elevated (shared Memory Saturation confusion — Pair B)
    m[8]  = HI     # shared with Data Skew — Pair A confusion
    m[10] = HI     # spill KEY
    m[16] = HI     # sort_buffer KEY
    m[17] = HI     # disk_io KEY
    s = _base_std()
    s[0]  = KEY_SIG
    s[8]  = KEY_SIG
    s[10] = KEY_SIG
    s[16] = KEY_SIG
    s[17] = KEY_SIG
    params.append(ClassParams(
        name="shuffle_spill", mean=m, std=s,
        corr=[(10, 16, 0.78), (10, 17, 0.70), (8, 9, 0.72)],
        error_type_dist=np.array([0.14, 0.12, 0.50, 0.09, 0.08, 0.07]),
        subsystem_dist =np.array([0.12, 0.18, 0.55, 0.10, 0.05]),
        severity_dist  =np.array([0.06, 0.16, 0.58, 0.20]),
    ))

    # ── Class 3: Spot Instance Interruption ───────────────────────────────────
    # Key discriminators:
    #   [24] executor_count_at_fail → LO   (sudden executor drop)
    #   [22] failure_timing_fraction→ BG   (random timing — wide std)
    # Shared confusion with API Rate Limiting (5) — Pair C:
    #   [23] retry_attempt_count    → HI   (shared high retry with API RL)
    # Contrast with API RL via:
    #   [15] shuffle_fetch_wait_ms  → LO   (low — contrasts with API Rate Limiting)
    m = _base_mean()
    m[15] = LO     # fetch_wait low (contrasts with API Rate Limiting)
    m[23] = HI     # retry KEY (shared with API Rate Limiting — Pair C confusion)
    m[24] = LO     # executor_count KEY (sudden drop — distinctive)
    s = _base_std()
    s[15] = KEY_SIG
    s[23] = KEY_SIG
    s[24] = KEY_SIG
    params.append(ClassParams(
        name="spot_interruption", mean=m, std=s,
        corr=[(23, 24, -0.55)],
        error_type_dist=np.array([0.10, 0.09, 0.14, 0.50, 0.10, 0.07]),
        subsystem_dist =np.array([0.10, 0.10, 0.14, 0.58, 0.08]),
        severity_dist  =np.array([0.07, 0.20, 0.54, 0.19]),
    ))

    # ── Class 4: Schema Drift ─────────────────────────────────────────────────
    # Key discriminators (early-failure pattern):
    #   [22] failure_timing_fraction→ LO   (very early — fails in parser/planner)
    #   [25] stage_id_at_failure    → LO   (first stage)
    #   [4]  total_input_bytes      → LO   (minimal — barely started reading)
    #   [8]  shuffle_read_bytes     → LO   (near zero — fails before shuffle)
    # Resource features [0–7] also somewhat lower (early exit before resources peak)
    m = _base_mean()
    m[4]  = LO     # input_bytes KEY (early exit)
    m[8]  = LO     # shuffle_read near-zero KEY
    m[22] = LO     # failure_timing KEY (very early)
    m[25] = LO     # stage_id KEY (first stage)
    # Additional weak signal from resource group
    m[0:4] = 0.42  # slightly lower resource features (early exit, overwritten by key where needed)
    s = _base_std()
    s[4]  = KEY_SIG
    s[8]  = KEY_SIG
    s[22] = KEY_SIG
    s[25] = KEY_SIG
    s[0:4] = BG_SIG   # resource features use background std
    params.append(ClassParams(
        name="schema_drift", mean=m, std=s,
        corr=[(22, 25, 0.80), (22, 4, 0.62)],
        error_type_dist=np.array([0.08, 0.06, 0.08, 0.07, 0.52, 0.19]),
        subsystem_dist =np.array([0.07, 0.08, 0.09, 0.10, 0.66]),
        severity_dist  =np.array([0.12, 0.56, 0.25, 0.07]),
    ))

    # ── Class 5: API Rate Limiting ────────────────────────────────────────────
    # Key discriminators:
    #   [15] shuffle_fetch_wait_ms  → HI   (very high — primary key)
    #   [24] executor_count_at_fail → HI   (executors alive, just throttled)
    # Shared confusion with Spot Interruption (3) — Pair C:
    #   [23] retry_attempt_count    → HI   (shared high retry with Spot)
    # Contrast with Spot Interruption via:
    #   [15] fetch_wait → HI (vs LO for Spot)
    #   [24] executor_count → HI (vs LO for Spot)
    m = _base_mean()
    m[15] = HI     # fetch_wait KEY (very high — distinguishes from Spot)
    m[23] = HI     # retry KEY (shared with Spot Interruption — Pair C confusion)
    m[24] = HI     # executor_count high (alive, just throttled — contrasts Spot)
    s = _base_std()
    s[15] = KEY_SIG
    s[23] = KEY_SIG
    s[24] = KEY_SIG
    params.append(ClassParams(
        name="api_rate_limiting", mean=m, std=s,
        corr=[(15, 23, 0.75)],
        error_type_dist=np.array([0.08, 0.07, 0.10, 0.10, 0.10, 0.55]),
        subsystem_dist =np.array([0.10, 0.14, 0.58, 0.12, 0.06]),
        severity_dist  =np.array([0.14, 0.56, 0.24, 0.06]),
    ))

    return params


# ── Sampling ──────────────────────────────────────────────────────────────────

def _apply_correlations(X: np.ndarray, corr: List[Tuple[int, int, float]],
                        rng: np.random.Generator) -> np.ndarray:
    """Inject pairwise feature correlations via Cholesky perturbation."""
    X = X.copy()
    for i, j, rho in corr:
        noise = rho * X[:, i] + np.sqrt(1 - rho**2) * rng.standard_normal(len(X))
        # Blend: replace column j so it correlates with column i
        X[:, j] = 0.5 * X[:, j] + 0.5 * noise
    return X


def _sample_class(params: ClassParams, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Sample n records from a class distribution."""
    X = rng.normal(loc=params.mean, scale=params.std, size=(n, N_FEATURES))
    X = _apply_correlations(X, params.corr, rng)

    # Clip continuous features to [0, 1]
    X = np.clip(X, 0.0, 1.0)

    # Sample categorical features from class-specific discrete distributions
    X[:, 20] = rng.choice(6, size=n, p=params.error_type_dist)   / 5.0
    X[:, 21] = rng.choice(5, size=n, p=params.subsystem_dist)    / 4.0
    X[:, 27] = rng.choice(4, size=n, p=params.severity_dist)     / 3.0

    return X


def _inject_hard_negatives(X_test: np.ndarray, y_test: np.ndarray,
                           fraction: float,
                           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct hard negatives in the test set.

    For each selected hard-negative pair, two records from different classes
    are assigned the same error_type_code. The pair (Memory Saturation,
    Shuffle Spill) is primary; (Data Skew, Shuffle Spill) is secondary.

    Returns:
        X_test_out    : updated feature matrix
        y_test_out    : unchanged labels
        is_hard_neg   : boolean mask, True for records in a hard-neg pair
    """
    n_test = len(X_test)
    n_hn   = int(n_test * fraction)   # number of hard-negative records

    X_out    = X_test.copy()
    is_hn    = np.zeros(n_test, dtype=bool)

    # Pair 1: Memory Saturation (1) ↔ Shuffle Spill (2)
    # Both get error_type_code = 2/5 (executor/shuffle)
    idx_ms = np.where(y_test == 1)[0]
    idx_ss = np.where(y_test == 2)[0]
    n_pair1 = n_hn // 2

    sel_ms = rng.choice(idx_ms, size=min(n_pair1, len(idx_ms)), replace=False)
    sel_ss = rng.choice(idx_ss, size=min(n_pair1, len(idx_ss)), replace=False)
    shared_code = 2.0 / 5.0   # error_type_code = 2, normalised to [0,1]
    X_out[sel_ms, 20] = shared_code
    X_out[sel_ss, 20] = shared_code
    is_hn[sel_ms] = True
    is_hn[sel_ss] = True

    # Pair 2: Data Skew (0) ↔ Shuffle Spill (2) — remaining quota
    remaining = n_hn - 2 * n_pair1
    if remaining > 0:
        idx_ds = np.where(y_test == 0)[0]
        idx_ss2 = np.setdiff1d(idx_ss, sel_ss)
        n_pair2 = remaining // 2
        sel_ds  = rng.choice(idx_ds,  size=min(n_pair2, len(idx_ds)),  replace=False)
        sel_ss2 = rng.choice(idx_ss2, size=min(n_pair2, len(idx_ss2)), replace=False)
        shared_code2 = 0.0   # error_type_code = 0 (skew/partition)
        X_out[sel_ds,  20] = shared_code2
        X_out[sel_ss2, 20] = shared_code2
        is_hn[sel_ds]  = True
        is_hn[sel_ss2] = True

    return X_out, y_test, is_hn


# ── Normalization ─────────────────────────────────────────────────────────────

class PercentileNormalizer:
    """
    Percentile-relative normalization.

    In production, percentiles are computed from a rolling window of
    historical executions of the same job. Here they are estimated from
    the training set, simulating that historical context.

    fit(X_train) -> computes per-feature 5th and 95th percentiles.
    transform(X) -> maps each feature to [0, 1] using the fitted percentiles.
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
        denom = np.where(self.p_hi - self.p_lo > 1e-9,
                         self.p_hi - self.p_lo, 1.0)
        return np.clip((X - self.p_lo) / denom, 0.0, 1.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ── OOD scale variant ─────────────────────────────────────────────────────────

def _apply_ood_scale(X: np.ndarray, scale_range: Tuple[float, float],
                     rng: np.random.Generator) -> np.ndarray:
    """
    Multiply resource and shuffle features (indices 0–19) by a random
    scale factor drawn from scale_range to simulate a different workload
    size. Categorical / diagnostic features (20–27) are unchanged.
    """
    scale = rng.uniform(scale_range[0], scale_range[1], size=len(X))
    X_ood = X.copy()
    X_ood[:, :20] = np.clip(X[:, :20] * scale[:, None], 0.0, 1.0)
    return X_ood


# ── Dataset assembly ──────────────────────────────────────────────────────────

def generate_dataset(
    n_per_class:  int   = 10_000,
    hard_neg_frac: float = 0.20,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    ood_scale_train: Tuple[float, float] = (1.0, 3.5),
    ood_scale_test:  Tuple[float, float] = (3.0, 5.5),
) -> Dict:
    """
    Generate and split the full synthetic failure dataset.

    Returns a dict with keys:
        X_train, y_train
        X_val,   y_val
        X_test,  y_test,  is_hard_neg_test
        X_ood_train, y_ood_train   (scaled for OOD experiment training)
        X_ood_test,  y_ood_test    (scaled for OOD experiment evaluation)
        normalizer                 (fitted PercentileNormalizer)
        feature_names
        class_names
    """
    rng    = np.random.default_rng(seed)
    params = _make_params()

    # ── Sample raw features ───────────────────────────────────────────────────
    X_list, y_list = [], []
    for cls_idx, p in enumerate(params):
        X_cls = _sample_class(p, n_per_class, rng)
        X_list.append(X_cls)
        y_list.append(np.full(n_per_class, cls_idx, dtype=np.int64))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    # ── Stratified split ──────────────────────────────────────────────────────
    train_frac, val_frac, test_frac = split_ratios
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all,
        test_size=test_frac, random_state=seed, stratify=y_all,
    )
    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac_of_trainval, random_state=seed, stratify=y_train_val,
    )

    # ── Hard negatives (test set only) ────────────────────────────────────────
    X_test, y_test, is_hard_neg = _inject_hard_negatives(
        X_test, y_test, hard_neg_frac, rng
    )

    # ── OOD variants (Experiment 4) ───────────────────────────────────────────
    # Training split: scale resource features by train range
    X_ood_train = _apply_ood_scale(X_train, ood_scale_train, rng)
    # Test split: scale by held-out large-cluster range
    X_ood_test  = _apply_ood_scale(X_test,  ood_scale_test,  rng)

    # ── Percentile-relative normalization ─────────────────────────────────────
    # Main splits: normalizer fitted on raw (small-cluster) training data
    normalizer = PercentileNormalizer()
    X_train_n = normalizer.fit_transform(X_train)
    X_val_n   = normalizer.transform(X_val)
    X_test_n  = normalizer.transform(X_test)

    # OOD splits: normalize with the TRAINING normalizer (not re-fitted).
    # This deliberately exposes the saturation effect: large-cluster resource
    # feature values exceed the training-set 95th percentile, causing 68.4% of
    # OOD features to saturate at the normalized ceiling (1.0). This is the
    # mechanism behind SupCon's OOD F1 collapse to 0.5926 (Section 3.4).
    X_ood_train_n = normalizer.transform(X_ood_train)
    X_ood_test_n  = normalizer.transform(X_ood_test)

    return {
        "X_train": X_train_n, "y_train": y_train,
        "X_val":   X_val_n,   "y_val":   y_val,
        "X_test":  X_test_n,  "y_test":  y_test,
        "is_hard_neg_test": is_hard_neg,
        "X_ood_train": X_ood_train_n, "y_ood_train": y_train,
        "X_ood_test":  X_ood_test_n,  "y_ood_test":  y_test,
        "normalizer": normalizer,
        "feature_names": FEATURE_NAMES,
        "class_names":   CLASS_NAMES,
    }


# ── Save to CSV ───────────────────────────────────────────────────────────────

def _to_dataframe(X: np.ndarray, y: np.ndarray,
                  is_hard_neg: np.ndarray = None) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df["is_hard_negative"] = (is_hard_neg.astype(int)
                               if is_hard_neg is not None
                               else 0)
    return df


def save_splits(dataset: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "train": (_to_dataframe(dataset["X_train"], dataset["y_train"]),
                  "train.csv"),
        "val":   (_to_dataframe(dataset["X_val"],   dataset["y_val"]),
                  "val.csv"),
        "test":  (_to_dataframe(dataset["X_test"],  dataset["y_test"],
                                dataset["is_hard_neg_test"]),
                  "test.csv"),
    }

    for split_name, (df, fname) in splits.items():
        path = os.path.join(output_dir, fname)
        df.to_csv(path, index=False)
        print(f"  Saved {split_name:5s}: {len(df):>6,} records → {path}")

    # Full unsplit dataset
    df_full = pd.concat([
        splits["train"][0], splits["val"][0], splits["test"][0]
    ], ignore_index=True)
    full_path = os.path.join(output_dir, "full_dataset.csv")
    df_full.to_csv(full_path, index=False)
    print(f"  Saved full   : {len(df_full):>6,} records → {full_path}")

    # OOD splits
    df_ood_train = _to_dataframe(dataset["X_ood_train"], dataset["y_ood_train"])
    df_ood_test  = _to_dataframe(dataset["X_ood_test"],  dataset["y_ood_test"],
                                  dataset["is_hard_neg_test"])
    df_ood_train.to_csv(os.path.join(output_dir, "ood_train.csv"), index=False)
    df_ood_test.to_csv( os.path.join(output_dir, "ood_test.csv"),  index=False)
    print(f"  Saved ood_train / ood_test → {output_dir}")


def _write_report(dataset: Dict, output_dir: str) -> None:
    """Write a short generation report."""
    report_lines = [
        "Synthetic Dataset Generation Report",
        "=" * 40,
        f"Total records : {sum(len(dataset[k]) for k in ['X_train','X_val','X_test']):,}",
        f"Train         : {len(dataset['X_train']):,}",
        f"Val           : {len(dataset['X_val']):,}",
        f"Test          : {len(dataset['X_test']):,}",
        f"  of which hard negatives: {dataset['is_hard_neg_test'].sum():,}",
        f"Features      : {N_FEATURES}",
        f"Classes       : {N_CLASSES}  ({', '.join(CLASS_NAMES)})",
        "",
        "Per-class record counts (test set):",
    ]
    for cls_idx, name in enumerate(CLASS_NAMES):
        count = (dataset["y_test"] == cls_idx).sum()
        hn    = (dataset["is_hard_neg_test"] & (dataset["y_test"] == cls_idx)).sum()
        report_lines.append(f"  {cls_idx} {name:<25s} {count:>5,}  ({hn} hard neg)")

    report_path = os.path.join(output_dir, "generation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  Report       → {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate the synthetic Lakehouse failure dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Example
            ───────
              python synthetic_data_generator.py --output_dir data/ --seed 42
        """),
    )
    p.add_argument("--output_dir",     default="data/",  help="Directory for CSV outputs")
    p.add_argument("--n_per_class",    type=int, default=10_000, help="Records per class")
    p.add_argument("--hard_neg_frac",  type=float, default=0.20, help="Hard negative fraction of test set")
    p.add_argument("--seed",           type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Generating dataset  seed={args.seed}  n_per_class={args.n_per_class:,}")
    dataset = generate_dataset(
        n_per_class=args.n_per_class,
        hard_neg_frac=args.hard_neg_frac,
        seed=args.seed,
    )
    save_splits(dataset, args.output_dir)
    _write_report(dataset, args.output_dir)
    print("Done.")
    return dataset


if __name__ == "__main__":
    main()
