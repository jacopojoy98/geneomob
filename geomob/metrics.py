"""Evaluation metrics.

`similarity_metrics` implements the self-similarity protocol GPE inherits from
t2vec (their Section 5.1): split each test trajectory into odd- and
even-indexed halves, embed both, and rank each odd-half against every
even-half. Reproducing MR / MRR / MP / KP@10 on Porto, T-drive, GeoLife, Roma
and AIS makes the numbers here directly comparable to their Tables 10-13.

`probe_*` implements Experiment F: how much of a mobility statistic survives
in the tokens.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# trajectory similarity (GPE / t2vec protocol)
# --------------------------------------------------------------------------
def similarity_metrics(emb_a: np.ndarray, emb_b: np.ndarray, k: int = 10,
                       emb_full: np.ndarray | None = None) -> dict:
    """emb_a[i] and emb_b[i] are the two halves of test trajectory i.

    MR   mean rank of the true partner (lower better, 1 = perfect)
    MRR  mean reciprocal rank
    MP   mean precision = fraction ranked first
    KP   k-nn precision: overlap of the k-nn of T_i in D with those of the
         half-rate copy (needs emb_full, the embeddings of the whole test
         trajectories; omitted if not supplied)
    """
    A = _norm(emb_a)
    B = _norm(emb_b)
    S = A @ B.T
    n = len(A)
    true = S[np.arange(n), np.arange(n)]
    ranks = 1 + (S > true[:, None]).sum(axis=1)
    out = {"MR": float(ranks.mean()),
           "MRR": float((1.0 / ranks).mean()),
           "MP": float((ranks == 1).mean())}
    if emb_full is not None:
        out[f"KP@{k}"] = _knn_precision(emb_full, emb_a, emb_b, k)
    return out


def _knn_precision(emb_full, emb_a, emb_b, k):
    Sf = _norm(emb_full) @ _norm(emb_full).T
    Sh = _norm(emb_a) @ _norm(emb_b).T
    np.fill_diagonal(Sf, -np.inf)
    eta = np.argsort(-Sf, axis=1)[:, :k]
    eta_p = np.argsort(-Sh, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(eta, eta_p)]))


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------
# Experiment F: mobility statistics and linear probes
# --------------------------------------------------------------------------
def mobility_stats(xy: np.ndarray, stay_radius_m: float = 100.0,
                   stay_min_steps: int = 3) -> dict:
    """Established statistics, chosen so that some depend on information the
    construction *deliberately* discards and some do not.

    Predicted pattern (the sharp version of Hypothesis 4):
      * centroid_dist and radius_of_gyration depend on which torus wrap a
        point falls in -> a measurable probing gap is expected;
      * total_distance and mean_step depend only on the invariant radial
        channel -> no gap expected;
      * net_heading depends on the angular channel -> recoverable from the
        GEO tokens but not from an invariant-only ablation.
    """
    d = np.diff(xy, axis=0)
    r = np.linalg.norm(d, axis=1)
    c = xy.mean(axis=0)
    dev = xy - c
    # entropy of visited 200 m cells
    cells = np.floor(xy / 200.0).astype(int)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    p = counts / counts.sum()
    # stay points: consecutive runs inside a small radius
    stays, i = 0, 0
    while i < len(xy):
        j = i
        while j + 1 < len(xy) and np.linalg.norm(xy[j + 1] - xy[i]) < stay_radius_m:
            j += 1
        if j - i + 1 >= stay_min_steps:
            stays += 1
        i = max(j, i) + 1
    net = xy[-1] - xy[0]
    return {
        "radius_of_gyration": float(np.sqrt((dev ** 2).sum(1).mean())),
        "entropy": float(-(p * np.log(p + 1e-12)).sum()),
        "total_distance": float(r.sum()),
        "mean_step": float(r.mean()),
        "stay_points": float(stays),
        "centroid_dist": float(np.linalg.norm(c)),
        "net_heading_cos": float(net[0] / (np.linalg.norm(net) + 1e-9)),
        "net_heading_sin": float(net[1] / (np.linalg.norm(net) + 1e-9)),
    }


def ridge_probe(X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
                test_frac: float = 0.3, seed: int = 0,
                min_cv: float = 1e-3):
    """Held-out R^2 of a ridge probe, or None if the target is degenerate.

    Probe capacity is fixed across representations, so the comparison is about
    the representation. A target with near-zero relative variance (e.g. stay
    points on synthetic data that contains no stays) is unpredictable in the
    trivial sense and would report a meaningless R^2 of ~1 for everything, so
    it is reported as None instead.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.std() / (np.abs(y).mean() + 1e-12) < min_cv:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int((1 - test_frac) * len(idx))
    tr, te = idx[:k], idx[k:]
    Xtr, Xte = X[tr], X[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]
    Xte = np.c_[Xte, np.ones(len(Xte))]
    ym, ys = y[tr].mean(), y[tr].std() + 1e-9
    w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]),
                        Xtr.T @ ((y[tr] - ym) / ys))
    pred = Xte @ w
    true = (y[te] - ym) / ys
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum() + 1e-12
    return float(1 - ss_res / ss_tot)


def pool_tokens(tokens: np.ndarray) -> np.ndarray:
    """Mean+std pooling: the cheapest fixed-size summary of a token sequence."""
    return np.concatenate([tokens.mean(0), tokens.std(0)])
