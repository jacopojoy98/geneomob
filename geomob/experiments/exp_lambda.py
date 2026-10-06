"""The lambda sweep: how much does the time channel matter, and whose?

ST2Vec defines its learning target as a blend of a spatial and a temporal
ground truth,

    D(T_i, T_j) = lambda * D_S(T_i^s, T_j^s) + (1 - lambda) * D_T(T_i^t, T_j^t)

and fixes lambda = 0.5. Sweeping it instead turns a single number into a
curve, and the curve is the experiment: at lambda = 1 the target is purely
spatial and the time channel is irrelevant; at lambda = 0 it is purely
temporal and the spatial channels are. GPE cannot participate in the
right-hand end at all -- it encodes (lon, lat) and nothing else -- and that
absence is the result rather than an omission.

Unlike the rest of this repo, training here is **supervised**: ST2Vec
regresses the embedding distance onto the exact measure with the triplet loss
of its Eq. (12), and HR@10 then asks how faithfully the fast learned model
reproduces the slow exact one. Self-supervised contrastive training answers a
different question and is not comparable, so it is not offered here.

The competitors for the time channel are:

  none      no temporal information at all (the GPE position)
  raw       the timestamp as a scalar, standardised
  time2vec  ST2Vec Eq. (3): t'[i] = w_i t + p_i for i = 0, cos(w_i t + p_i)
            otherwise, with **learnable** w and p. This is Time2Vec (Kazemi
            et al.) and is the fair opponent: sinusoidal, multi-frequency,
            and free to discover any period it likes.
  cyclic    the C24 x C7 GENEO: fixed harmonics, exact shift equivariance,
            no linear term.

Predictions worth registering before running: the cyclic code should hold its
advantage as lambda falls, the gap should be largest at small training sizes,
and only `time2vec` should drift under a global shift of every timestamp,
because only it carries a non-periodic linear term.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from ..config import save_results
from ..data import load_dataset
from ..graph import (build_node_graph, ground_truth_matrix, hit_ratio_at_k,
                     temporal_ground_truth)
from ..models import TrajEncoder, pad_batch, set_seed

TIME_KINDS = ("none", "raw", "time2vec", "cyclic")


# ==========================================================================
# time embeddings
# ==========================================================================
class TimeEmbedding(nn.Module):
    """One interface, four temporal representations.

    `cyclic` is fixed and exactly equivariant under a shift of the clock;
    `time2vec` is learnable and carries a linear term, which is what lets it
    model non-periodic drift and also what breaks shift equivariance.
    """

    def __init__(self, kind: str = "cyclic", q: int = 8,
                 n_hour_harm: int = 4, n_dow_harm: int = 3):
        super().__init__()
        self.kind = kind
        if kind == "none":
            self.dim = 0
        elif kind == "raw":
            self.dim = 1
        elif kind == "time2vec":
            self.dim = q + 1
            self.w = nn.Parameter(torch.randn(q + 1) * 0.1)
            self.p = nn.Parameter(torch.zeros(q + 1))
        elif kind == "cyclic":
            self.register_buffer("mh", torch.arange(1, n_hour_harm + 1).float())
            self.register_buffer("md", torch.arange(1, n_dow_harm + 1).float())
            self.dim = 2 * (n_hour_harm + n_dow_harm)
        else:
            raise ValueError(kind)

    def forward(self, t):
        """t: (B, L) hours since the epoch. Returns (B, L, dim)."""
        if self.kind == "none":
            return t[..., :0]
        if self.kind == "raw":
            return ((t - t.mean()) / (t.std() + 1e-6)).unsqueeze(-1)
        if self.kind == "time2vec":
            a = t.unsqueeze(-1) * self.w + self.p
            # element 0 stays linear; the rest are periodic (ST2Vec Eq. 3)
            return torch.cat([a[..., :1], torch.cos(a[..., 1:])], dim=-1)
        h = (t % 24.0).unsqueeze(-1) * self.mh * (2 * np.pi / 24.0)
        d = ((t // 24.0) % 7.0).unsqueeze(-1) * self.md * (2 * np.pi / 7.0)
        return torch.cat([torch.sin(h), torch.cos(h),
                          torch.sin(d), torch.cos(d)], dim=-1)


class STModel(nn.Module):
    """Pre-computed spatial tokens, plus a time embedding built inside the
    model so that `time2vec`'s parameters can actually be learned."""

    def __init__(self, spatial_dim: int, time_kind: str, hidden: int = 128,
                 out_dim: int = 128, kind: str = "lstm"):
        super().__init__()
        self.temb = TimeEmbedding(time_kind)
        self.enc = TrajEncoder(spatial_dim + self.temb.dim, hidden=hidden,
                               out_dim=out_dim, kind=kind)

    def forward(self, x, t, mask=None):
        e = self.temb(t)
        return self.enc(torch.cat([x, e], dim=-1) if e.shape[-1] else x, mask)


# ==========================================================================
# ground truth
# ==========================================================================
def _norm01(M):
    """Scale a dissimilarity matrix to [0, 1] using its off-diagonal mass, so
    the spatial and temporal terms are commensurable before blending."""
    off = M[~np.eye(len(M), dtype=bool)]
    lo, hi = float(off.min()), float(np.percentile(off, 99))
    return np.clip((M - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def build_ground_truth(trajs, measure="TP", cell_m=200.0, graph=None,
                       verbose=True, temporal_mode="absolute"):
    """Return (D_S, D_T), each normalised to [0, 1] and ready to blend."""
    g = graph or build_node_graph(trajs, cell_m=cell_m, verbose=verbose)
    D = g.distance_matrix()
    seqs, times = [], []
    for tr in trajs:
        s = g.snap(tr.lonlat)
        keep = np.r_[True, np.diff(s) != 0]
        seqs.append(s[keep])
        times.append(tr.t[keep])
    if verbose:
        print(f"  computing {measure} ground truth over {len(seqs)} trajectories")
    return (_norm01(ground_truth_matrix(D, seqs, measure, verbose)),
            _norm01(temporal_ground_truth(times, measure, verbose,
                                          mode=temporal_mode)),
            g)


# ==========================================================================
# training (ST2Vec Eq. 12)
# ==========================================================================
def train_regressor(model, X, T, GT, epochs=8, bs=64, lr=1e-3, alpha=4.0,
                    device="cpu", seed=0):
    """Regress exp(-||v_a - v_b||) onto exp(-alpha * D(T_a, T_b)).

    Triplets are (anchor, a near neighbour, a far one) under the ground truth,
    which is ST2Vec's sampling strategy; the loss weights each term by the
    ground-truth similarity so close pairs dominate.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(X)
    order = np.argsort(GT, axis=1)
    near, far = order[:, 1:1 + max(n // 10, 2)], order[:, -max(n // 10, 2):]

    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n - bs + 1, bs):
            a = perm[s:s + bs]
            p = near[a, rng.integers(0, near.shape[1], len(a))]
            q = far[a, rng.integers(0, far.shape[1], len(a))]
            va = model(*_batch(X, T, a, device))
            vp = model(*_batch(X, T, p, device))
            vq = model(*_batch(X, T, q, device))
            dp = torch.exp(-torch.norm(va - vp, dim=1))
            dq = torch.exp(-torch.norm(va - vq, dim=1))
            gp = torch.from_numpy(np.exp(-alpha * GT[a, p])).float().to(device)
            gq = torch.from_numpy(np.exp(-alpha * GT[a, q])).float().to(device)
            loss = (gp * (gp - dp) ** 2 + gq * (gq - dq) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def _batch(X, T, idx, device):
    x, m = pad_batch([X[i] for i in idx], device)
    L = x.shape[1]
    t = np.zeros((len(idx), L), dtype=np.float32)
    for r, i in enumerate(idx):
        t[r, :len(T[i])] = T[i]
    return x, torch.from_numpy(t).to(device), m


@torch.no_grad()
def embed_all(model, X, T, device="cpu", bs=128):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        idx = np.arange(s, min(s + bs, len(X)))
        x, t, m = _batch(X, T, idx, device)
        out.append(model(x, t, m).cpu().numpy())
    model.train()
    return np.concatenate(out)


# ==========================================================================
# experiment
# ==========================================================================
def run(dataset="synthetic_hard", path=None, measure="TP",
        lambdas=(1.0, 0.75, 0.5, 0.25, 0.0), time_kinds=TIME_KINDS,
        n_train=400, n_test=200, cell_m=200.0, epochs=8, k=10,
        temporal_mode="cyclic", shift_test_h=168.0, device="cpu", seed=0,
        split="user", out=None, **load_kw):
    """Sweep lambda and report HR@k for each time embedding.

    `shift_test_h` re-evaluates the trained models on a test set whose
    timestamps have all been shifted. The default of 168 hours is one week --
    a **period of C24 x C7** -- which is what makes the test sharp: an exactly
    periodic code is unchanged to floating point, while any code carrying a
    non-periodic linear term drifts. Shifting by a non-period (say 5 hours)
    tests nothing, since every honest time code should react to a real change
    of hour.

    `temporal_mode` selects the temporal ground truth: `absolute` for
    ST2Vec's |t_i - t_j|, `cyclic` for circular time-of-day plus day-of-week.
    See `graph._time_cost` -- they reward different structure and both are
    worth reporting.
    """
    from .exp_similarity import Tokenizer, _split
    from ..encoders import DisplacementGEO, TorusLattice

    if dataset.startswith("synthetic") and "n" not in load_kw:
        load_kw["n"] = n_train + n_test
    trajs = load_dataset(dataset, path, **load_kw)
    train, test = _split(trajs, dataset, split, n_train, n_test, seed)
    print(f"{len(train)} train / {len(test)} test trajectories")

    ref = np.concatenate([t.lonlat for t in train]).mean(0)
    tok = Tokenizer(loc=TorusLattice(ref_deg=ref), disp=DisplacementGEO(),
                    name="GEO spatial")
    tok.fit(train)

    print(f"ground truth (train), temporal mode = {temporal_mode}")
    DS_tr, DT_tr, g = build_ground_truth(train, measure, cell_m,
                                         temporal_mode=temporal_mode)
    print("ground truth (test)")
    DS_te, DT_te, _ = build_ground_truth(test, measure, cell_m, graph=g,
                                         temporal_mode=temporal_mode)

    X_tr = [tok(t) for t in train]
    X_te = [tok(t) for t in test]
    T_tr = [t.t.astype(np.float32) for t in train]
    T_te = [t.t.astype(np.float32) for t in test]
    T_te_shift = ([t + shift_test_h for t in T_te]
                  if shift_test_h is not None else None)

    res = {"dataset": dataset, "measure": measure, "k": k,
           "lambdas": list(lambdas), "time_kinds": list(time_kinds),
           "n_train": len(train), "n_test": len(test),
           "temporal_mode": temporal_mode,
           "shift_test_h": shift_test_h, "grid": {}}

    for lam in lambdas:
        GT_tr = lam * DS_tr + (1 - lam) * DT_tr
        GT_te = lam * DS_te + (1 - lam) * DT_te
        row = {}
        for kind in time_kinds:
            set_seed(seed)
            model = STModel(tok.dim, kind).to(device)
            train_regressor(model, X_tr, T_tr, GT_tr, epochs=epochs,
                            device=device, seed=seed)
            emb = embed_all(model, X_te, T_te, device)
            sim = -_pdist(emb)
            cell = {f"HR@{k}": hit_ratio_at_k(sim, GT_te, k)}
            if T_te_shift is not None:
                e2 = embed_all(model, X_te, T_te_shift, device)
                cell[f"HR@{k} shifted"] = hit_ratio_at_k(-_pdist(e2), GT_te, k)
                cell["shift_drop"] = cell[f"HR@{k}"] - cell[f"HR@{k} shifted"]
                cell["embedding_drift"] = float(
                    np.mean(np.linalg.norm(e2 - emb, axis=1))
                    / (np.mean(np.linalg.norm(emb, axis=1)) + 1e-9))
            row[kind] = cell
            print(f"  lambda={lam:.2f}  {kind:9s} " + "  ".join(
                f"{a}={b:.3f}" for a, b in cell.items()))
        res["grid"][str(lam)] = row

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


def _pdist(e):
    d = np.sqrt(np.maximum((e ** 2).sum(1)[:, None] - 2 * e @ e.T
                           + (e ** 2).sum(1)[None, :], 0))
    return d


if __name__ == "__main__":
    run()
