"""Experiment A -- equivariance and sample efficiency on a real downstream task.

Three changes from the earlier synthetic version, all of them about making the
comparison one a reviewer will accept:

1. **The baselines are the tokenizers compared everywhere else** (GPE, Grid,
   XY, TriW, Space2Vec), not an unrelated GRU on raw displacements. Backbone,
   head, optimiser and schedule are identical across rows; only the tokenizer
   changes. A separate row adds the equivariant aggregator on top of the GEO
   tokens, so the contribution of the tokenizer and the contribution of the
   equivariant head can be read apart.

2. **A real task on real trajectories.** Destination prediction: given the
   first `prefix` fraction of a trip, predict where it ends. This is one of
   UniTE's standard downstream adaptors, so it connects directly to Experiment
   B rather than being a bespoke toy.

3. **Two targets with different symmetry types**, which turns the equivariance
   measurement into a sharper instrument:

     `destination` -- the remaining displacement from the last observed point.
                      Rotating the trip must rotate the answer, so the test is
                      pred(R.T) = R pred(T), i.e. rho = R.
     `eta`         -- the remaining travel time. Rotating the trip must NOT
                      change the answer, so the test is pred(R.T) = pred(T),
                      i.e. rho = identity. Equivariance and invariance are the
                      same property with different rho, and a construction
                      that only ever gets tested on one of them is
                      under-tested.

Note what the destination task does to the absolute-position baselines. It is
naturally expressed as a displacement, so it does not require knowing where on
Earth the trip is -- which is exactly the asymmetry the equivariant channel is
designed to exploit. Report `--task eta` alongside it: ETA depends on
absolute position through traffic and geography, so it is the task where the
absolute codes should do well, and a construction that wins both is more
convincing than one that wins the favourable one.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import torch
import torch.nn as nn

from ..config import save_results
from ..data import load_dataset
from ..equivariance import _rel
from ..geo import enu_to_lonlat, lonlat_to_enu
from ..models import EquivariantAggregator, TrajEncoder, pad_batch, set_seed

TASKS = ("destination", "eta")


# --------------------------------------------------------------------------
# task construction
# --------------------------------------------------------------------------
def build_task(trajs, task: str = "destination", prefix: float = 0.7,
               min_points: int = 12):
    """Split each trip into an observed prefix and a target.

    Everything is computed in a per-trip local metric frame anchored at the
    last observed point, so the target is a displacement in metres and the
    group action on it is unambiguous.
    """
    items = []
    for tr in trajs:
        n = len(tr)
        k = int(prefix * n)
        if k < min_points or n - k < 2:
            continue
        xy, ref = lonlat_to_enu(tr.lonlat)
        anchor = xy[k - 1]
        if task == "destination":
            y = xy[-1] - anchor
        elif task == "eta":
            y = np.array([(tr.t[-1] - tr.t[k - 1]) * 60.0])   # minutes
        else:
            raise ValueError(task)
        items.append({"xy": xy[:k], "ref": ref, "anchor": anchor, "y": y,
                      "t": tr.t[:k], "uid": tr.uid})
    return items


def rotate_item(item, theta: float):
    """Act on one example by a rotation about its anchor, target included."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    out = dict(item)
    out["xy"] = (item["xy"] - item["anchor"]) @ R.T + item["anchor"]
    out["y"] = item["y"] @ R.T if len(item["y"]) == 2 else item["y"]
    return out


def _to_traj(item):
    from ..data import Traj
    return Traj(enu_to_lonlat(item["xy"], item["ref"]), item["t"], item["uid"])


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class TaskModel(nn.Module):
    """Shared backbone plus a linear head. Identical for every tokenizer."""

    def __init__(self, in_dim, out_dim, hidden=128, kind="lstm"):
        super().__init__()
        self.enc = TrajEncoder(in_dim, hidden=hidden, out_dim=hidden, kind=kind)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, mask=None):
        return self.head(self.enc(x, mask))


class AggregatorModel(nn.Module):
    """GEO tokens through the equivariant aggregator of Eq. (2).

    Only valid for an equivariant 2-d target; for `eta` the invariant radial
    channel alone is used, which is the invariant counterpart of the same
    construction.
    """

    def __init__(self, n_inv, out_dim, hidden=128):
        super().__init__()
        self.equivariant = out_dim == 2
        if self.equivariant:
            self.agg = EquivariantAggregator(n_inv, hidden=hidden)
        else:
            self.gru = nn.GRU(n_inv, hidden, num_layers=2, batch_first=True,
                              bidirectional=True)
            self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, out_dim))
        self.n_inv = n_inv

    def forward(self, x, mask=None):
        if self.equivariant:
            return self.agg(x, mask)
        h, _ = self.gru(x[..., :self.n_inv])
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.out(pooled)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _tokenize(tok, items, scale):
    return [(tok(_to_traj(it)) / (scale if tok is None else 1.0)).astype(np.float32)
            for it in items]


class Scaled(nn.Module):
    """Wrap a model so it predicts in standardised units and reports metres.

    Without this the comparison is unfair in a way that has nothing to do with
    symmetry: targets of order 500 m make MSE gradients huge, every
    linear-head model parks on the constant predictor, and the equivariant
    aggregator appears to win because its output is a combination of unit
    vectors and therefore happens to start on the right scale. Standardising
    the target removes that confound, so what is left is the structure.
    """

    def __init__(self, model, scale: float):
        super().__init__()
        self.model, self.scale = model, float(scale)

    def forward(self, x, mask=None):
        return self.model(x, mask) * self.scale


def _train(model, X, Y, epochs, lr=1e-3, bs=128, device="cpu", seed=0):
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
    for _ in range(epochs):
        perm = np.random.permutation(len(X))
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            if len(b) < 2:
                continue
            x, m = pad_batch([X[i] for i in b], device)
            loss = nn.functional.mse_loss(model(x, m), Yt[b])
            opt.zero_grad()
            loss.backward()
            # Applied identically to every row. Without it the high-dimensional
            # absolute-position tokenizers occasionally diverge on small
            # training sets, and a diverged baseline is not evidence about
            # symmetry -- it is evidence that the baseline was not tuned.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def _predict(model, X, device="cpu", bs=256):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        x, m = pad_batch(X[s:s + bs], device)
        out.append(model(x, m).cpu().numpy())
    model.train()
    return np.concatenate(out)


def _error(pred, Y, task):
    if task == "destination":
        return float(np.mean(np.linalg.norm(pred - Y, axis=1)))   # metres
    return float(np.sqrt(np.mean((pred - Y) ** 2)))               # minutes


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run(dataset="synthetic", path=None, task="destination",
        which=("gpe", "geo_spatial", "grid", "xy", "fourier", "space2vec"),
        sizes=(200, 500, 1000, 2000), n_test=600, epochs=10, prefix=0.7,
        device="cpu", seed=0, out=None, split="user", n_probe=256,
        n_attrib=24, kind="lstm", **load_kw):
    from .exp_similarity import _split, build_tokenizers

    trajs = load_dataset(dataset, path, **load_kw)
    train_tr, test_tr = _split(trajs, dataset, split, max(sizes), n_test, seed)
    train_items = build_task(train_tr, task, prefix)
    test_items = build_task(test_tr, task, prefix)
    if not train_items or not test_items:
        raise ValueError("no usable examples; lower --prefix or min length")

    Ytr = np.stack([it["y"] for it in train_items])
    Yte = np.stack([it["y"] for it in test_items])
    unit = "m" if task == "destination" else "min"
    # A constant predictor is the floor any model must beat.
    baseline_err = _error(np.repeat(Ytr.mean(0)[None], len(Yte), 0), Yte, task)
    # Common output scale for every model, fitted on the training targets only.
    yscale = float(np.sqrt((Ytr ** 2).sum(1).mean())) or 1.0
    print(f"task={task}  {len(train_items)} train / {len(test_items)} test  "
          f"constant-predictor error {baseline_err:.1f} {unit}")

    toks = build_tokenizers(tuple(which))
    res = {"task": task, "dataset": dataset, "sizes": list(sizes),
           "unit": unit, "constant_predictor": baseline_err,
           "target_scale": yscale,
           "n_train": len(train_items), "n_test": len(test_items),
           "models": {}}

    for key, tok in toks.items():
        tok.fit(train_tr)
        Xtr_all = [tok(_to_traj(it)).astype(np.float32) for it in train_items]
        Xte = [tok(_to_traj(it)).astype(np.float32) for it in test_items]
        rows, model = [], None
        for n in sizes:
            n = min(n, len(Xtr_all))
            model = _train(Scaled(TaskModel(tok.dim, Yte.shape[1], kind=kind),
                                  yscale),
                           Xtr_all[:n], Ytr[:n], epochs, device=device,
                           seed=seed)
            rows.append(_error(_predict(model, Xte, device), Yte, task))
            print(f"  [{tok.name:22s}] n={n:5d}  err={rows[-1]:.1f} {unit}")
        res["models"][tok.name] = {
            "dim": tok.dim, "errors": rows,
            "equivariance_error": _model_equivariance(
                model, tok, test_items[:n_probe], task, device, seed),
            "explanation_equivariance_error": _explanation_equivariance(
                model, tok, test_items[:n_attrib], device, seed)}

    # the full construction: GEO tokens + equivariant aggregator
    if "geo_spatial" in toks:
        tok = toks["geo_spatial"]
        n_inv = tok.disp.n_inv if tok.disp is not None else tok.dim
        rows, model = [], None
        Xtr_all = [_agg_tokens(tok, it) for it in train_items]
        Xte = [_agg_tokens(tok, it) for it in test_items]
        for n in sizes:
            n = min(n, len(Xtr_all))
            model = _train(Scaled(AggregatorModel(n_inv, Yte.shape[1]), yscale),
                           Xtr_all[:n], Ytr[:n], epochs, device=device,
                           seed=seed)
            rows.append(_error(_predict(model, Xte, device), Yte, task))
            print(f"  [GEO + equiv. aggregator] n={n:5d}  "
                  f"err={rows[-1]:.1f} {unit}")
        res["models"]["GEO + equiv. aggregator"] = {
            "dim": n_inv + 2, "errors": rows,
            "equivariance_error": _model_equivariance(
                model, tok, test_items[:n_probe], task, device, seed,
                tokenize=_agg_tokens),
            "explanation_equivariance_error": _explanation_equivariance(
                model, tok, test_items[:n_attrib], device, seed,
                tokenize=_agg_tokens)}

    # Loud warning rather than a quiet bad number: a row worse than the
    # constant predictor is not a finding about symmetry, it is a row that
    # failed to fit, and it must not be reported as a comparison.
    failed = [m for m, r in res["models"].items()
              if min(r["errors"]) > baseline_err]
    if failed:
        res["warning_failed_to_fit"] = failed
        print("\nWARNING: these rows never beat the constant predictor and are "
              "NOT usable as a comparison:\n  " + "\n  ".join(failed)
              + "\n  Increase --epochs or the corpus size before reporting "
                "them; on a few hundred trips the shared backbone overfits.")

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


def _agg_tokens(tok, item):
    """Displacement channel only, laid out as [invariant | angular]."""
    d = np.diff(item["xy"], axis=0)
    return tok.disp.encode_disp(d).astype(np.float32)


def _model_equivariance(model, tok, items, task, device, seed, n_theta=8,
                        tokenize=None):
    """E_theta || pred(R.T) - rho(theta) pred(T) ||, normalised.

    rho is the rotation for `destination` and the identity for `eta`: the same
    measurement covers equivariance and invariance.
    """
    rng = np.random.default_rng(seed)
    tk = (lambda it: tokenize(tok, it)) if tokenize else \
         (lambda it: tok(_to_traj(it)).astype(np.float32))
    base = _predict(model, [tk(it) for it in items], device)
    errs = []
    for _ in range(n_theta):
        th = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        got = _predict(model, [tk(rotate_item(it, th)) for it in items], device)
        want = base @ R.T if task == "destination" else base
        errs.append(_rel(got, want))
    return float(np.mean(errs))


def _explanation_equivariance(model, tok, items, device, seed, n_theta=4,
                              tokenize=None):
    """Attributions live on input positions, so they must rotate with them."""
    rng = np.random.default_rng(seed)
    tk = (lambda it: tokenize(tok, it)) if tokenize else \
         (lambda it: tok(_to_traj(it)).astype(np.float32))

    # eval mode, but gradients still flow: BatchNorm cannot take a batch of
    # one in training mode, and running statistics are what we want anyway.
    model.eval()

    def attrib(it):
        x, m = pad_batch([tk(it)], device)
        x.requires_grad_(True)
        model(x, m).sum().backward()
        g = x.grad[0].cpu().numpy()
        model.zero_grad(set_to_none=True)
        d = np.diff(it["xy"], axis=0)
        w = np.linalg.norm(g[:len(d)], axis=1, keepdims=True)
        return w * d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)

    errs = []
    A0 = [attrib(it) for it in items]
    for _ in range(n_theta):
        th = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        for it, a0 in zip(items, A0):
            errs.append(_rel(attrib(rotate_item(it, th)), a0 @ R.T))
    model.train()
    return float(np.mean(errs))


if __name__ == "__main__":
    run()
