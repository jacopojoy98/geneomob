"""Downstream tasks: the encodings as primitive tokens for a larger model.

Every task uses the same grid,

    encoders  x  backbones  x  seeds

with identical backbone, head, optimiser, epochs, gradient clipping and token
standardisation, so the only thing that differs between rows is the encoding
of the raw GPS points. Four tasks, all on GeoLife by default:

  tul       trajectory-user linking       acc@1, acc@5, macro P / R / F1
  eta       travel-time estimation        MAE, RMSE, MAPE (minutes)
  mode      transportation-mode detection accuracy, macro F1, per-class F1
  anomaly   channel-aligned anomalies     per-type AUROC (+ SE), prediction checks

The encoder set is a 2 x 2 factorial, which is what makes the comparison
interpretable:

                     spatial only        + time + speed channels
    GPE              gpe                 gpe_ts
    GEO (yours)      geo                 geo_ts

GPE has no temporal channel of its own, so comparing GPE against GEO-with-time
would confound the spatial encoding with the extra information. `gpe_ts` gives
GPE exactly the same time and speed channels as `geo_ts`: the gpe -> geo and
gpe_ts -> geo_ts differences then measure the spatial encoding alone, and the
gpe -> gpe_ts difference measures what the extra channels are worth. Spatial
dimensions are matched (GPE h=128; GEO 64 location + 64 displacement).

Leakage guards, each of which would otherwise inflate a result:

* ETA: GeoLife samples at a near-fixed time interval, so the NUMBER of points
  encodes the duration and a model could count tokens. Paths are resampled at
  a fixed arc-length step first. The time channel is given only the departure
  time (broadcast along the path, as in standard travel-time estimation), and
  the speed channel is removed, since both encode the answer directly.
* TUL: split per user in time (each user's earliest trips train, latest
  test), so the same person cannot appear with near-duplicate commutes on both
  sides, and every user has training data.
* mode, eta, anomaly: user-disjoint splits by default.

Anomaly detection trains an LSTM or Transformer **sequence autoencoder** on
clean trajectories only, and never sees an anomaly or a label. Supervised
training on injected anomalies would let the model learn the injection
procedure instead of the anomaly, which is exactly the criticism the
channel-aligned design exists to avoid. Two scores are reported: the
reconstruction error, and the k-NN distance of a trajectory's latent code to
the clean training codes.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn

from ..config import save_results
from ..data import Traj, load_dataset
from ..encoders import (GPE, CyclicTimeGENEO, DisplacementGEO, SpeedGEO,
                        TorusLattice)
from ..geo import enu_to_lonlat, lonlat_to_enu
from ..models import TrajEncoder, pad_batch, set_seed

TASKS = ("tul", "eta", "mode", "anomaly")
MAIN_ENCODERS = ("gpe", "gpe_ts", "geo", "geo_ts")
# ablations used only by the anomaly channel checks
ANOMALY_ABLATIONS = ("geo_disp_ts", "geo_radial_ts")


# ==========================================================================
# encoders
# ==========================================================================
ENCODER_NAMES = {
    "gpe": "GPE",
    "gpe_ts": "GPE +time+speed",
    "geo": "GEO",
    "geo_ts": "GEO +time+speed",
    "geo_disp_ts": "GEO disp-only +time+speed",
    "geo_radial_ts": "GEO radial-only +time+speed",
}


def _geo_loc():
    from .exp_gpe_suite import geo_parts
    return geo_parts()[0]


def _geo_disp():
    # m=4 among the harmonics is what encodes alignment with a street grid
    # (see exp_anomaly for the finding)
    from .exp_gpe_suite import geo_parts
    return geo_parts()[1]


class _RadialOnly:
    def __init__(self):
        self.inner = _geo_disp()
        self.dim = self.inner.n_inv

    def encode_disp(self, dxy):
        return self.inner.encode_disp(dxy)[:, :self.inner.n_inv]


def make_tokenizer(key: str, task: str):
    """Build one encoder. For ETA the speed channel is dropped (it encodes
    the answer), and the time channel is fed only the departure time."""
    from .exp_similarity import Tokenizer
    ts = key.endswith("_ts")
    from .exp_gpe_suite import gpe_encoder, speed_encoder, time_encoder
    speed = speed_encoder() if (ts and task != "eta") else None
    tim = time_encoder() if ts else None
    name = ENCODER_NAMES[key] + (" (departure time)" if ts and task == "eta" else "")
    if key.startswith("gpe"):
        return Tokenizer(loc=gpe_encoder(), time=tim, speed=speed, name=name)
    if key in ("geo", "geo_ts"):
        return Tokenizer(loc=_geo_loc(), disp=_geo_disp(), time=tim,
                         speed=speed, name=name)
    if key == "geo_disp_ts":
        return Tokenizer(disp=_geo_disp(), time=tim, speed=speed, name=name)
    if key == "geo_radial_ts":
        return Tokenizer(disp=_RadialOnly(), time=tim, speed=speed, name=name)
    raise KeyError(key)


# ==========================================================================
# data preparation
# ==========================================================================
def resample_by_distance(traj: Traj, step_m: float = 30.0, min_points: int = 6,
                         max_points: int = 256):
    """Resample the path at a fixed arc-length step. Returns (Traj, length_m)
    or None if the path is too short. The returned Traj carries only the
    departure time, broadcast to every point.

    Paths longer than `max_points * step_m` get a proportionally wider step.
    The point count then stays a function of path LENGTH alone -- never of
    duration, which is the leak this function exists to remove."""
    xy, ref = lonlat_to_enu(traj.lonlat)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    length = float(s[-1])
    if length < step_m * (min_points - 1):
        return None
    step = max(step_m, length / (max_points - 1))
    grid = np.arange(0.0, length + 1e-9, step)
    rx = np.interp(grid, s, xy[:, 0])
    ry = np.interp(grid, s, xy[:, 1])
    t = np.full(len(grid), traj.t[0])
    return Traj(enu_to_lonlat(np.stack([rx, ry], 1), ref), t, traj.uid,
                traj.label), length


def split_user_disjoint(trajs, frac_train=0.8, seed=0):
    from ..csvdata import user_disjoint_split
    if any(t.uid is None for t in trajs):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(trajs))
        k = int(frac_train * len(trajs))
        return [trajs[i] for i in idx[:k]], [trajs[i] for i in idx[k:]]
    return user_disjoint_split(trajs, frac_train, seed)


def split_per_user_time(trajs, frac_train=0.8):
    """Each user's earliest trips train, latest test (TUL)."""
    by = {}
    for t in trajs:
        by.setdefault(t.uid, []).append(t)
    tr, te = [], []
    for u, ts in by.items():
        ts = sorted(ts, key=lambda x: x.t[0])
        k = max(1, int(round(frac_train * len(ts))))
        if k >= len(ts):
            k = len(ts) - 1
        tr += ts[:k]
        te += ts[k:]
    return tr, te


# ==========================================================================
# models
# ==========================================================================
class Head(nn.Module):
    def __init__(self, in_dim, out_dim, kind, hidden=None):
        from ..config import HP
        hidden = hidden or HP["model"]["hidden"]
        super().__init__()
        self.enc = TrajEncoder(in_dim, hidden=hidden, out_dim=hidden, kind=kind)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, m):
        return self.out(self.enc(x, m))


class SeqDecoder(nn.Module):
    """Decode a whole token sequence from a bottleneck code z."""

    def __init__(self, z_dim, out_dim, kind, hidden=None, max_len=1024, pdim=16):
        from ..config import HP
        hidden = hidden or HP["model"]["hidden"]
        super().__init__()
        pos = np.arange(max_len)[:, None] / (10000 ** (np.arange(pdim // 2) / (pdim // 2)))
        pe = np.concatenate([np.sin(pos), np.cos(pos)], 1).astype(np.float32)
        self.register_buffer("pe", torch.from_numpy(pe))
        self.kind = kind
        if kind == "transformer":
            self.proj = nn.Linear(z_dim + pdim, hidden)
            layer = nn.TransformerEncoderLayer(hidden, HP["model"]["heads"],
                                               4 * hidden, batch_first=True,
                                               dropout=HP["model"]["dropout"])
            self.body = nn.TransformerEncoder(layer, HP["model"]["layers"])
        else:
            self.body = nn.LSTM(z_dim + pdim, hidden,
                                num_layers=HP["model"]["layers"],
                                batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, z, L, mask):
        B = z.shape[0]
        x = torch.cat([z.unsqueeze(1).expand(B, L, -1),
                       self.pe[:L].unsqueeze(0).expand(B, L, -1)], -1)
        if self.kind == "transformer":
            h = self.body(self.proj(x), src_key_padding_mask=~mask)
        else:
            h, _ = self.body(x)
        return self.out(h)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, kind, z_dim=16, hidden=None):
        from ..config import HP
        hidden = hidden or HP["model"]["hidden"]
        super().__init__()
        self.enc = TrajEncoder(in_dim, hidden=hidden, out_dim=z_dim, kind=kind)
        self.dec = SeqDecoder(z_dim, in_dim, kind, hidden=hidden)

    def forward(self, x, m):
        z = self.enc(x, m)
        return z, self.dec(z, x.shape[1], m)


# ==========================================================================
# training helpers
# ==========================================================================
class Standardiser:
    """Per-dimension token standardisation, fitted on the training split."""

    def fit(self, X):
        A = np.concatenate(X, 0)
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-6
        return self

    def __call__(self, X):
        return [((x - self.mu) / self.sd).astype(np.float32) for x in X]


def _batches(lengths, bs, rng):
    """Length-bucketed batches in random order.

    Padding every batch to its longest member makes a random batch as slow as
    the longest sequence in the corpus; grouping similar lengths removes most
    of that waste. Ties are reshuffled each epoch and the batch order is
    shuffled, so the optimisation is not biased by length.
    """
    lengths = np.asarray(lengths)
    order = np.lexsort((rng.random(len(lengths)), lengths))
    batches = [order[s:s + bs] for s in range(0, len(order), bs)]
    for i in rng.permutation(len(batches)):
        if len(batches[i]) >= 2:
            yield batches[i]


def _fit(model, X, loss_fn, epochs, bs, lr, device, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lengths = [len(x) for x in X]
    for _ in range(epochs):
        for b in _batches(lengths, bs, rng):
            x, m = pad_batch([X[i] for i in b], device)
            loss = loss_fn(model, x, m, b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def _apply(model, X, device, bs=256, fn=None):
    model.eval()
    order = np.argsort([len(x) for x in X])      # length-sorted inference
    outs = []
    for s in range(0, len(X), bs):
        idx = order[s:s + bs]
        x, m = pad_batch([X[i] for i in idx], device)
        outs.append(fn(model, x, m) if fn else model(x, m).cpu().numpy())
    model.train()
    res = np.concatenate(outs)
    back = np.empty_like(res)
    back[order] = res                            # restore original order
    return back


# ==========================================================================
# metrics
# ==========================================================================
def classification_metrics(logits, y, n_classes, k=5):
    pred = logits.argmax(1)
    out = {"acc@1": float((pred == y).mean())}
    if n_classes > k:
        top = np.argsort(-logits, 1)[:, :k]
        out[f"acc@{k}"] = float(np.mean([y[i] in top[i] for i in range(len(y))]))
    present = np.unique(y)          # macro over classes present in the test set
    P, R, F = [], [], []
    for c in present:
        tp = np.sum((pred == c) & (y == c))
        p = tp / max(np.sum(pred == c), 1)
        r = tp / max(np.sum(y == c), 1)
        P.append(p)
        R.append(r)
        F.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    out.update({"macro_P": float(np.mean(P)), "macro_R": float(np.mean(R)),
                "macro_F1": float(np.mean(F))})
    return out, {int(c): float(f) for c, f in zip(present, F)}


def regression_metrics(pred, y, mape_floor=1.0):
    e = pred - y
    m = y >= mape_floor            # MAPE is undefined for near-zero targets
    return {"MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE": float(np.mean(np.abs(e[m]) / y[m]) * 100) if m.any() else float("nan")}


# ==========================================================================
# hand-crafted baselines
# ==========================================================================
def kinematic_features(traj: Traj):
    """The features a mode-detection paper would use without any learning."""
    xy = lonlat_to_enu(traj.lonlat)[0]
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(traj.t) * 3600.0
    ok = dt > 0
    v = d[ok] / dt[ok] if ok.any() else np.zeros(1)
    a = np.abs(np.diff(v)) / np.maximum(dt[ok][1:], 1e-6) if len(v) > 1 else np.zeros(1)
    psi = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    turn = np.abs(np.angle(np.exp(1j * np.diff(psi)))) if len(psi) > 1 else np.zeros(1)
    dist = d.sum()
    return np.array([np.mean(v), np.median(v), np.percentile(v, 85),
                     np.percentile(v, 95), np.mean(a), np.percentile(a, 85),
                     turn.sum() / max(dist, 1.0), np.mean(v < 0.5),
                     np.log1p(dist), np.log1p(max(traj.t[-1] - traj.t[0], 0) * 60)],
                    dtype=np.float64)


def _fit_linear(Xtr, ytr, Xte, n_classes=None, steps=400, wd=1e-3, seed=0):
    """Multinomial logistic regression, or least squares if n_classes is None."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    if n_classes is None:
        A1 = np.c_[A, np.ones(len(A))]
        w = np.linalg.solve(A1.T @ A1 + wd * np.eye(A1.shape[1]), A1.T @ ytr)
        return np.c_[B, np.ones(len(B))] @ w
    torch.manual_seed(seed)
    lin = nn.Linear(A.shape[1], n_classes)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=steps)
    At = torch.from_numpy(A).float()
    yt = torch.from_numpy(ytr).long()

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lin(At), yt) + wd * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return lin(torch.from_numpy(B).float()).numpy()


# ==========================================================================
# the four tasks
# ==========================================================================
def task_tul(trajs, encoders, backbones, seeds, cfg):
    counts = {}
    for t in trajs:
        counts[t.uid] = counts.get(t.uid, 0) + 1
    trajs = [t for t in trajs if t.uid is not None and counts[t.uid] >= cfg["min_trips"]]
    train, test = split_per_user_time(trajs)
    if cfg.get("validation"):
        # hyperparameter search: the test trips are set aside untouched and
        # the training trips are split again, the same way
        train, test = split_per_user_time(train)
    users = sorted({t.uid for t in train})
    ix = {u: i for i, u in enumerate(users)}
    test = [t for t in test if t.uid in ix]
    ytr = np.array([ix[t.uid] for t in train])
    yte = np.array([ix[t.uid] for t in test])
    C = len(users)
    base = {"majority": float(np.bincount(yte, minlength=C).max() / len(yte)),
            "random": 1.0 / C}
    print(f"[tul] {C} users, {len(train)} train / {len(test)} test, "
          f"majority {base['majority']:.3f}, random {base['random']:.3f}")

    def run(key, kind, seed):
        tok = make_tokenizer(key, "tul")
        tok.fit(train)
        S = Standardiser().fit([tok(t) for t in train])
        Xtr, Xte = S([tok(t) for t in train]), S([tok(t) for t in test])
        yt = torch.from_numpy(ytr).long()
        model = Head(tok.dim, C, kind)
        _fit(model, Xtr, lambda M, x, m, b: nn.functional.cross_entropy(
            M(x, m), yt[b].to(x.device)), cfg["epochs"], cfg["bs"], cfg["lr"],
            cfg["device"], seed)
        return classification_metrics(_apply(model, Xte, cfg["device"]), yte, C)[0]

    return _grid(run, encoders, backbones, seeds, "tul"), base, \
        {"n_users": C, "n_train": len(train), "n_test": len(test)}


def task_eta(trajs, encoders, backbones, seeds, cfg):
    items = []
    for t in trajs:
        dur_min = (t.t[-1] - t.t[0]) * 60.0
        if not (0.5 <= dur_min <= cfg["eta_max_min"]):
            continue
        r = resample_by_distance(t, cfg["eta_step_m"])
        if r is None:
            continue
        items.append((r[0], r[1], dur_min, t.t[0]))
    # split the (path, length, duration, departure) items by user
    tr_i, te_i = _split_indices(items, seed=0)
    train, test = [items[i] for i in tr_i], [items[i] for i in te_i]
    if cfg.get("validation"):       # search: re-split train, test untouched
        a_i, b_i = _split_indices(train, seed=1)
        train, test = [train[i] for i in a_i], [train[i] for i in b_i]
    ytr = np.array([it[2] for it in train])
    yte = np.array([it[2] for it in test])
    print(f"[eta] {len(train)} train / {len(test)} test paths "
          f"(resampled every {cfg['eta_step_m']:.0f} m), "
          f"median duration {np.median(yte):.1f} min")

    # baselines: constant, and least squares on length + departure time
    def lin_feats(its):
        return np.array([[np.log1p(it[1]), it[1] / 1000.0,
                          np.sin(2 * np.pi * (it[3] % 24) / 24), np.cos(2 * np.pi * (it[3] % 24) / 24),
                          np.sin(2 * np.pi * ((it[3] // 24) % 7) / 7), np.cos(2 * np.pi * ((it[3] // 24) % 7) / 7)]
                         for it in its])
    base = {"constant": regression_metrics(np.full(len(yte), ytr.mean()), yte),
            "length+departure linear": regression_metrics(
                _fit_linear(lin_feats(train), ytr, lin_feats(test)), yte)}
    scale = float(ytr.std() + 1e-6)
    mu = float(ytr.mean())

    def run(key, kind, seed):
        tok = make_tokenizer(key, "eta")
        tok.fit([it[0] for it in train])
        S = Standardiser().fit([tok(it[0]) for it in train])
        Xtr = S([tok(it[0]) for it in train])
        Xte = S([tok(it[0]) for it in test])
        yt = torch.from_numpy(((ytr - mu) / scale).astype(np.float32))
        model = Head(tok.dim, 1, kind)
        _fit(model, Xtr, lambda M, x, m, b: nn.functional.mse_loss(
            M(x, m).squeeze(-1), yt[b].to(x.device)), cfg["epochs"], cfg["bs"],
            cfg["lr"], cfg["device"], seed)
        pred = _apply(model, Xte, cfg["device"]).squeeze(-1) * scale + mu
        return regression_metrics(pred, yte)

    return _grid(run, encoders, backbones, seeds, "eta"), base, \
        {"n_train": len(train), "n_test": len(test), "step_m": cfg["eta_step_m"]}


def _split_indices(items, frac_train=0.8, seed=0):
    """User-disjoint split over tuples whose first element is a Traj."""
    uids = [it[0].uid for it in items]
    rng = np.random.default_rng(seed)
    if any(u is None for u in uids):
        idx = rng.permutation(len(items))
        k = int(frac_train * len(items))
        return list(idx[:k]), list(idx[k:])
    users = sorted(set(uids))
    rng.shuffle(users)
    keep = set(users[:max(int(frac_train * len(users)), 1)])
    return ([i for i, u in enumerate(uids) if u in keep],
            [i for i, u in enumerate(uids) if u not in keep])


def task_mode(trajs, encoders, backbones, seeds, cfg):
    trajs = [t for t in trajs if t.label and t.label != "other"]
    if not trajs:
        raise ValueError("mode task: no mode-labelled trajectories. On GeoLife "
                         "they come from each user's labels.txt.")
    classes = sorted({t.label for t in trajs})
    ix = {c: i for i, c in enumerate(classes)}
    train, test = (split_user_disjoint(trajs, seed=0) if cfg["mode_split"] == "user"
                   else split_user_disjoint([Traj(t.lonlat, t.t, None, t.label)
                                             for t in trajs], seed=0))
    if cfg.get("validation"):       # search: re-split train, test untouched
        train, test = (split_user_disjoint(train, seed=1)
                       if cfg["mode_split"] == "user" else
                       split_user_disjoint([Traj(t.lonlat, t.t, None, t.label)
                                            for t in train], seed=1))
    ytr = np.array([ix[t.label] for t in train])
    yte = np.array([ix[t.label] for t in test])
    C = len(classes)
    Ftr = np.stack([kinematic_features(t) for t in train])
    Fte = np.stack([kinematic_features(t) for t in test])
    hand, hand_pc = classification_metrics(_fit_linear(Ftr, ytr, Fte, C), yte, C)
    base = {"majority": classification_metrics(
                np.eye(C)[np.full(len(yte), np.bincount(ytr, minlength=C).argmax())],
                yte, C)[0],
            "hand-crafted kinematics + logistic": hand}
    counts = {c: int((ytr == ix[c]).sum()) for c in classes}
    print(f"[mode] classes {counts} (train), {len(train)} train / {len(test)} test, "
          f"hand-crafted macro-F1 {hand['macro_F1']:.3f}")

    per_class = {}

    def run(key, kind, seed):
        tok = make_tokenizer(key, "mode")
        tok.fit(train)
        S = Standardiser().fit([tok(t) for t in train])
        Xtr, Xte = S([tok(t) for t in train]), S([tok(t) for t in test])
        yt = torch.from_numpy(ytr).long()
        model = Head(tok.dim, C, kind)
        _fit(model, Xtr, lambda M, x, m, b: nn.functional.cross_entropy(
            M(x, m), yt[b].to(x.device)), cfg["epochs"], cfg["bs"], cfg["lr"],
            cfg["device"], seed)
        met, pc = classification_metrics(_apply(model, Xte, cfg["device"]), yte, C)
        per_class.setdefault(f"{key}/{kind}", []).append(pc)
        return met

    grid = _grid(run, encoders, backbones, seeds, "mode")
    return grid, base, {"classes": classes, "n_train": len(train),
                        "n_test": len(test), "train_counts": counts,
                        "hand_crafted_per_class": {classes[c]: v for c, v in hand_pc.items()},
                        "per_class_f1": {k: {classes[c]: float(np.mean([d.get(c, np.nan) for d in v]))
                                             for c in range(C)} for k, v in per_class.items()}}


def task_anomaly(trajs, encoders, backbones, seeds, cfg):
    from ..anomaly import EXPECTED, TYPES, build_anomaly_set, trivial_scores
    from .exp_anomaly import _check_predictions, auroc, auroc_se, knn_score

    # Per-user temporal split, NOT user-disjoint. Anomaly detection asks
    # whether a trip deviates from established behaviour, so the normal test
    # trips must come from the same population as training. Under a
    # user-disjoint split every normal test trip belongs to an unseen person
    # with unseen homes and workplaces -- already an "unseen area" -- and the
    # location-aware encoders are penalised for knowing where things are:
    # measured, that split put every encoder at chance (AUROC ~0.48).
    train, test = split_per_user_time(trajs)
    ref = np.concatenate([t.lonlat for t in train]).mean(0)
    items, _ = build_anomaly_set(test, types=TYPES, rate=cfg["anomaly_rate"],
                                 severity=cfg["severity"], ref=ref, seed=0)
    kinds = np.array([it.kind for it in items])
    # the displacement ablations only exist if GEO has those parts
    d = _geo_disp()
    abl = [k for k in ANOMALY_ABLATIONS if k not in encoders and d is not None
           and not (k == "geo_radial_ts" and d.n_inv == 0)]
    if len(abl) < len([k for k in ANOMALY_ABLATIONS if k not in encoders]):
        print("  [anomaly] note: this GEO layout has no "
              + ("displacement channel" if d is None else "radial part")
              + "; the corresponding ablation row is omitted")
    enc_all = tuple(encoders) + tuple(abl)

    def per_type(s):
        normal = kinds == "normal"
        out, se = {}, {}
        for ty in TYPES:
            m = normal | (kinds == ty)
            out[ty] = auroc(s[m], kinds[m] == ty)
            se[ty] = auroc_se(out[ty], int((kinds == ty).sum()), int(normal.sum()))
        out["all"] = auroc(s, ~normal)
        out["_se"] = se
        return out

    def run(key, kind, seed):
        tok = make_tokenizer(key, "anomaly")
        tok.fit(train)
        S = Standardiser().fit([tok(t) for t in train])
        Xtr = S([tok(t) for t in train])
        Xte = S([tok(it.traj) for it in items])
        model = AutoEncoder(tok.dim, kind, z_dim=cfg["z_dim"])

        def loss(M, x, m, b):
            _, r = M(x, m)
            err = ((r - x) ** 2).mean(-1)
            return (err * m).sum() / m.sum()
        _fit(model, Xtr, loss, cfg["epochs"], cfg["bs"], cfg["lr"],
             cfg["device"], seed)

        def z_and_err(M, x, m):
            z, r = M(x, m)
            e = (((r - x) ** 2).mean(-1) * m).sum(1) / m.sum(1)
            return torch.cat([z, e.unsqueeze(1)], 1).cpu().numpy()
        Ztr = _apply(model, Xtr, cfg["device"], fn=z_and_err)
        Zte = _apply(model, Xte, cfg["device"], fn=z_and_err)
        # reference with no model at all: k-NN on mean+std pooled tokens.
        # If the LSTM/Transformer cannot beat this, it added nothing.
        from ..metrics import pool_tokens
        Ptr = np.stack([pool_tokens(x) for x in Xtr])
        Pte = np.stack([pool_tokens(x) for x in Xte])
        return {"latent_knn": per_type(knn_score(Ztr[:, :-1], Zte[:, :-1], cfg["k"])),
                "reconstruction": per_type(Zte[:, -1]),
                "pooled_tokens_no_model": per_type(knn_score(Ptr, Pte, cfg["k"]))}

    raw = {}
    for key in enc_all:
        for kind in backbones:
            runs = []
            for seed in seeds:
                t0 = time.perf_counter()
                runs.append(run(key, kind, seed))
                print(f"  [anomaly] {ENCODER_NAMES[key]:28s} {kind:11s} seed {seed}  "
                      f"latent-kNN={runs[-1]['latent_knn']['all']:.3f}  "
                      f"recon={runs[-1]['reconstruction']['all']:.3f}  "
                      f"no-model={runs[-1]['pooled_tokens_no_model']['all']:.3f}  "
                      f"({time.perf_counter() - t0:.0f}s)")
            raw.setdefault(key, {})[kind] = {
                sc: _mean_dict([r[sc] for r in runs]) for sc in SCORES}
            raw[key][kind]["_seeds"] = [
                {"seed": int(sd), **{sc: {"all": r[sc]["all"]} for sc in SCORES}}
                for sd, r in zip(seeds, runs)]

    triv = {f"[trivial] {n}": per_type(s) for n, s in trivial_scores(items, ref).items()}
    checks = {}
    for kind in backbones:
        for sc in SCORES:
            row = {ENCODER_NAMES[k]: raw[k][kind][sc] for k in raw}
            checks[f"{kind}/{sc}"] = _check_predictions(
                {"runs": {"1": row}}, TYPES, pairs=ANOMALY_PAIRS)
    counts = {ty: int((kinds == ty).sum()) for ty in TYPES}
    return raw, {"trivial": triv}, {"n_train": len(train), "n_test": len(items),
                                    "counts": counts, "expected": EXPECTED,
                                    "prediction_check": checks}


SCORES = ("latent_knn", "reconstruction", "pooled_tokens_no_model")

# registered before running; exact representation names
ANOMALY_PAIRS = [
    ("off_grid", "GEO radial-only +time+speed", "missed", 0.60),
    ("unseen_area", "GEO disp-only +time+speed", "missed", 0.60),
    ("unseen_area", "GEO +time+speed", "caught", 0.65),
    ("wrong_time", "GEO", "missed", 0.60),
    ("wrong_time", "GPE", "missed", 0.60),
    ("wrong_time", "GEO +time+speed", "caught", 0.65),
    ("wrong_time", "GPE +time+speed", "caught", 0.65),
    ("kinematic", "GEO", "missed", 0.60),
    ("kinematic", "GPE", "missed", 0.60),
    ("kinematic", "GEO +time+speed", "caught", 0.65),
]


# ==========================================================================
# grid + aggregation
# ==========================================================================
def _mean_dict(ds):
    """Mean (and std, when >1 seed) of nested metric dicts."""
    out = {}
    for k in ds[0]:
        if isinstance(ds[0][k], dict):
            out[k] = _mean_dict([d[k] for d in ds])
        else:
            v = np.array([d[k] for d in ds], dtype=float)
            out[k] = float(np.nanmean(v))
            if len(v) > 1:
                out[k + "_std"] = float(np.nanstd(v))
    return out


def _grid(run, encoders, backbones, seeds, task):
    res = {}
    for key in encoders:
        for kind in backbones:
            runs = []
            for seed in seeds:
                t0 = time.perf_counter()
                runs.append(run(key, kind, seed))
                head = {k: v for k, v in runs[-1].items() if not k.startswith("_")}
                print(f"  [{task}] {ENCODER_NAMES[key]:22s} {kind:11s} seed {seed}  "
                      + "  ".join(f"{a}={b:.3f}" for a, b in list(head.items())[:4])
                      + f"  ({time.perf_counter() - t0:.0f}s)")
            cell = _mean_dict(runs)
            # keep every seed's numbers: they are the replicates behind the
            # paired tests and the per-seed critical-difference blocks
            cell["_seeds"] = [{"seed": int(sd), **r} for sd, r in zip(seeds, runs)]
            res.setdefault(key, {})[kind] = cell
    return res


# ==========================================================================
# entry point
# ==========================================================================
def run(dataset="geolife", path=None, tasks=TASKS, encoders=MAIN_ENCODERS,
        backbones=("lstm", "transformer"), seeds=(0,), epochs=10, bs=64,
        lr=1e-3, max_traj=20000, min_trips=10, eta_step_m=30.0,
        eta_max_min=180.0, mode_split="user", anomaly_rate=0.28,
        severity=1.0, z_dim=16, k=10, device="cpu", out=None):
    cfg = dict(epochs=epochs, bs=bs, lr=lr, device=device, min_trips=min_trips,
               eta_step_m=eta_step_m, eta_max_min=eta_max_min,
               mode_split=mode_split, anomaly_rate=anomaly_rate,
               severity=severity, z_dim=z_dim, k=k)
    kw = {} if dataset.startswith("synthetic") else {"max_traj": max_traj}
    if dataset.startswith("synthetic"):
        kw["n"] = min(max_traj, 3000)
    trajs = load_dataset(dataset, path, **kw)

    res = {"dataset": dataset, "encoders": {k: ENCODER_NAMES[k] for k in encoders},
           "backbones": list(backbones), "seeds": list(seeds), "config": cfg,
           "tasks": {}, "skipped_tasks": {}}
    runners = {"tul": task_tul, "eta": task_eta, "mode": task_mode,
               "anomaly": task_anomaly}

    def skip(task, why):
        res["skipped_tasks"][task] = why
        print(f"  !! SKIPPED task '{task}': {why}")

    for task in tasks:
        data = trajs
        if task == "mode":
            if dataset == "geolife":
                try:
                    data = load_dataset("geolife", path, max_traj=max_traj,
                                        labelled_only=True)
                except (FileNotFoundError, ValueError) as e:
                    skip(task, f"no mode-labelled GeoLife trajectories ({e}); "
                               f"labels.txt files are needed")
                    continue
            elif not any(t.label for t in trajs):
                skip(task, "this corpus carries no mode labels "
                           "(GeoLife does, via labels.txt)")
                continue
        t0 = time.perf_counter()
        try:
            grid, base, meta = runners[task](data, encoders, backbones, seeds, cfg)
        except (ValueError, IndexError, KeyError, ZeroDivisionError) as e:
            # typically "not enough users / labelled classes / trips" on a
            # small corpus; the other tasks can still run
            import traceback
            traceback.print_exc()
            skip(task, f"{type(e).__name__}: {e}")
            continue
        res["tasks"][task] = {"results": grid, "baselines": base, "meta": meta,
                              "seconds": time.perf_counter() - t0}
        if out:                      # checkpoint after every task
            save_results(res, out)
    _summary(res)
    if out:
        save_results(res, out)
    return res


HEADLINE = {"tul": ("acc@1", True), "eta": ("MAE", False),
            "mode": ("macro_F1", True)}


def _summary(res):
    print("\n==== summary ====")
    for task, why in (res.get("skipped_tasks") or {}).items():
        print(f"[{task}] NOT RUN: {why}")
    for task, d in res["tasks"].items():
        if task == "anomaly":
            print("[anomaly] AUROC (latent kNN), all types pooled")
            for key, bb in d["results"].items():
                print("  " + ENCODER_NAMES[key].ljust(30) + "  ".join(
                    f"{kind}={v['latent_knn']['all']:.3f}" for kind, v in bb.items()))
            continue
        metric, higher = HEADLINE[task]
        print(f"[{task}] {metric} ({'higher' if higher else 'lower'} is better)")
        for key, bb in d["results"].items():
            print("  " + ENCODER_NAMES[key].ljust(30) + "  ".join(
                f"{kind}={v[metric]:.3f}" + (f"±{v[metric + '_std']:.3f}" if metric + "_std" in v else "")
                for kind, v in bb.items()))
        for name, b in d["baselines"].items():
            v = b if not isinstance(b, dict) else b.get(metric)
            if v is not None:
                print(f"  (baseline) {name:28s} {v:.3f}")


if __name__ == "__main__":
    run()
