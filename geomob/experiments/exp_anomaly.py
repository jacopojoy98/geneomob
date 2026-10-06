"""Anomaly detection with a channel-aligned taxonomy.

The experiment is a prediction test, not a leaderboard. `anomaly.EXPECTED`
states, before anything runs, which channel should register each anomaly type
and which ablation should provably fail to. This script fills in the table and
marks every cell where the prediction was wrong.

Detection is unsupervised: pool each trajectory's tokens, fit nothing but the
clean training set, and score a test trajectory by its mean distance to its k
nearest clean neighbours. No labels are used at any point, and the same
detector is applied to every representation, so the comparison is about the
representation alone.

Two guards against the usual criticism of synthetic-anomaly work:

* `trivial_scores` detectors (path length, mean speed, duration, bounding-box
  area, start hour) are reported in the same table. If one of them matches the
  model on some anomaly type, the model has shown nothing there.
* the severity sweep (`--severities`) shows detection degrading smoothly as the
  perturbation shrinks, rather than a single operating point chosen after the
  fact.

Two findings from the first runs are baked in as configurations rather than
left implicit, because both contradicted the initial predictions:

* **A re-timed trajectory is invisible to the base construction.** phi(dx)
  encodes displacement magnitude and phi_time encodes absolute time-of-day;
  neither sees dt. A one-line mean-speed detector scores 1.000 where the
  tokenizer scored 0.54. `SpeedGEO` closes it.
* **Grid orientation needs the m=4 angular harmonic.** With m=1 the
  `off_grid` signal is entirely positional -- the 90-degree control scores
  *higher* than the 45-degree anomaly, at every grid strength up to a
  perfectly grid-locked corpus. Mean+std pooling keeps the net heading and
  discards heading-mod-90, which is what alignment is. Adding m=4 reverses
  the sign of the gap. Equivariant tokens are necessary but not sufficient;
  the harmonic order must match the symmetry, and the aggregation must
  preserve it.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import save_results
from ..anomaly import EXPECTED, TYPES, build_anomaly_set, trivial_scores
from ..data import load_dataset
from ..encoders import (GPE, CyclicTimeGENEO, DisplacementGEO, GridEmbed,
                        RawXY, SpeedGEO, TorusLattice)
from ..metrics import pool_tokens


# --------------------------------------------------------------------------
# representations, chosen so each prediction in EXPECTED has a witness
# --------------------------------------------------------------------------
class RadialOnly:
    """The displacement GEO with its angular channel removed.

    This is the rotation-*invariant* ablation, and it is in the comparison for
    one reason: it is predicted to be blind to `off_grid`, which preserves
    every invariant exactly. If it detects off-grid anomalies anyway, the
    prediction is wrong and something else is leaking.
    """

    name = "radial only (rot-invariant)"

    def __init__(self, **kw):
        self.inner = DisplacementGEO(**kw)
        self.dim = self.inner.n_inv

    def encode_disp(self, dxy):
        return self.inner.encode_disp(dxy)[:, :self.inner.n_inv]


def build_representations(which, ref):
    from .exp_similarity import Tokenizer
    reg = {
        "geo_full": lambda: Tokenizer(loc=TorusLattice(ref_deg=ref),
                                      disp=DisplacementGEO(),
                                      time=CyclicTimeGENEO(),
                                      name="GEO loc+disp+time"),
        "geo_spatial": lambda: Tokenizer(loc=TorusLattice(ref_deg=ref),
                                         disp=DisplacementGEO(),
                                         name="GEO loc+disp"),
        # Grid alignment is heading-mod-90, i.e. a C4 phenomenon, so it lives
        # in the m=4 harmonic of the angular channel. With m=1 only, mean+std
        # pooling keeps the NET heading and destroys the mod-90 structure, and
        # the orientation gap comes out negative however strong the grid is.
        # Adding m=4 flips its sign. Equivariant tokens are necessary but not
        # sufficient: the harmonic order has to match the symmetry you intend
        # to detect, and the aggregation has to preserve it.
        "geo_c4": lambda: Tokenizer(loc=TorusLattice(ref_deg=ref),
                                    disp=DisplacementGEO(harmonics=(1, 4)),
                                    time=CyclicTimeGENEO(), speed=SpeedGEO(),
                                    name="GEO +speed +m=4 harmonic"),
        # the configuration the kinematic row motivates
        "geo_speed": lambda: Tokenizer(loc=TorusLattice(ref_deg=ref),
                                       disp=DisplacementGEO(),
                                       time=CyclicTimeGENEO(),
                                       speed=SpeedGEO(),
                                       name="GEO +speed (all channels)"),
        "disp_only": lambda: Tokenizer(disp=DisplacementGEO(),
                                       name="disp only (no abs. position)"),
        "radial_only": lambda: Tokenizer(disp=RadialOnly(),
                                         name="radial only (rot-invariant)"),
        "time_only": lambda: Tokenizer(time=CyclicTimeGENEO(),
                                       name="time only"),
        "gpe": lambda: Tokenizer(loc=GPE(h=128), name="GPE"),
        "grid": lambda: Tokenizer(loc=GridEmbed(150.0), name="Grid"),
        "xy": lambda: Tokenizer(loc=RawXY(normalize=True), name="XY"),
    }
    return {k: reg[k]() for k in which}


# --------------------------------------------------------------------------
# detector + metric
# --------------------------------------------------------------------------
def knn_score(train_X, test_X, k: int = 10):
    """Mean distance to the k nearest clean trajectories. Higher = stranger."""
    mu, sd = train_X.mean(0), train_X.std(0) + 1e-9
    A, B = (train_X - mu) / sd, (test_X - mu) / sd
    d = np.sqrt(np.maximum(
        (B ** 2).sum(1)[:, None] - 2 * B @ A.T + (A ** 2).sum(1)[None, :], 0))
    k = min(k, d.shape[1])
    return np.sort(d, axis=1)[:, :k].mean(1)


def auroc_se(a: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of an AUROC.

    With a handful of positives per anomaly type the noise floor is large --
    around 0.07 for 17 positives against 680 negatives -- so a raw threshold
    test manufactures spurious pass/fail verdicts. Every check reports this
    alongside, and calls within two standard errors are marked inconclusive
    rather than decided.
    """
    if not np.isfinite(a) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
         + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(v, 0.0)))


def auroc(scores, labels):
    """Rank-based AUROC; 0.5 is chance, and no sklearn dependency."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so a constant score scores exactly 0.5
    s = np.asarray(scores, float)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------
def run(dataset="synthetic_hard", path=None,
        which=("geo_c4", "geo_speed", "geo_full", "geo_spatial", "disp_only",
               "radial_only", "time_only", "gpe", "grid"),
        types=TYPES, rate=0.28, severities=(1.0, 0.5, 0.25), k=10,
        n_train=2000, n_test=1000, split="user", seed=0, out=None, **load_kw):
    # the synthetic generator makes exactly as many as it is asked for, so
    # request enough for both splits rather than silently truncating
    if dataset.startswith("synthetic") and "n" not in load_kw:
        load_kw["n"] = n_train + n_test
    trajs = load_dataset(dataset, path, **load_kw)
    from .exp_similarity import _split
    train, test = _split(trajs, dataset, split, n_train, n_test, seed)
    if len(test) < 50:
        raise ValueError(
            f"only {len(test)} test trajectories; lower --n-train or supply a "
            f"larger corpus (the split reserves n_train for fitting)")
    ref = np.concatenate([t.lonlat for t in train]).mean(0)
    print(f"{len(train)} clean training / {len(test)} test trajectories")

    res = {"dataset": dataset, "rate": rate, "k": k, "types": list(types),
           "severities": list(severities), "expected": EXPECTED,
           "n_train": len(train), "n_test": len(test), "runs": {}}

    for sev in severities:
        items, _ = build_anomaly_set(test, types=types, rate=rate,
                                     severity=sev, ref=ref, seed=seed)
        kinds = np.array([it.kind for it in items])
        row = {}

        # --- the representations
        for key, tok in build_representations(which, ref).items():
            tok.fit(train)
            Xtr = np.stack([pool_tokens(tok(t)) for t in train])
            Xte = np.stack([pool_tokens(tok(it.traj)) for it in items])
            s = knn_score(Xtr, Xte, k)
            row[tok.name] = _per_type_auroc(s, kinds, types)

        # --- the guard
        for name, s in trivial_scores(items, ref).items():
            row[f"[trivial] {name}"] = _per_type_auroc(s, kinds, types)

        res["runs"][str(sev)] = row
        _print_table(row, types, sev)

    res["prediction_check"] = _check_predictions(res, types)
    print("\n" + json.dumps(res["prediction_check"], indent=2, default=float))
    if out:
        save_results(res, out)
    return res


def _per_type_auroc(scores, kinds, types):
    """One-vs-normal AUROC per anomaly type, so a representation that catches
    only one kind cannot hide behind a pooled average."""
    normal = kinds == "normal"
    out, se = {}, {}
    for ty in types:
        m = normal | (kinds == ty)
        out[ty] = auroc(scores[m], (kinds[m] == ty))
        se[ty] = auroc_se(out[ty], int((kinds == ty).sum()), int(normal.sum()))
    out["all"] = auroc(scores, ~normal)
    out["_se"] = se
    return out


def _print_table(row, types, sev):
    print(f"\nAUROC by anomaly type, severity {sev}  (0.5 = chance)")
    head = "  " + "representation".ljust(30) + "".join(t[:9].rjust(11) for t in types)
    print(head + "        all")
    for name, d in row.items():
        line = "  " + name[:30].ljust(30)
        for t in types:
            line += f"{d[t]:11.3f}"
        print(line + f"{d['all']:11.3f}")


def _check_predictions(res, types, pairs=None):
    """Did each prediction in EXPECTED hold at the strongest severity?

    `caught` needs AUROC clearly above chance for the representation named;
    `missed` needs it near chance for the ablation named. Both are reported
    rather than asserted, so a failed prediction shows up as a finding.
    """
    sev = max(res["runs"], key=float)       # the key itself, whatever its spelling
    row = res["runs"][sev]

    def get(exact, ty):
        # exact match: "GEO loc+disp" is a substring of "GEO loc+disp+time",
        # and substring matching silently graded the no-time ablation using
        # the with-time row, inverting the conclusion.
        d = row.get(exact)
        return None if d is None else d.get(ty)

    checks = {}
    # The orientation diagnostic: off_grid (45 deg, breaks alignment) minus
    # off_grid_control (90 deg, preserves it). Both move positions equally, so
    # a positive gap is orientation sensitivity and nothing else.
    if "off_grid_control" in types:
        for name, d in row.items():
            if np.isfinite(d.get("off_grid", np.nan)) and \
               np.isfinite(d.get("off_grid_control", np.nan)):
                checks.setdefault("orientation_sensitivity", {})[name] = {
                    "off_grid": d["off_grid"],
                    "control_90deg": d["off_grid_control"],
                    "gap": d["off_grid"] - d["off_grid_control"]}

    pairs = pairs or [("off_grid", "radial only (rot-invariant)", "missed", 0.60),
             ("unseen_area", "disp only (no abs. position)", "missed", 0.60),
             ("wrong_time", "GEO loc+disp", "missed", 0.60),
             ("kinematic", "GEO loc+disp+time", "missed", 0.60),
             ("wrong_time", "time only", "caught", 0.65),
             ("kinematic", "GEO +speed (all channels)", "caught", 0.65),
             ("unseen_area", "GEO loc+disp", "caught", 0.65)]
    for ty, rep, kind, thr in pairs:
        v = get(rep, ty)
        if v is None or not np.isfinite(v):
            continue
        se = row[rep].get("_se", {}).get(ty, float("nan"))
        ok = (v < thr) if kind == "missed" else (v > thr)
        # within two standard errors of the threshold the data cannot decide
        decided = bool(np.isfinite(se) and abs(v - thr) > 2 * se)
        checks[f"{ty} {kind} by '{rep}'"] = {
            "auroc": v, "se": se, "threshold": thr,
            "verdict": ("held" if ok else "failed") if decided else "inconclusive",
            "prediction_held": bool(ok)}
    return checks


if __name__ == "__main__":
    run()
