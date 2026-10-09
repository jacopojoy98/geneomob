"""Relocated-city test: what happens to a trained model when the same city is
somewhere else on the globe, or turned.

    python -m geomob.cli relocate --config geomob.toml

For each dataset a model is trained once per method, on the city where it is.
The TEST set is then moved as one rigid body (`data.relocate_city`) -- every
distance, angle, duration and relative position on the ground is unchanged --
along three sweeps:

    latitude     the city centred at 0, 20, 40, 60, 70 degrees north
    rotation     the city turned by 0 ... 90 degrees about its centre
    translation  the city moved east by 0 ... 1000 km

and each task is re-scored. A representation that respects the symmetries of
the plane gives flat curves; one tied to raw longitude/latitude need not,
because a degree of longitude is a different number of metres at each
latitude and because rotating a city mixes its two coordinates.

Tasks
    similarity   GPE's own task (are two halves of a trip still neighbours?).
                 This survives a rigid move with almost any smooth encoding,
                 so expect little separation: it is the control.
    eta          travel time from the path. Supervised: the model has learned
                 a mapping from tokens to minutes, and the answer does not
                 depend on where the city is or how it is turned.
    mode         transport mode (GeoLife labels). Same argument: walking is
                 walking at any latitude and in any direction.

Rows (all evaluated zero-shot: no labels, no gradient steps on the new city):

    GPE                      as trained
    GPE + space shift        GPE paper's SS remap of the new bounding box onto
                             the training one (their own remedy)
    GEO (fixed frame)        GEO with its frame left at the training city
    GEO (origin re-fit)      frame origin re-fit on the new city (unsupervised)
    GEO (canonical frame)    origin AND orientation re-fit: the frame is the
                             centroid and principal axes of the point cloud
    GPE in canonical frame   diagnostic: GPE fed the same canonical-frame
                             coordinates, to separate what the FRAME buys from
                             what the lattice buys

The canonical-frame rows are trained in their own frame (one extra training
each). Because the relocated test set is an exact rigid copy, its canonical
coordinates are identical at every sweep value, so those rows are flat *by
construction*: the experiment verifies the invariance rather than estimating
it, and what it measures is how much the other rows lose.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from ..config import save_results
from ..data import Traj, bbox, load_available, relocate_city, space_shift
from ..geo import enu_to_lonlat, lonlat_to_enu, rot2

ROWS = ("GPE", "GPE + space shift", "GEO (fixed frame)", "GEO (origin re-fit)",
        "GEO (canonical frame)", "GPE in canonical frame")
SWEEPS = ("latitude", "rotation", "translation")


# --------------------------------------------------------------------------
# canonical frame
# --------------------------------------------------------------------------
class CanonicalFrame:
    """Origin = centroid of the point cloud; x-axis = its first principal
    axis, pointed towards the heavier tail (third moment) so the 180-degree
    ambiguity is resolved from the data alone. Fitted without labels."""

    def fit(self, trajs):
        ll = np.concatenate([t.lonlat for t in trajs])
        self.ref = ll.mean(0)
        xy = lonlat_to_enu(ll, self.ref)[0]
        xy = xy - xy.mean(0)
        w, v = np.linalg.eigh(np.cov(xy.T))
        ax = v[:, np.argmax(w)]
        ang = float(np.arctan2(ax[1], ax[0]))
        if np.mean((xy @ ax) ** 3) < 0:
            ang += np.pi
        # second axis: right-handed, also pointed at its heavier tail is not
        # possible without a reflection, so only the first axis is signed
        self.angle = ang
        return self

    def to_pseudo_lonlat(self, lonlat):
        """Canonical-frame metres, written as lon/lat about (0, 0) so that any
        encoder can consume them (at the equator a degree is the same length
        on both axes, so this is metric-faithful)."""
        xy = lonlat_to_enu(lonlat, self.ref)[0] @ rot2(-self.angle).T
        return enu_to_lonlat(xy, np.zeros(2))


class FramedTokenizer:
    """An encoder applied in the canonical frame of whatever corpus it was
    last fitted on. `refit` moves the frame to a new corpus; the encoder and
    the trained model are untouched."""

    def __init__(self, inner, name):
        self.inner, self.name, self.frame = inner, name, CanonicalFrame()

    @property
    def dim(self):
        return self.inner.dim

    def fit(self, trajs, force=False):
        self.frame.fit(trajs)
        if hasattr(self.inner.loc, "ref_deg"):
            self.inner.loc.ref_deg = np.zeros(2)
        if hasattr(self.inner, "ref_rad"):      # learnable GPE: precision anchor
            self.inner.fit([Traj(self.frame.to_pseudo_lonlat(t.lonlat), t.t)
                            for t in trajs], force=True)
        return self

    def refit(self, trajs):
        self.frame.fit(trajs)
        return self

    def __call__(self, traj):
        return self.inner(Traj(self.frame.to_pseudo_lonlat(traj.lonlat),
                               traj.t, traj.uid))

    def __getattr__(self, n):
        # expose a learnable inner tokenizer's front end / raw columns
        if n in ("inner", "frame", "name") or n.startswith("__"):
            raise AttributeError(n)
        if n in ("frontend", "raw_cols"):
            return getattr(self.inner, n)
        raise AttributeError(n)


# --------------------------------------------------------------------------
# tasks: prepare data, train once, score on a (possibly moved) test set
# --------------------------------------------------------------------------
class SimilarityTask:
    name, metric, higher = "similarity", "KP@10", True

    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self, trajs):
        from .exp_gpe_suite import split_gpe
        self.train, self.test = split_gpe(trajs, self.cfg["n_train"], self.cfg["n_test"])
        return self

    def fit(self, tok, kind, seed):
        from .exp_similarity import train_contrastive
        c = self.cfg
        return train_contrastive(tok, self.train, kind=kind, epochs=c["epochs"],
                                 bs=c["bs"], lr=c["lr"], device=c["device"], seed=seed)

    def score(self, model, tok, moved):
        from .exp_similarity import evaluate
        return evaluate(model, tok, moved, self.cfg["device"])


class _Supervised:
    """Shared by ETA and mode: standardise tokens, train a head, predict."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, tok, kind, seed):
        import torch
        from . import exp_downstream as D
        c = self.cfg
        S = D.Standardiser().fit([tok(t) for t in self.train],
                                 getattr(tok, "raw_cols", None))
        X = S([tok(t) for t in self.train])
        model = D._head(tok, self.out_dim, kind)
        y = torch.from_numpy(self.y_train_t)
        D._fit(model, X, lambda M, x, m, b: self.loss(M(x, m), y[b].to(x.device)),
               c["epochs"], min(c["bs"], 64), c["lr"], c["device"], seed)
        return (model, S)

    def score(self, state, tok, moved):
        from . import exp_downstream as D
        model, S = state
        out = D._apply(model, S([tok(t) for t in moved]), self.cfg["device"])
        return self.metrics(out)


class EtaTask(_Supervised):
    name, metric, higher, out_dim = "eta", "MAE", False, 1

    def prepare(self, trajs):
        import inspect
        from . import exp_downstream as D
        sig = inspect.signature(D.run).parameters
        step, mx = sig["eta_step_m"].default, sig["eta_max_min"].default
        items = []
        for t in trajs:
            dur = (t.t[-1] - t.t[0]) * 60.0
            if not (0.5 <= dur <= mx):
                continue
            r = D.resample_by_distance(t, step)
            if r is not None:
                items.append((r[0], dur))
        if len(items) < 50:
            raise ValueError(f"only {len(items)} usable paths for travel time")
        tr, te = D._split_indices(items, seed=0)
        te = te[:self.cfg["n_test"]]
        self.train, self.test = [items[i][0] for i in tr], [items[i][0] for i in te]
        ytr = np.array([items[i][1] for i in tr])
        self.y_test = np.array([items[i][1] for i in te])
        self.mu, self.sd = float(ytr.mean()), float(ytr.std() + 1e-6)
        self.y_train_t = ((ytr - self.mu) / self.sd).astype(np.float32)
        return self

    def loss(self, out, y):
        import torch.nn.functional as F
        return F.mse_loss(out.squeeze(-1), y)

    def metrics(self, out):
        from .exp_downstream import regression_metrics
        return regression_metrics(out.squeeze(-1) * self.sd + self.mu, self.y_test)


class ModeTask(_Supervised):
    name, metric, higher = "mode", "macro_F1", True

    def prepare(self, trajs):
        from . import exp_downstream as D
        trajs = [t for t in trajs if t.label and t.label != "other"]
        if len(trajs) < 50:
            raise ValueError("no mode labels in this corpus (GeoLife has them)")
        classes = sorted({t.label for t in trajs})
        ix = {c: i for i, c in enumerate(classes)}
        self.train, test = D.split_user_disjoint(trajs, seed=0)
        self.test = test[:self.cfg["n_test"]]
        self.out_dim = len(classes)
        self.y_train_t = np.array([ix[t.label] for t in self.train]).astype(np.int64)
        self.y_test = np.array([ix[t.label] for t in self.test])
        return self

    def loss(self, out, y):
        import torch.nn.functional as F
        return F.cross_entropy(out, y)

    def metrics(self, out):
        from .exp_downstream import classification_metrics
        return classification_metrics(out, self.y_test, self.out_dim)[0]


TASKS = {"similarity": SimilarityTask, "eta": EtaTask, "mode": ModeTask}


# --------------------------------------------------------------------------
# one dataset x task
# --------------------------------------------------------------------------
def _moved(test, ref, sweep, value):
    if sweep == "latitude":
        return relocate_city(test, lat=value, ref=ref)
    if sweep == "rotation":
        return relocate_city(test, rotate_deg=value, ref=ref)
    return relocate_city(test, shift_km=(value, 0.0), ref=ref)


def _encoders(task_name, channels):
    """Fresh (GPE, GEO) tokenizers: spatial only, or with time and speed."""
    if channels == "ts" and task_name != "similarity":
        from .exp_downstream import make_tokenizer
        return (lambda: make_tokenizer("gpe_ts", task_name),
                lambda: make_tokenizer("geo_ts", task_name))
    from .exp_gpe_suite import make_encoder
    return (lambda: make_encoder("gpe"), lambda: make_encoder("geo"))


def run_task(task, kind, sweeps, seed, channels="spatial"):
    train, test = task.train, task.test
    mk_gpe, mk_geo = _encoders(task.name, channels)
    toks = {"gpe": mk_gpe(), "geo": mk_geo(),
            "geo_canon": FramedTokenizer(mk_geo(), "GEO (canonical frame)"),
            "gpe_canon": FramedTokenizer(mk_gpe(), "GPE in canonical frame")}
    models = {}
    for k, tok in toks.items():
        tok.fit(train)
        t0 = time.perf_counter()
        models[k] = task.fit(tok, kind, seed)
        print(f"    trained {k:10s} ({time.perf_counter() - t0:.0f}s)")

    ref = np.concatenate([t.lonlat for t in test]).mean(0)
    train_box = bbox(train)

    def score(row, moved):
        if row == "GPE":
            return task.score(models["gpe"], toks["gpe"], moved)
        if row == "GPE + space shift":
            ss = space_shift(moved, bbox(moved), train_box)
            ss = [Traj(a.lonlat, a.t, b.uid, b.label) for a, b in zip(ss, moved)]
            return task.score(models["gpe"], toks["gpe"], ss)
        if row == "GEO (fixed frame)":
            return task.score(models["geo"], toks["geo"], moved)
        if row == "GEO (origin re-fit)":
            t2 = copy.deepcopy(toks["geo"])
            t2.loc.ref_deg = np.concatenate([t.lonlat for t in moved]).mean(0)
            return task.score(models["geo"], t2, moved)
        key = "geo_canon" if row.startswith("GEO") else "gpe_canon"
        return task.score(models[key], copy.deepcopy(toks[key]).refit(moved), moved)

    m = task.metric
    out = {"centre_lonlat": [float(ref[0]), float(ref[1])], "metric": m,
           "higher_is_better": task.higher, "n_train": len(train),
           "n_test": len(test), "sweeps": {}}
    for sweep, values in sweeps.items():
        res = {"values": list(values), "rows": {r: [] for r in ROWS}}
        for v in values:
            moved = _moved(test, ref, sweep, v)
            for r in ROWS:
                res["rows"][r].append(score(r, moved))
            print(f"    {sweep:11s} {v:>7g}  {m}: " + "  ".join(
                f"{r.split(' (')[0][:3]}{'*' if 'canon' in r else ''}"
                f"={res['rows'][r][-1][m]:.3f}" for r in ROWS))
        out["sweeps"][sweep] = res
    out["in_place"] = {r: score(r, test) for r in ROWS}   # untouched test set
    return out


def _mean_runs(runs):
    """Average the metric dicts over seeds (and keep the spread)."""
    def avg(cells):
        keys = [k for k, v in cells[0].items() if isinstance(v, (int, float))]
        d = {k: float(np.mean([c[k] for c in cells])) for k in keys}
        if len(cells) > 1:
            d.update({k + "_std": float(np.std([c[k] for c in cells])) for k in keys})
        return d
    out = copy.deepcopy(runs[0])
    for sweep, res in out["sweeps"].items():
        for r in res["rows"]:
            res["rows"][r] = [avg([x["sweeps"][sweep]["rows"][r][i] for x in runs])
                              for i in range(len(res["values"]))]
    for r in out["in_place"]:
        out["in_place"][r] = avg([x["in_place"][r] for x in runs])
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(datasets=("tdrive", "porto", "roma", "ais", "geolife"), paths=None,
        tasks=("similarity", "eta", "mode"), kinds=("lstm",),
        channels="spatial",
        latitudes=(0.0, 20.0, 40.0, 60.0, 70.0),
        angles=(0.0, 15.0, 30.0, 45.0, 60.0, 90.0),
        shifts_km=(0.0, 1.0, 10.0, 100.0, 1000.0),
        n_train=None, n_test=2000, max_traj=20000, epochs=5, bs=128, lr=1e-3,
        seeds=(0,), device="cpu", out=None):
    cfg = dict(epochs=epochs, bs=bs, lr=lr, device=device, n_train=n_train,
               n_test=n_test)
    sweeps = {"latitude": tuple(latitudes), "rotation": tuple(angles),
              "translation": tuple(shifts_km)}
    sweeps = {k: v for k, v in sweeps.items() if len(v)}
    raw, skipped = load_available(
        datasets, paths or {},
        lambda n: ({"n": (n_train or 4 * n_test) + n_test} if n.startswith("synthetic")
                   else {"max_traj": max_traj}),
        required=1, what="relocate")
    res = {"relocate": True, "rows": list(ROWS), "kinds": list(kinds),
           "tasks": list(tasks), "channels": channels,
           "units": {"latitude": "degrees north", "rotation": "degrees",
                     "translation": "km east"},
           "datasets": {}, "skipped_datasets": skipped, "skipped_tasks": {},
           "seeds": list(seeds)}
    for name, trajs in raw.items():
        for tname in tasks:
            data = trajs
            try:
                if tname == "mode" and name == "geolife":
                    from ..data import load_dataset
                    data = load_dataset("geolife", (paths or {}).get("geolife"),
                                        max_traj=max_traj, labelled_only=True)
                task = TASKS[tname](cfg).prepare(data)
            except (ValueError, FileNotFoundError) as e:
                res["skipped_tasks"][f"{name}/{tname}"] = str(e)
                print(f"  !! SKIPPED {tname} on {name}: {e}")
                continue
            for kind in kinds:
                print(f"[relocate] {name} / {tname} / {kind}: "
                      f"{len(task.train)} train, {len(task.test)} test")
                runs = [run_task(task, kind, sweeps, s, channels) for s in seeds]
                res["datasets"].setdefault(name, {}).setdefault(kind, {})[tname] = \
                    _mean_runs(runs)
                save_results(res, out)
    if not res["datasets"]:
        raise FileNotFoundError("relocate: no usable dataset / task")
    _summary(res)
    save_results(res, out)
    return res


def _summary(res):
    print("\n==== relocated-city test: in place -> worst value along each sweep ====")
    for name, by_kind in res["datasets"].items():
        for kind, by_task in by_kind.items():
            for tname, d in by_task.items():
                m, hi = d["metric"], d["higher_is_better"]
                print(f"[{name} / {tname} / {kind}]  {m} "
                      f"({'higher' if hi else 'lower'} is better)")
                for r in res["rows"]:
                    parts = [f"{sw} {(min if hi else max)(c[m] for c in s['rows'][r]):.3f}"
                             for sw, s in d["sweeps"].items()]
                    print(f"  {r:24s} in place {d['in_place'][r][m]:.3f} | "
                          + "  ".join(parts))
