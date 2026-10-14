"""Robustness to rigid motions: GEO with every channel against GPE exactly as
published, on a test set that has been translated, rotated, or both.

    python -m geomob.cli robust --config geomob.toml
    python -m geomob.cli robust --config geomob.toml --search      # tune GEO first

Methods
    GPE (paper)   Liu et al. (KDD'25): fixed frequency ladder (Algorithm 1,
                  [encoders] h and gpe_eps_over_2pi) on raw longitude/latitude,
                  spatial only, no frame, nothing learned in the encoder. The
                  `gpe_learn_freqs` option is ignored here on purpose.
    GEO (full)    every GEO channel: location lattice + displacement (radial and
                  angular) + time + speed, built from [encoders] (learned
                  wavelengths if `learn_lambdas` is set), with its frame
                  re-fitted on the transformed city WITHOUT labels. Per task:
                    similarity  location + displacement + what
                                --similarity-extras says (default: speed).
                                Time is off by default: the two halves of a
                                trip share their timestamps, so the time
                                channel would give the answer away
                    eta         location + displacement + departure time (no
                                speed: it would encode the answer)
                    mode        location + displacement + time + speed

    with --diagnostics, two more rows that cost at most one extra training:
    GEO (full, fixed frame)   the same trained GEO model, frame NOT re-fitted
    GPE (paper) in GEO frame  GPE trained and tested in GEO's re-fitted frame

Transformations of the test set (`--scope`)
    city   (default) the whole city is moved as one rigid body, as a deployment
           to another place would see it. The frame is re-fitted on the moved
           city (training + test trajectories, no labels); scores use the test
           trajectories only.
    trip   every test trajectory is moved on its own: a random direction for
           the shift, a random sign for the rotation (about its own centre).
           A scattered test set has no common frame, so GEO keeps the frame of
           the training city here: this scope measures what the channels
           themselves tolerate.

    translation   shift by each of --shifts-km (city scope: towards
                  --direction-deg, 0 = east)
    rotation      turn by each of --angles degrees
    combined      turn AND shift, each pair of --combined ("km:degrees")

Hyperparameter search (--search)
    The values to try are [search_space] in the config file (same format as
    the `search` command). For every dataset x task x model, each candidate
    GEO configuration is trained on the training split minus a validation
    part (--val-frac; user-disjoint where users exist) and scored on that
    validation part, in place and under every transformation above. The
    score that picks the winner is --select-on: the mean over all of them
    (default), the in-place value, or the worst one. The TEST set is never
    looked at. GPE is not searched: it is the published method, with the
    published hyperparameters. If [search_space] contains [model] keys, only
    GEO's model is tuned (a message says so).

    Trials are logged to <search-dir>/trials.jsonl and the search resumes from
    it; the winner of each setting is written as best_<dataset>_<task>_<model>.toml.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np

from .. import config as _cfg
from ..config import save_results
from ..data import Traj, load_available, relocate_city
from .exp_relocate import TASKS, FramedTokenizer, _mean_runs

SWEEPS = ("translation", "rotation", "combined")
GPE_ROW, GEO_ROW = "GPE (paper)", "GEO (full)"
DIAG_ROWS = ("GEO (full, fixed frame)", "GPE (paper) in GEO frame")


# --------------------------------------------------------------------------
# the two methods
# --------------------------------------------------------------------------
def gpe_paper():
    """GPE as published: fixed ladder on raw lon/lat, spatial only."""
    from .exp_gpe_suite import gpe_encoder
    from .exp_similarity import Tokenizer
    return Tokenizer(loc=gpe_encoder(), name=GPE_ROW)


SIM_EXTRAS = ("none", "speed", "time", "time_speed")


def geo_channels(task_name, sim_extras="speed"):
    """(use time, use speed) for GEO (full) on this task."""
    if task_name == "eta":
        return True, False
    if task_name == "similarity":
        if sim_extras not in SIM_EXTRAS:
            raise SystemExit(f"robust: similarity_extras must be one of {SIM_EXTRAS}")
        return "time" in sim_extras, "speed" in sim_extras
    return True, True


def geo_full(task_name, sim_extras="speed"):
    from .exp_downstream import _geo_disp, _geo_loc
    from .exp_gpe_suite import maybe_learnable_geo, speed_encoder, time_encoder
    use_t, use_s = geo_channels(task_name, sim_extras)
    inner = maybe_learnable_geo(_geo_loc(), _geo_disp(),
                                time=time_encoder() if use_t else None,
                                speed=speed_encoder() if use_s else None,
                                name=GEO_ROW)
    return FramedTokenizer(inner, GEO_ROW)


def describe_geo(task_name, sim_extras="speed"):
    use_t, use_s = geo_channels(task_name, sim_extras)
    parts = ["location", "displacement"]
    if use_t:
        parts.append("departure time" if task_name == "eta" else "time")
    if use_s:
        parts.append("speed")
    return " + ".join(parts)


@contextlib.contextmanager
def using(hp):
    """Run with these hyperparameters, then put the previous ones back."""
    old = copy.deepcopy(_cfg.HP)
    _cfg.HP.clear()
    _cfg.HP.update(copy.deepcopy(hp))
    try:
        yield
    finally:
        _cfg.HP.clear()
        _cfg.HP.update(old)


# --------------------------------------------------------------------------
# transformations
# --------------------------------------------------------------------------
def parse_combined(items):
    """["1:15", "10:30"] or [[1, 15], [10, 30]] -> [(1.0, 15.0), (10.0, 30.0)]"""
    out = []
    for it in items or ():
        try:
            if isinstance(it, str):
                d, a = it.replace(",", ":").split(":")
            else:
                d, a = it
            out.append((float(d), float(a)))
        except (ValueError, TypeError):
            raise SystemExit(f"robust: combined value {it!r} must be 'km:degrees', "
                             "e.g. 10:30")
    return out


def make_sweeps(shifts_km, angles, combined):
    sw = {"translation": [(float(d), 0.0) for d in shifts_km],
          "rotation": [(0.0, float(a)) for a in angles],
          "combined": parse_combined(combined)}
    return {k: v for k, v in sw.items() if v}


def _label(sweep, d, a):
    if sweep == "translation":
        return f"{d:g}"
    if sweep == "rotation":
        return f"{a:g}"
    return f"{d:g} km\n{a:g}\N{DEGREE SIGN}"


def _value(sweep, d, a):
    return d if sweep == "translation" else a if sweep == "rotation" else [d, a]


def move_city(city, ref, d_km, a_deg, direction_deg=0.0):
    phi = np.deg2rad(direction_deg)
    return relocate_city(city, rotate_deg=a_deg,
                         shift_km=(d_km * np.cos(phi), d_km * np.sin(phi)), ref=ref)


def move_trips(trajs, d_km, a_deg, seed):
    """Each trajectory moved on its own (random direction, random turn sign)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in trajs:
        phi = rng.uniform(0, 2 * np.pi)
        sign = rng.choice((-1.0, 1.0))
        m = relocate_city([t], rotate_deg=sign * a_deg,
                          shift_km=(d_km * np.cos(phi), d_km * np.sin(phi)),
                          ref=t.lonlat.mean(0))[0]
        out.append(Traj(m.lonlat, t.t, t.uid, t.label))
    return out


class Mover:
    """Yields (moved city, moved test) for a transformation; the same moved
    data for every method."""

    def __init__(self, train, test, scope, direction_deg=0.0):
        self.city = list(train) + list(test)
        self.n_tr, self.scope, self.direction = len(train), scope, direction_deg
        self.ref = np.concatenate([t.lonlat for t in self.city]).mean(0)
        self._cache = {}

    def __call__(self, d_km, a_deg):
        key = (d_km, a_deg)
        if key not in self._cache:
            if (d_km, a_deg) == (0.0, 0.0):
                moved = self.city
            elif self.scope == "city":
                moved = move_city(self.city, self.ref, d_km, a_deg, self.direction)
            else:
                seed = int(1000 * d_km + a_deg) % (2 ** 31)
                moved = self.city[:self.n_tr] + move_trips(self.city[self.n_tr:],
                                                           d_km, a_deg, seed)
            self._cache = {key: moved}          # keep one: memory
        moved = self._cache[key]
        return moved, moved[self.n_tr:]


# --------------------------------------------------------------------------
# one training per method, scored on every transformation
# --------------------------------------------------------------------------
class Setting:
    """Trained models for one dataset x task x model x seed."""

    def __init__(self, task, kind, seed, base_hp, geo_hp, scope, sim_extras,
                 diagnostics):
        self.task, self.scope = task, scope
        self.base_hp, self.geo_hp = base_hp, geo_hp
        city = list(task.train) + list(task.test)
        self.toks, self.models = {}, {}
        jobs = [("gpe", base_hp, gpe_paper),
                ("geo", geo_hp, lambda: geo_full(task.name, sim_extras))]
        if diagnostics:
            jobs.append(("gpe_frame", base_hp,
                         lambda: FramedTokenizer(gpe_paper(), DIAG_ROWS[1])))
        for key, hp, make in jobs:
            with using(hp):
                tok = make()
                tok.fit(city)
                t0 = time.perf_counter()
                self.models[key] = task.fit(tok, kind, seed)
                self.toks[key] = tok
            print(f"    trained {key:10s} dim={tok.dim:4d} "
                  f"({time.perf_counter() - t0:.0f}s)")
        self.rows = [GPE_ROW, GEO_ROW] + (list(DIAG_ROWS) if diagnostics else [])
        if scope == "trip" and DIAG_ROWS[0] in self.rows:
            self.rows.remove(DIAG_ROWS[0])      # same as GEO (full) in trip scope

    def _framed(self, key, hp, moved_city, moved_test, refit):
        tok, model = self.toks[key], self.models[key]
        with using(hp):
            if not refit:
                return self.task.score(model, tok, moved_test)
            saved = copy.deepcopy(tok.frame)
            try:
                tok.refit(moved_city)
                return self.task.score(model, tok, moved_test)
            finally:
                tok.frame = saved

    def score(self, row, moved_city, moved_test):
        refit = self.scope == "city"
        if row == GPE_ROW:
            with using(self.base_hp):
                return self.task.score(self.models["gpe"], self.toks["gpe"], moved_test)
        if row == GEO_ROW:
            return self._framed("geo", self.geo_hp, moved_city, moved_test, refit)
        if row == DIAG_ROWS[0]:
            return self._framed("geo", self.geo_hp, moved_city, moved_test, False)
        return self._framed("gpe_frame", self.base_hp, moved_city, moved_test, refit)


def evaluate_setting(st, mover, sweeps, rows=None):
    """{"in_place": {row: metrics}, "sweeps": {sweep: {row: [metrics]}}}"""
    rows = rows or st.rows
    city, test = mover(0.0, 0.0)
    out = {"in_place": {r: st.score(r, city, test) for r in rows}, "sweeps": {}}
    for sweep, pairs in sweeps.items():
        res = {r: [] for r in rows}
        for d, a in pairs:
            mc, mt = mover(d, a)
            for r in rows:
                res[r].append(st.score(r, mc, mt))
        out["sweeps"][sweep] = res
    return out


def run_once(task, kind, sweeps, seed, base_hp, geo_hp, scope, direction_deg,
             sim_extras, diagnostics):
    st = Setting(task, kind, seed, base_hp, geo_hp, scope, sim_extras, diagnostics)
    mover = Mover(task.train, task.test, scope, direction_deg)
    ev = evaluate_setting(st, mover, sweeps)
    m = task.metric
    out = {"centre_lonlat": [float(mover.ref[0]), float(mover.ref[1])],
           "metric": m, "higher_is_better": task.higher,
           "n_train": len(task.train), "n_test": len(task.test),
           "token_dims": {GPE_ROW: st.toks["gpe"].dim, GEO_ROW: st.toks["geo"].dim},
           "in_place": ev["in_place"], "sweeps": {}}
    for sweep, pairs in sweeps.items():
        out["sweeps"][sweep] = {
            "values": [_value(sweep, d, a) for d, a in pairs],
            "labels": [_label(sweep, d, a) for d, a in pairs],
            "rows": ev["sweeps"][sweep]}
        print(f"    {sweep:11s} " + "  ".join(
            f"{r.split(' (')[0]}{'' if r in (GPE_ROW, GEO_ROW) else '*'}: "
            + ",".join(f"{c[m]:.3f}" for c in ev["sweeps"][sweep][r])
            for r in st.rows))
    return out


# --------------------------------------------------------------------------
# hyperparameter search on a validation split
# --------------------------------------------------------------------------
def holdout(task, frac=0.2, seed=0, epochs=None):
    """A copy of `task` whose test part is a validation slice of its
    training part (user-disjoint for the supervised tasks)."""
    from . import exp_downstream as D
    v = copy.copy(task)
    v.cfg = dict(task.cfg, **({"epochs": epochs} if epochs else {}))
    if task.name == "similarity":
        n = max(int(frac * len(task.train)), 20)
        n = min(n, task.cfg.get("n_test") or n)
        v.train, v.test = list(task.train[:-n]), list(task.train[-n:])
        return v
    i_tr, i_val = D._split_indices([(t,) for t in task.train], 1.0 - frac, seed)
    if not i_val or not i_tr:
        raise ValueError("too few users to carve out a validation split")
    y = task.y_train_t
    v.train = [task.train[i] for i in i_tr]
    v.test = [task.train[i] for i in i_val]
    v.y_train_t = y[i_tr]
    v.y_test = (y[i_val] * task.sd + task.mu) if task.name == "eta" else y[i_val]
    return v


def _aggregate(raw, metric, higher):
    vals = [raw["in_place"]] + [x for v in raw["sweeps"].values() for x in v]
    return {"in_place": raw["in_place"], "mean": float(np.mean(vals)),
            "worst": float(min(vals) if higher else max(vals))}


def search_geo(task, kind, space, *, method, n_trials, search_seed, sweeps,
               scope, direction_deg, select_on, val_frac, search_epochs,
               search_seeds, sim_extras, log_path, setting, resume=True):
    from .exp_gpe_suite import geo_layout
    from .exp_search import _invalid, _where, make_trials
    base_hp = copy.deepcopy(_cfg.HP)
    val = holdout(task, val_frac, epochs=search_epochs)
    m, hi = task.metric, task.higher
    combos, total = make_trials(space, method, n_trials, search_seed)
    signature = {"setting": setting, "epochs": val.cfg["epochs"],
                 "seeds": list(search_seeds), "val_frac": val_frac, "scope": scope,
                 "sweeps": {k: [list(p) for p in v] for k, v in sweeps.items()},
                 "direction": direction_deg, "sim_extras": sim_extras,
                 "n_val": len(val.test), "base": base_hp}
    sig = json.dumps(signature, sort_keys=True, default=float)
    done = {}
    if resume and log_path.exists():
        for line in open(log_path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("signature") == sig:
                done[json.dumps(r["params"], sort_keys=True)] = r
    print(f"  search ({setting}): {total} combination(s), running {len(combos)} "
          f"+ base on {len(val.train)} train / {len(val.test)} validation; "
          f"select on {select_on} {m}" + (f" ({len(done)} already done)" if done else ""))
    mover = Mover(val.train, val.test, scope, direction_deg)
    trials = []
    for i, params in enumerate([{}] + combos):
        name = "base" if i == 0 else f"trial {i:03d}"
        pk = json.dumps(params, sort_keys=True)
        if pk in done:
            rec = dict(done[pk], name=name)
            trials.append(rec)
            continue
        hp = copy.deepcopy(base_hp)
        for k, v in params.items():
            hp[_where(k)][k] = v
        rec = {"name": name, "params": params, "signature": sig}
        bad = _invalid(hp)
        if bad:
            rec.update(status="invalid", reason=bad)
        else:
            rec["geo_layout"] = geo_layout(None, hp["encoders"])["text"]
            t0 = time.perf_counter()
            try:
                per_seed = []
                for s in search_seeds:
                    with using(hp):
                        tok = geo_full(task.name, sim_extras)
                        tok.fit(mover.city)
                        model = val.fit(tok, kind, s)
                        rec["token_dim"] = tok.dim

                        class _St:      # minimal Setting for evaluate_setting
                            rows = [GEO_ROW]

                            @staticmethod
                            def score(row, mc, mt, tok=tok, model=model):
                                if scope == "city":
                                    saved = copy.deepcopy(tok.frame)
                                    tok.refit(mc)
                                    try:
                                        return val.score(model, tok, mt)
                                    finally:
                                        tok.frame = saved
                                return val.score(model, tok, mt)
                        ev = evaluate_setting(_St, mover, sweeps)
                    per_seed.append({"in_place": ev["in_place"][GEO_ROW][m],
                                     "sweeps": {k: [c[m] for c in v[GEO_ROW]]
                                                for k, v in ev["sweeps"].items()}})
                raw = {"in_place": float(np.mean([p["in_place"] for p in per_seed])),
                       "sweeps": {k: list(np.mean([p["sweeps"][k] for p in per_seed], 0))
                                  for k in per_seed[0]["sweeps"]}}
                rec.update(status="ok", raw=raw)
            except Exception as e:  # noqa: BLE001 - one bad trial must not end the search
                import traceback
                traceback.print_exc()
                rec.update(status="failed", reason=f"{type(e).__name__}: {e}")
            rec["seconds"] = round(time.perf_counter() - t0, 1)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        trials.append(rec)
        if rec["status"] == "ok":
            a = _aggregate(rec["raw"], m, hi)
            print(f"    [{name}] {m}: in place {a['in_place']:.4f}  mean {a['mean']:.4f}"
                  f"  worst {a['worst']:.4f}  ({rec['seconds']:.0f}s)  {params}")
        else:
            print(f"    [{name}] {rec['status']}: {rec.get('reason')}  {params}")

    ok = [t for t in trials if t["status"] == "ok"]
    if not ok:
        raise RuntimeError(f"search ({setting}): no trial finished")
    for t in ok:
        t.update(_aggregate(t["raw"], m, hi))
        t["value"] = t[select_on]
    ranked = sorted(ok, key=lambda t: t["value"], reverse=hi)
    best = ranked[0]
    best_hp = copy.deepcopy(base_hp)
    for k, v in best["params"].items():
        best_hp[_where(k)][k] = v
    print(f"  -> best {best['name']} ({select_on} {m} = {best['value']:.4f}; "
          f"base {next((t['value'] for t in ok if t['name'] == 'base'), float('nan')):.4f})"
          f"  {best['params']}")
    for t in trials:
        t.pop("signature", None)
    info = {"select_on": select_on, "metric": m, "higher_is_better": hi,
            "n_val": len(val.test), "n_train": len(val.train),
            "epochs": val.cfg["epochs"], "seeds": list(search_seeds),
            "n_combinations": total, "best": {k: best[k] for k in
                                              ("name", "params", "value", "in_place",
                                               "mean", "worst", "geo_layout")
                                              if k in best},
            "trials": [{k: t.get(k) for k in ("name", "params", "status", "reason",
                                                "value", "in_place", "mean", "worst",
                                                "geo_layout", "token_dim", "seconds")}
                       for t in trials]}
    return best_hp, info


def _write_search_files(search_dir, setting, info, base_config, best_hp, space):
    from .exp_search import _write_best
    safe = setting.replace("/", "_")
    _write_best(search_dir / f"best_{safe}.toml", base_config or {}, best_hp,
                info["best"], f"{info['select_on']} {info['metric']}")
    path = search_dir / "search_results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["setting", "name", "status", "selected_on", "value",
                        "in_place", "mean", "worst", "params", "geo_layout",
                        "token_dim", "seconds"])
        for t in info["trials"]:
            w.writerow([setting, t["name"], t["status"], info["select_on"],
                        t.get("value"), t.get("in_place"), t.get("mean"),
                        t.get("worst"), json.dumps(t["params"]),
                        t.get("geo_layout", ""), t.get("token_dim"), t.get("seconds")])


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(datasets=("tdrive", "porto", "roma", "ais", "geolife"), paths=None,
        tasks=("similarity", "eta", "mode"), kinds=("lstm",), scope="city",
        shifts_km=(1.0, 10.0, 100.0, 1000.0), angles=(15.0, 30.0, 45.0, 90.0),
        combined=("1:15", "10:30", "100:45", "1000:90"), direction_deg=0.0,
        diagnostics=False, similarity_extras="speed",
        n_train=None, n_test=2000, max_traj=20000, epochs=5, bs=128, lr=1e-3,
        seeds=(0,), device="cpu", out=None,
        search=False, space=None, base_config=None, search_method="grid",
        n_trials=20, search_seed=0, search_epochs=None, search_seeds=(0,),
        select_on="mean", val_frac=0.2, search_dir=None, resume=True):
    if scope not in ("city", "trip"):
        raise SystemExit("robust: scope must be city or trip")
    if select_on not in ("mean", "in_place", "worst"):
        raise SystemExit("robust: select_on must be mean, in_place or worst")
    sweeps = make_sweeps(shifts_km, angles, combined)
    if not sweeps:
        raise SystemExit("robust: no transformation to test (shifts_km, angles "
                         "and combined are all empty)")
    if search:
        from .exp_search import check_space, _where
        space = check_space(space or {})
        if any(_where(k) == "model" for k in space):
            print("  note: [search_space] has [model] keys; only GEO's model is "
                  "tuned, GPE keeps the base architecture")
        search_dir = Path(search_dir or (Path(out).with_suffix("").as_posix() + "_search"
                                         if out else "results/robust_search"))
        search_dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            for f in ("trials.jsonl", "search_results.csv"):
                (search_dir / f).unlink(missing_ok=True)
        (search_dir / "search_results.csv").unlink(missing_ok=True)

    cfg = dict(epochs=epochs, bs=bs, lr=lr, device=device, n_train=n_train,
               n_test=n_test)
    raw, skipped = load_available(
        datasets, paths or {},
        lambda n: ({"n": (n_train or 4 * n_test) + n_test} if n.startswith("synthetic")
                   else {"max_traj": max_traj}),
        required=1, what="robust")
    base_hp = copy.deepcopy(_cfg.HP)
    rows = [GPE_ROW, GEO_ROW] + (list(DIAG_ROWS) if diagnostics else [])
    if diagnostics and scope == "trip":
        rows.remove(DIAG_ROWS[0])     # GEO keeps its frame anyway in trip scope
    res = {"robust": True, "rows": rows, "kinds": list(kinds), "tasks": list(tasks),
           "scope": scope, "direction_deg": direction_deg,
           "units": {"translation": "km" + (f", towards {direction_deg:g} deg from east"
                                            if scope == "city" and direction_deg
                                            else ", east" if scope == "city"
                                            else ", random direction per trip"),
                     "rotation": "degrees" + (" about the city centre" if scope == "city"
                                              else " about each trip's centre, random sign"),
                     "combined": "km and degrees"},
           "methods": {GPE_ROW: "GPE as published: fixed frequency ladder on raw "
                                "lon/lat, spatial only, no frame",
                       GEO_ROW: {t: describe_geo(t, similarity_extras) for t in tasks}},
           "search": bool(search), "datasets": {}, "skipped_datasets": skipped,
           "skipped_tasks": {}, "seeds": list(seeds)}
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
                setting = f"{name}/{tname}/{kind}"
                print(f"[robust] {setting}: {len(task.train)} train, "
                      f"{len(task.test)} test; GEO = {describe_geo(tname, similarity_extras)}")
                geo_hp, info = base_hp, None
                if search:
                    try:
                        geo_hp, info = search_geo(
                            task, kind, space, method=search_method,
                            n_trials=n_trials, search_seed=search_seed,
                            sweeps=sweeps, scope=scope, direction_deg=direction_deg,
                            select_on=select_on, val_frac=val_frac,
                            search_epochs=search_epochs, search_seeds=search_seeds,
                            sim_extras=similarity_extras,
                            log_path=search_dir / "trials.jsonl", setting=setting,
                            resume=resume)
                        _write_search_files(search_dir, setting, info, base_config,
                                            geo_hp, space)
                    except (ValueError, RuntimeError) as e:
                        print(f"  !! search skipped for {setting}: {e}; using the "
                              "base settings")
                        info = {"error": str(e)}
                runs = [run_once(task, kind, sweeps, s, base_hp, geo_hp, scope,
                                 direction_deg, similarity_extras, diagnostics)
                        for s in seeds]
                d = _mean_runs(runs)
                d["geo_settings"] = {k: v for k, v in geo_hp["encoders"].items()
                                     if v != base_hp["encoders"].get(k)}
                d["geo_settings"].update({k: v for k, v in geo_hp["model"].items()
                                          if v != base_hp["model"].get(k)})
                from .exp_gpe_suite import geo_layout
                with using(geo_hp):
                    d["geo_layout"] = geo_layout(None, geo_hp["encoders"])["text"]
                d["geo_channels"] = describe_geo(tname, similarity_extras)
                if info is not None:
                    d["search"] = info
                res["datasets"].setdefault(name, {}).setdefault(kind, {})[tname] = d
                save_results(res, out)
    if not res["datasets"]:
        raise FileNotFoundError("robust: no usable dataset / task")
    _summary(res)
    save_results(res, out)
    return res


def _summary(res):
    print(f"\n==== robustness ({res['scope']} scope): in place -> worst value "
          "along each transformation ====")
    for name, by_kind in res["datasets"].items():
        for kind, by_task in by_kind.items():
            for tname, d in by_task.items():
                m, hi = d["metric"], d["higher_is_better"]
                print(f"[{name} / {tname} / {kind}]  {m} "
                      f"({'higher' if hi else 'lower'} is better); GEO: "
                      f"{d['geo_channels']}; {d['geo_layout']}"
                      + (f"; tuned: {d['geo_settings']}" if d.get("geo_settings") else ""))
                for r in res["rows"]:
                    parts = [f"{sw} {(min if hi else max)(c[m] for c in s['rows'][r]):.3f}"
                             for sw, s in d["sweeps"].items()]
                    print(f"  {r:26s} in place {d['in_place'][r][m]:.3f} | "
                          + "  ".join(parts))
