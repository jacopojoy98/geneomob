"""Hyperparameter search for the GEO encoder (and, optionally, the model).

    python -m geomob.cli search --config geomob.toml

The values to try are listed in the config file:

    [search_space]                       # any [encoders] or [model] key
    geo_loc_lam_min_m = [10, 25, 50]
    geo_loc_lam_max_m = [50000, 200000]
    geo_harmonics     = [[1], [1, 4], [1, 2, 4]]

and how to search in `[search]` (objective, metric, grid or random, ...).

Selection is done on a VALIDATION split carved out of the training data. The
test data that `gpe_suite` / `downstream` report on is never looked at, so the
numbers in the paper are not tuned on their own test set.

    similarity   for each dataset: [ ... train ... | n_val | n_test ]
                 the final n_test is the gpe_suite test set (untouched), the
                 n_val before it is what trials are scored on
    tul/eta/mode the task's own train/test split is made first, the test part
                 is dropped, and the training part is split again the same way

Outputs, in --out-dir:
    trials.jsonl          one line per finished trial (the search resumes
                          from this file if interrupted)
    search_results.json   every trial + the run record
    search_results.csv    the same as a table, best first
    best.toml             the input config with the best values written into
                          [encoders] / [model]; pass it to run_all
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .. import config as _cfg

# metric -> True if larger is better
SIM_METRICS = {"MRR": True, "MP": True, "KP@10": True, "MR": False,
               "global_MRR": True}
TASK_METRICS = {"tul": {"acc@1": True, "acc@5": True, "macro_F1": True},
                "eta": {"MAE": False, "RMSE": False, "MAPE": False},
                "mode": {"acc@1": True, "macro_F1": True}}
DEFAULT_METRIC = {"similarity": "KP@10", "tul": "acc@1", "eta": "MAE",
                  "mode": "macro_F1"}


# --------------------------------------------------------------------------
# search space
# --------------------------------------------------------------------------
def _where(key):
    for sec in ("encoders", "model"):
        if key in _cfg.HP[sec]:
            return sec
    valid = list(_cfg.HP["encoders"]) + list(_cfg.HP["model"])
    _cfg._unknown("key", key, valid, "[search_space]")


def check_space(space: dict) -> dict:
    if not space:
        raise SystemExit(
            "search: no [search_space] section in the config. Add one, e.g.\n"
            "  [search_space]\n  geo_loc_lam_min_m = [10, 25, 50]\n"
            "  geo_harmonics = [[1], [1, 4]]")
    out = {}
    for k, vals in space.items():
        k = k.replace("-", "_")
        _where(k)
        if not isinstance(vals, list) or not vals:
            raise SystemExit(f"[search_space] {k} must be a non-empty list of "
                             f"candidate values, got {vals!r}")
        if k == "geo_harmonics" and not all(isinstance(v, list) for v in vals):
            raise SystemExit("[search_space] geo_harmonics must be a list of "
                             "lists, e.g. [[1], [1, 4], [1, 2, 4]]")
        out[k] = vals
    return out


def _invalid(hp):
    """Reason this combination cannot be built, or None."""
    e, m = hp["encoders"], hp["model"]
    if e["geo_loc_lam_min_m"] >= e["geo_loc_lam_max_m"]:
        return "geo_loc_lam_min_m >= geo_loc_lam_max_m"
    if e["geo_disp_lam_min_m"] >= e["geo_disp_lam_max_m"]:
        return "geo_disp_lam_min_m >= geo_disp_lam_max_m"
    if e["speed_lo"] >= e["speed_hi"]:
        return "speed_lo >= speed_hi"
    if e["h"] % 8:
        return "h must be a multiple of 8"
    try:
        from .exp_gpe_suite import geo_layout
        geo_layout(None, e)
    except ValueError as err:
        return str(err)
    if m["hidden"] % m["heads"]:
        return "hidden must be divisible by heads"
    return None


def make_trials(space, method, n_trials, seed):
    keys = list(space)
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*(space[k] for k in keys))]
    total = len(combos)
    if method == "random" and n_trials < total:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in sorted(rng.choice(total, n_trials, replace=False))]
    return combos, total


def _key(params):
    return json.dumps(params, sort_keys=True)


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------
def load_similarity(datasets, paths, n_train, n_val, n_test, max_traj):
    """name -> (train, validation). The last n_test of each corpus is the
    gpe_suite test set and is dropped here."""
    from ..data import load_available
    raw, skipped = load_available(
        datasets, paths, lambda n: ({"n": n_train + n_val + n_test}
                                    if n.startswith("synthetic")
                                    else {"max_traj": max_traj}),
        required=0, what="search")
    data = {}
    for name, trajs in raw.items():
        dev = trajs[:-n_test] if n_test else trajs
        if len(dev) <= n_val + 50:
            skipped[name] = (f"{len(trajs)} trajectories: too few for "
                             f"n_test={n_test} + n_val={n_val} + training")
            print(f"  !! SKIPPED dataset '{name}': {skipped[name]}")
            continue
        train, val = dev[:-n_val], dev[-n_val:]
        train = train[-n_train:] if n_train else train
        data[name] = (train, val)
        print(f"  {name}: {len(train)} train / {len(val)} validation "
              f"({n_test} test held out, not used)")
    if not data:
        raise FileNotFoundError(
            "search: no usable dataset.\n" + "\n".join(
                f"    {n}: {r}" for n, r in skipped.items()))
    return data, skipped


def similarity_objective(data, key, kind, tcfg, seeds):
    """All similarity metrics for the current HP, averaged over datasets and
    seeds. `key` is 'geo' or 'gpe'."""
    from .exp_gpe_suite import train_and_evaluate
    per, dim = {}, None
    for src in data:
        reps = [train_and_evaluate(key, src, data, kind, tcfg, seed=s)
                for s in seeds]
        dim = reps[0]["dim"]
        row = {m: float(np.mean([r["local"][m] for r in reps]))
               for m in ("MR", "MRR", "MP", "KP@10")}
        g = [list(v.values())[-1] for r in reps for v in r["global"].values()]
        if g:
            row["global_MRR"] = float(np.mean(g))
        row["seconds"] = float(np.mean([r["time_s"]["total"] for r in reps]))
        per[src] = row
    mean = {m: float(np.mean([per[d][m] for d in per if m in per[d]]))
            for m in ("MR", "MRR", "MP", "KP@10", "global_MRR", "seconds")
            if any(m in per[d] for d in per)}
    return {"metrics": mean, "per_dataset": per, "token_dim": dim}


def task_objective(task, trajs, enc, kind, dcfg, seeds):
    from . import exp_downstream as D
    fn = {"tul": D.task_tul, "eta": D.task_eta, "mode": D.task_mode}[task]
    grid, _, _ = fn(trajs, (enc,), (kind,), seeds, dcfg)
    v = grid[enc][kind]
    dim = D.make_tokenizer(enc, task).dim
    return {"metrics": {k: float(x) for k, x in v.items()
                        if isinstance(x, (int, float))},
            "per_dataset": {}, "token_dim": dim}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(space=None, base_config=None, objective="similarity", metric=None,
        method="grid", n_trials=20, datasets=("tdrive", "porto", "roma", "ais",
                                              "geolife"),
        paths=None, encoder=None, kind="lstm", epochs=3, bs=128, lr=1e-3,
        n_train=4000, n_val=1000, n_test=5000, max_traj=20000, seeds=(0,),
        search_seed=0, device="cpu", out_dir="results/search", resume=True):
    space = check_space(space or {})
    metric = metric or DEFAULT_METRIC[objective]
    table = SIM_METRICS if objective == "similarity" else TASK_METRICS[objective]
    if metric not in table:
        raise SystemExit(f"search: metric '{metric}' is not available for "
                         f"objective '{objective}'; choose from {list(table)}")
    higher = table[metric]
    encoder = encoder or ("geo" if objective == "similarity" else "geo_ts")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos, total = make_trials(space, method, n_trials, search_seed)
    print(f"search: objective={objective}  metric={metric} "
          f"({'higher' if higher else 'lower'} is better)  encoder={encoder}  "
          f"model={kind}")
    print(f"  search space: {total} combination(s) over {list(space)}; "
          f"running {len(combos)} ({method}) + the current settings as 'base'")

    # ---- data, loaded once -------------------------------------------------
    skipped = {}
    if objective == "similarity":
        data, skipped = load_similarity(datasets, paths or {}, n_train, n_val,
                                        n_test, max_traj)
        if metric == "global_MRR" and len(data) < 2:
            raise SystemExit("search: metric global_MRR needs at least two "
                             "usable datasets")
        tcfg = dict(epochs=epochs, bs=bs, lr=lr, device=device)

        def evaluate_current(key=encoder):
            return similarity_objective(data, key, kind, tcfg, seeds)
    else:
        from ..data import load_dataset
        import inspect
        from . import exp_downstream as D
        kw = {"max_traj": max_traj}
        if objective == "mode":
            kw["labelled_only"] = True
        trajs = load_dataset("geolife", (paths or {}).get("geolife"), **kw)
        sig = inspect.signature(D.run).parameters
        dcfg = {k: sig[k].default for k in
                ("min_trips", "eta_step_m", "eta_max_min", "mode_split",
                 "anomaly_rate", "severity", "z_dim", "k")}
        dcfg.update(epochs=epochs, bs=min(bs, 64), lr=lr, device=device,
                    validation=True)

        def evaluate_current(key=encoder):
            return task_objective(objective, trajs, key, kind, dcfg, seeds)

    base_hp = copy.deepcopy(_cfg.HP)
    done = {}
    log = out_dir / "trials.jsonl"
    signature = {"objective": objective, "kind": kind, "epochs": epochs,
                 "encoder": encoder, "n_train": n_train, "n_val": n_val,
                 "seeds": list(seeds),
                 "datasets": sorted(data) if objective == "similarity" else ["geolife"],
                 "base": base_hp}
    if resume and log.exists():
        for line in open(log):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("signature") == signature:
                done[_key(r["params"]) + r["name"]] = r
        if done:
            print(f"  resuming: {len(done)} trial(s) already in {log}")
    elif log.exists():
        log.unlink()

    def set_hp(params):
        hp = copy.deepcopy(base_hp)
        for k, v in params.items():
            hp[_where(k)][k] = v
        return hp

    def run_trial(name, params, key=encoder):
        ck = _key(params) + name
        if ck in done:
            return done[ck]
        hp = set_hp(params)
        rec = {"name": name, "params": params, "signature": signature}
        bad = _invalid(hp)
        if not bad and key.startswith("geo"):
            from .exp_gpe_suite import geo_layout
            rec["geo_layout"] = geo_layout(None, hp["encoders"])["text"]
        if bad:
            rec.update(status="invalid", reason=bad)
            print(f"  [{name}] skipped (invalid: {bad})  {params}")
        else:
            _cfg.HP.clear()
            _cfg.HP.update(hp)
            t0 = time.perf_counter()
            try:
                r = evaluate_current(key)
                rec.update(status="ok", value=r["metrics"].get(metric), **r)
            except Exception as e:  # noqa: BLE001 - one bad trial must not end the search
                import traceback
                traceback.print_exc()
                rec.update(status="failed", reason=f"{type(e).__name__}: {e}")
            finally:
                _cfg.HP.clear()
                _cfg.HP.update(copy.deepcopy(base_hp))
            rec["seconds"] = round(time.perf_counter() - t0, 1)
            if rec["status"] == "ok":
                print(f"  [{name}] {metric}={rec['value']:.4f}  "
                      f"dim={rec['token_dim']}"
                      + (f" ({rec['geo_layout']})" if rec.get("geo_layout") else "")
                      + f"  ({rec['seconds']:.0f}s)  {params}")
        with open(log, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        done[ck] = rec
        return rec

    # ---- references, then the trials ---------------------------------------
    gpe_key = "gpe" if objective == "similarity" else (
        "gpe_ts" if encoder.endswith("_ts") else "gpe")
    ref_gpe = run_trial("GPE (reference)", {}, key=gpe_key)
    base = run_trial("base", {})
    trials = []
    for i, params in enumerate(combos, 1):
        trials.append(run_trial(f"trial {i:03d}", params))
        left = len(combos) - i
        ok = [t for t in trials if t["status"] == "ok" and t.get("seconds")]
        if ok and left and i % 5 == 0:
            eta = left * float(np.mean([t["seconds"] for t in ok])) / 60
            print(f"  ... {i}/{len(combos)} done, about {eta:.0f} min left")

    ok = [t for t in [base] + trials if t["status"] == "ok" and t["value"] is not None]
    if not ok:
        raise SystemExit("search: no trial finished successfully; see the "
                         "messages above")
    ranked = sorted(ok, key=lambda t: t["value"], reverse=higher)
    best = ranked[0]

    # ---- sensitivity: mean of the metric for each value of each key --------
    marg = {}
    for k in space:
        rows = {}
        for t in trials:
            if t["status"] == "ok" and t["value"] is not None:
                rows.setdefault(json.dumps(t["params"][k]), []).append(t["value"])
        marg[k] = {v: {"mean": float(np.mean(x)), "best": float((max if higher else min)(x)),
                       "n": len(x)} for v, x in rows.items()}

    res = {"objective": objective, "metric": metric, "higher_is_better": higher,
           "encoder": encoder, "kind": kind, "method": method,
           "space": space, "n_combinations": total,
           "datasets": signature["datasets"], "skipped_datasets": skipped,
           "validation": ("n_val before the held-out n_test" if objective == "similarity"
                          else "task split applied twice; test part unused"),
           "reference_gpe": ref_gpe, "base": base, "trials": trials,
           "best": best, "sensitivity": marg}
    for r in [ref_gpe, base] + trials:
        r.pop("signature", None)
    _cfg.save_results(res, out_dir / "search_results.json")
    _write_csv(out_dir / "search_results.csv", ranked, ref_gpe, space, metric,
               base_hp)
    _write_best(out_dir / "best.toml", base_config or {}, set_hp(best["params"]),
                best, metric)

    # ---- report ------------------------------------------------------------
    print(f"\n==== search result ({metric}, "
          f"{'higher' if higher else 'lower'} is better, on validation) ====")
    if ref_gpe["status"] == "ok":
        print(f"  GPE reference : {ref_gpe['value']:.4f}")
    if base["status"] == "ok":
        print(f"  base settings : {base['value']:.4f}")
    print(f"  best          : {best['value']:.4f}  ({best['name']})")
    for k, v in best["params"].items():
        print(f"      {k} = {v}")
    print("  top 5:")
    for t in ranked[:5]:
        print(f"    {t['value']:.4f}  {t['name']:10s} {t['params']}")
    print("  sensitivity (per value: best / mean of the metric over its trials):")
    for k, rows in marg.items():
        print(f"    {k}: " + "   ".join(
            f"{v} -> {d['best']:.4f} / {d['mean']:.4f}" for v, d in rows.items()))
    bad = [t for t in trials if t["status"] != "ok"]
    if bad:
        print(f"  {len(bad)} trial(s) not run: " + "; ".join(
            sorted({t.get('reason', '?') for t in bad})))
    print(f"\n  wrote {out_dir}/search_results.json, search_results.csv, best.toml")
    print(f"  to use the best settings:  python -m geomob.cli run_all "
          f"--config {out_dir}/best.toml")
    return res


def _write_csv(path, ranked, ref, space, metric, base_hp):
    # a row that did not set a key (base, GPE reference) ran with the base value
    base = {k: base_hp[_where(k)][k] for k in space}
    metrics = sorted({m for t in ranked for m in t.get("metrics", {})})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", metric] + list(space)
                   + ["token_dim", "geo_layout", "seconds"]
                   + [f"metric:{m}" for m in metrics])
        rows = [(i, t) for i, t in enumerate(ranked, 1)]
        if ref.get("status") == "ok":
            rows.append(("ref", ref))
        for i, t in rows:
            w.writerow([i, t["name"], t["value"]]
                       + [json.dumps(t["params"].get(k, base[k])) for k in space]
                       + [t.get("token_dim"), t.get("geo_layout", ""), t.get("seconds")]
                       + [t["metrics"].get(m, "") for m in metrics])


def _write_best(path, base_config, hp, best, metric):
    """The input config with the winning [encoders]/[model], minus the search
    sections, so it can be handed straight to run_all / gpe_suite."""
    cfg = {k: copy.deepcopy(v) for k, v in base_config.items()
           if k not in ("search", "search_space", "model", "encoders")}
    lines = [f"# best settings found by `geomob search` "
             f"({metric} = {best['value']:.4f} on validation, {best['name']})",
             f"# changed: {json.dumps(best['params'])}", ""]
    for sec, d in cfg.items():
        if isinstance(d, dict):
            lines.append(_cfg._section(sec, d))
    lines.append(_cfg._section("model", hp["model"]))
    lines.append(_cfg._section("encoders", hp["encoders"]))
    Path(path).write_text("\n".join(lines))
