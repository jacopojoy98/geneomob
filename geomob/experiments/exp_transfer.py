"""Transfer across datasets: GEO with every channel against GPE as published,
trained on one dataset and evaluated on another.

    python -m geomob.cli transfer_suite --config geomob.toml --seeds 0 1 2

Methods (the same two as `robust`)
    GPE (paper)   fixed frequency ladder on raw lon/lat, spatial only, no frame
    GEO (full)    location + displacement + time + speed as the task allows
                  (similarity: + `similarity_extras`, default none; eta:
                  + departure time; mode: + time + speed), in a frame fitted
                  on whatever city it is applied to, without labels

For every task, every ordered pair (source -> target) of usable datasets and
every model:

    in domain     trained on the target's own training split (reference)
    zero-shot     the source model applied to the target's test split. The
                  only adaptation is label-free: GEO re-fits its frame on the
                  target city; with `adapt_norm` (default on) both methods
                  re-fit the per-dimension token standardisation on the
                  target's unlabelled training tokens. Travel times are read
                  out with the SOURCE label scale: no target label is used.
    fine-tuned    the source model trained further on a fraction of the
                  target's training split (`fractions`, e.g. 1 %, 10 %, 100 %);
                  travel times are standardised with that fraction's labels
    from scratch  a fresh model trained on the same fraction of the target
                  (the baseline that says whether the source helped at all)

With --diagnostics the zero-shot step also reports GEO with its frame left at
the source city (same model, no extra training).

Similarity needs no labels, so a "fraction" there is a fraction of the
target's training trajectories. Mode transfer needs mode labels on two
datasets with the same classes (GeoLife alone has them, so it is skipped
unless you add another labelled corpus).

Hyperparameter search (--search)
    A source model must read the target with the same tokenizer, so GEO gets
    ONE configuration per task and model, shared by every dataset. Each
    candidate from [search_space] (same format as `search`) is trained on
    every dataset's training split minus a validation part (`val_frac`,
    user-disjoint where there are users) and scored on that part, in domain.
    Scores are made comparable across datasets by dividing by the base
    configuration's score on the same dataset (inverted for lower-is-better
    metrics), and the candidate with the best mean ratio wins. Test sets are
    never used. GPE is not searched. Files in <out>_search/: trials.jsonl
    (resumable), search_results.csv, best_<task>_<model>.toml.

Cost per task, model and seed: one training per dataset and method (also the
in-domain reference), one per pair x fraction (fine-tuning), one per dataset x
fraction below 100 % (from scratch).
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .. import config as _cfg
from ..config import save_results
from ..data import load_available
from .exp_relocate import TASKS, FramedTokenizer
from .exp_robust import GEO_ROW, GPE_ROW, describe_geo, geo_full, gpe_paper, using

DIAG_ROW = "GEO (full, source frame)"


# --------------------------------------------------------------------------
# one dataset prepared for one task
# --------------------------------------------------------------------------
class Domain:
    def __init__(self, name, task):
        self.name, self.task = name, task
        self.train, self.test = list(task.train), list(task.test)
        self.city = self.train + self.test
        self.y_train = self.y_test = None
        self.classes = getattr(task, "classes", None)
        if task.name == "eta":
            self.y_train = task.y_train_t * task.sd + task.mu       # minutes
            self.y_test = np.asarray(task.y_test, dtype=np.float64)
        elif task.name == "mode":
            self.y_train = np.asarray(task.y_train_t)
            self.y_test = np.asarray(task.y_test)

    def subset(self, frac, seed, min_n=32):
        n = len(self.train)
        if frac >= 1.0:
            return np.arange(n)
        k = int(min(n, max(min_n, round(frac * n))))
        return np.sort(np.random.default_rng(1000 + seed).permutation(n)[:k])


# --------------------------------------------------------------------------
# train / score, one code path per task type
# --------------------------------------------------------------------------
def _tokens(tok, trajs):
    return [tok(t) for t in trajs]


def train(dom, tok, idx, kind, seed, cfg, init=None, adapt_norm=True):
    """Train on dom.train[idx]; from `init` (a state from another dataset)
    when given. Returns a state dict."""
    trajs = [dom.train[i] for i in idx]
    epochs = cfg["ft_epochs"] if init is not None else cfg["epochs"]
    lr = cfg["ft_lr"] if init is not None else cfg["lr"]
    if dom.task.name == "similarity":
        from .exp_similarity import train_contrastive
        model = copy.deepcopy(init["model"]) if init is not None else None
        model = train_contrastive(tok, trajs, kind=kind, epochs=epochs,
                                  bs=min(cfg["bs"], max(len(trajs) // 2, 2)),
                                  lr=lr, device=cfg["device"], seed=seed, model=model)
        return {"model": model}
    import torch
    from . import exp_downstream as D
    raw = getattr(tok, "raw_cols", None)
    X = _tokens(tok, trajs)
    if init is None:
        S = D.Standardiser().fit(X, raw)
        if dom.task.name == "eta":
            y = dom.y_train[idx]
            mu, sd = float(y.mean()), float(y.std() + 1e-6)
            model = D._head(tok, 1, kind, seed)
        else:
            mu = sd = None
            model = D._head(tok, len(dom.classes), kind, seed)
    else:
        model = copy.deepcopy(init["model"])
        mu, sd = init["mu"], init["sd"]
        if dom.task.name == "eta":
            # target labels are in use from here on, so they set the scale
            # (only zero-shot must read out on the source's scale)
            y = dom.y_train[idx]
            mu, sd = float(y.mean()), float(y.std() + 1e-6)
        S = (D.Standardiser().fit(_tokens(tok, dom.train), raw) if adapt_norm
             else init["S"])
    Xs = S(X)
    if dom.task.name == "eta":
        yt = torch.from_numpy(((dom.y_train[idx] - mu) / sd).astype(np.float32))
        loss = lambda M, x, m, b: torch.nn.functional.mse_loss(
            M(x, m).squeeze(-1), yt[b].to(x.device))
    else:
        yt = torch.from_numpy(dom.y_train[idx].astype(np.int64))
        loss = lambda M, x, m, b: torch.nn.functional.cross_entropy(
            M(x, m), yt[b].to(x.device))
    D._fit(model, Xs, loss, epochs, min(cfg["bs"], 64), lr, cfg["device"], seed)
    return {"model": model, "S": S, "mu": mu, "sd": sd}


def score(dom, state, tok, adapt_norm=True):
    """Metrics of `state` on dom.test, read through tokenizer `tok`."""
    if dom.task.name == "similarity":
        from .exp_similarity import evaluate
        return evaluate(state["model"], tok, dom.test, dom.task.cfg["device"])
    from . import exp_downstream as D
    S = state["S"]
    if adapt_norm and state.get("_foreign"):
        S = D.Standardiser().fit(_tokens(tok, dom.train), getattr(tok, "raw_cols", None))
    out = D._apply(state["model"], S(_tokens(tok, dom.test)), dom.task.cfg["device"])
    if dom.task.name == "eta":
        return D.regression_metrics(out.squeeze(-1) * state["sd"] + state["mu"],
                                    dom.y_test)
    return D.classification_metrics(out, dom.y_test, len(dom.classes))[0]


# --------------------------------------------------------------------------
# one task x model x seed
# --------------------------------------------------------------------------
def _tok_for(method, dom, sim_extras, src_tok=None, refit=True):
    """A tokenizer for `dom`. GEO: the source tokenizer (same wavelengths,
    possibly learned) with its frame re-fitted on dom's city."""
    if src_tok is not None:
        t = copy.deepcopy(src_tok)
        if method == "geo" and refit:
            t.refit(dom.city)
        return t
    t = gpe_paper() if method == "gpe" else geo_full(dom.task.name, sim_extras)
    t.fit(dom.city)
    return t


def run_seed(domains, kind, seed, cfg, base_hp, geo_hp, fractions, sim_extras,
             adapt_norm, diagnostics):
    rows = {"gpe": GPE_ROW, "geo": GEO_ROW}
    hp_of = {"gpe": base_hp, "geo": geo_hp}
    names = list(domains)
    out = {"in_domain": {}, "scratch": {}, "pairs": {}}
    states, toks = {}, {}
    # 1) full training on every dataset: the source models and the in-domain
    #    references at once
    for n in names:
        dom = domains[n]
        out["in_domain"][n], out["scratch"][n] = {}, {}
        for m in ("gpe", "geo"):
            with using(hp_of[m]):
                t0 = time.perf_counter()
                tok = _tok_for(m, dom, sim_extras)
                st = train(dom, tok, dom.subset(1.0, seed), kind, seed, cfg)
                states[n, m], toks[n, m] = st, tok
                out["in_domain"][n][rows[m]] = score(dom, st, tok)
                # 2) from scratch on fractions of this dataset
                out["scratch"][n][rows[m]] = []
                for f in fractions:
                    if f >= 1.0:
                        out["scratch"][n][rows[m]].append(out["in_domain"][n][rows[m]])
                        continue
                    t2 = _tok_for(m, dom, sim_extras)
                    s2 = train(dom, t2, dom.subset(f, seed), kind, seed, cfg)
                    out["scratch"][n][rows[m]].append(score(dom, s2, t2))
            print(f"    {n:8s} {m}: in domain {_fmt(out['in_domain'][n][rows[m]], dom)}"
                  f"  ({time.perf_counter() - t0:.0f}s with scratch runs)")
    # 3) every ordered pair
    for src in names:
        for dst in names:
            if src == dst:
                continue
            S, T = domains[src], domains[dst]
            if T.task.name == "mode" and S.classes != T.classes:
                print(f"    {src} -> {dst}: skipped (different mode classes)")
                continue
            key = f"{src}→{dst}"
            pr = {"src": src, "dst": dst, "zero_shot": {}, "finetune": {}}
            for m in ("gpe", "geo"):
                with using(hp_of[m]):
                    tok = _tok_for(m, T, sim_extras, src_tok=toks[src, m])
                    st = dict(states[src, m], _foreign=True)
                    pr["zero_shot"][rows[m]] = score(T, st, tok, adapt_norm)
                    if m == "geo" and diagnostics:
                        t_src = _tok_for(m, T, sim_extras, src_tok=toks[src, m],
                                         refit=False)
                        pr["zero_shot"][DIAG_ROW] = score(T, st, t_src, adapt_norm)
                    pr["finetune"][rows[m]] = []
                    for f in fractions:
                        ft = train(T, tok, T.subset(f, seed), kind, seed, cfg,
                                   init=states[src, m], adapt_norm=adapt_norm)
                        pr["finetune"][rows[m]].append(score(T, ft, tok))
            out["pairs"][key] = pr
            print(f"    {key:18s} zero-shot  GPE {_fmt(pr['zero_shot'][GPE_ROW], T)}"
                  f"  GEO {_fmt(pr['zero_shot'][GEO_ROW], T)}   | fine-tuned "
                  + "  ".join(f"{f:g}: {_fmt(pr['finetune'][GPE_ROW][i], T)}/"
                              f"{_fmt(pr['finetune'][GEO_ROW][i], T)}"
                              for i, f in enumerate(fractions)))
    out["token_dims"] = {GPE_ROW: toks[names[0], "gpe"].dim,
                         GEO_ROW: toks[names[0], "geo"].dim}
    return out


def _fmt(cell, dom):
    m = dom.task.metric
    return f"{cell[m]:.3f}"


# --------------------------------------------------------------------------
# averaging over seeds
# --------------------------------------------------------------------------
def _num(v):
    return isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)


def _mean(runs):
    """Average nested results over seeds; every number gets a `<key>_std`."""
    def walk(xs):
        x0 = xs[0]
        if isinstance(x0, dict):
            out = {k: walk([x[k] for x in xs]) for k in x0}
            if len(xs) > 1:
                for k, v in x0.items():
                    if _num(v):
                        out[k + "_std"] = float(np.std([x[k] for x in xs]))
            return out
        if isinstance(x0, list):
            return [walk([x[i] for x in xs]) for i in range(len(x0))]
        if _num(x0):
            return float(np.mean(xs))
        return x0
    return walk(runs)


# --------------------------------------------------------------------------
# one GEO configuration per task and model, chosen across all datasets
# --------------------------------------------------------------------------
def search_shared(domains, kind, space, *, method, n_trials, search_seed,
                  val_frac, search_epochs, search_seeds, sim_extras, cfg,
                  log_path, setting, resume=True):
    import json
    from .exp_gpe_suite import geo_layout
    from .exp_robust import holdout
    from .exp_search import _invalid, _where, make_trials
    base_hp = copy.deepcopy(_cfg.HP)
    scfg = dict(cfg, epochs=search_epochs or cfg["epochs"])
    vals = {}
    for n, d in domains.items():
        try:
            vals[n] = Domain(n, holdout(d.task, val_frac, epochs=scfg["epochs"]))
        except ValueError as e:
            print(f"  search: {n} left out ({e})")
    if not vals:
        raise RuntimeError("no dataset could provide a validation split")
    some = next(iter(vals.values())).task
    m, hi = some.metric, some.higher
    combos, total = make_trials(space, method, n_trials, search_seed)
    signature = json.dumps({"setting": setting, "datasets": sorted(vals),
                            "epochs": scfg["epochs"], "seeds": list(search_seeds),
                            "val_frac": val_frac, "sim_extras": sim_extras,
                            "n_val": {n: len(v.test) for n, v in vals.items()},
                            "base": base_hp}, sort_keys=True, default=float)
    done = {}
    if resume and log_path.exists():
        for line in open(log_path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("signature") == signature:
                done[json.dumps(r["params"], sort_keys=True)] = r
    print(f"  search ({setting}): {total} combination(s), running {len(combos)} + "
          f"base on " + ", ".join(f"{n} {len(v.train)}/{len(v.test)}"
                                  for n, v in vals.items())
          + f" (train/validation); {m}" + (f"; {len(done)} already done" if done else ""))
    trials = []
    for i, params in enumerate([{}] + combos):
        name = "base" if i == 0 else f"trial {i:03d}"
        key = json.dumps(params, sort_keys=True)
        if key in done:
            trials.append(dict(done[key], name=name))
            continue
        hp = copy.deepcopy(base_hp)
        for k, v in params.items():
            hp[_where(k)][k] = v
        rec = {"name": name, "params": params, "signature": signature}
        bad = _invalid(hp)
        if bad:
            rec.update(status="invalid", reason=bad)
        else:
            rec["geo_layout"] = geo_layout(None, hp["encoders"])["text"]
            t0 = time.perf_counter()
            try:
                per = {}
                with using(hp):
                    for n, v in vals.items():
                        xs = []
                        for sd in search_seeds:
                            tok = _tok_for("geo", v, sim_extras)
                            st = train(v, tok, v.subset(1.0, sd), kind, sd, scfg)
                            xs.append(score(v, st, tok)[m])
                        per[n] = float(np.mean(xs))
                rec.update(status="ok", per_dataset=per)
            except Exception as e:  # noqa: BLE001 - one bad trial must not end the search
                import traceback
                traceback.print_exc()
                rec.update(status="failed", reason=f"{type(e).__name__}: {e}")
            rec["seconds"] = round(time.perf_counter() - t0, 1)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        trials.append(rec)
        if rec["status"] == "ok":
            print(f"    [{name}] " + "  ".join(f"{n} {x:.4f}" for n, x in
                                             rec["per_dataset"].items())
                  + f"  ({rec['seconds']:.0f}s)  {params}")
        else:
            print(f"    [{name}] {rec['status']}: {rec.get('reason')}  {params}")
    ok = [t for t in trials if t["status"] == "ok"]
    if not ok:
        raise RuntimeError(f"search ({setting}): no trial finished")
    # reference per dataset: the base configuration (or the median trial)
    base = next((t for t in ok if t["name"] == "base"), None)
    ref = {n: (base["per_dataset"][n] if base else
               float(np.median([t["per_dataset"][n] for t in ok]))) for n in vals}
    for t in ok:
        r = [(t["per_dataset"][n] / ref[n]) if hi else (ref[n] / t["per_dataset"][n])
             for n in vals if ref[n] and t["per_dataset"][n]]
        t["value"] = float(np.mean(r))
    best = max(ok, key=lambda t: t["value"])
    best_hp = copy.deepcopy(base_hp)
    for k, v in best["params"].items():
        best_hp[_where(k)][k] = v
    print(f"  -> best {best['name']}: mean ratio to base {best['value']:.4f}  "
          f"{best['params']}")
    for t in trials:
        t.pop("signature", None)
    info = {"metric": m, "higher_is_better": hi, "selected_on":
            "mean over datasets of validation score / base score",
            "epochs": scfg["epochs"], "seeds": list(search_seeds),
            "n_val": {n: len(v.test) for n, v in vals.items()},
            "n_combinations": total, "reference": ref,
            "best": {k: best.get(k) for k in ("name", "params", "value",
                                               "per_dataset", "geo_layout")},
            "trials": [{k: t.get(k) for k in ("name", "params", "status", "reason",
                                                "value", "per_dataset", "geo_layout",
                                                "seconds")} for t in trials]}
    return best_hp, info


def _write_search_files(search_dir, setting, info, base_config, best_hp):
    import csv
    import json
    from .exp_search import _write_best
    cfg = copy.deepcopy(base_config or {})
    if isinstance(cfg.get("transfer_suite"), dict):
        cfg["transfer_suite"].pop("search", None)   # the file runs the winner
    _write_best(search_dir / f"best_{setting.replace('/', '_')}.toml",
                cfg, best_hp, info["best"],
                "mean validation score / base score")
    path = search_dir / "search_results.csv"
    names = sorted(info["reference"])
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["setting", "name", "status", "mean_ratio_to_base"]
                       + [f"{info['metric']}_{n}" for n in names]
                       + ["params", "geo_layout", "seconds"])
        for t in info["trials"]:
            pdv = t.get("per_dataset") or {}
            w.writerow([setting, t["name"], t["status"], t.get("value")]
                       + [pdv.get(n) for n in names]
                       + [json.dumps(t["params"]), t.get("geo_layout", ""),
                          t.get("seconds")])


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(datasets=("tdrive", "porto", "roma", "ais", "geolife"), paths=None,
        tasks=("similarity", "eta"), kinds=("lstm",),
        fractions=(0.01, 0.1, 1.0), similarity_extras="none",
        similarity_metric="KP@10", adapt_norm=True, diagnostics=False,
        n_train=None, n_test=2000, max_traj=20000, epochs=5, ft_epochs=None,
        bs=128, lr=1e-3, ft_lr=None, seeds=(0,), device="cpu", out=None,
        search=False, space=None, base_config=None, search_method="grid",
        n_trials=20, search_seed=0, search_epochs=None, search_seeds=(0,),
        val_frac=0.2, search_dir=None, resume=True):
    from pathlib import Path
    fractions = tuple(sorted({float(f) for f in fractions if f > 0}))
    if search:
        from .exp_search import _where, check_space
        space = check_space(space or {})
        if any(_where(k) == "model" for k in space):
            print("  note: [search_space] has [model] keys; only GEO's model is "
                  "tuned, GPE keeps the base architecture")
        search_dir = Path(search_dir or (Path(out).with_suffix("").as_posix() + "_search"
                                         if out else "results/transfer_search"))
        search_dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            (search_dir / "trials.jsonl").unlink(missing_ok=True)
        (search_dir / "search_results.csv").unlink(missing_ok=True)
    if any(f > 1 for f in fractions):
        raise SystemExit("transfer_suite: fractions must be in (0, 1]")
    cfg = dict(epochs=epochs, ft_epochs=ft_epochs or epochs, bs=bs, lr=lr,
               ft_lr=ft_lr or lr, device=device, n_train=n_train, n_test=n_test)
    raw, skipped = load_available(
        datasets, paths or {},
        lambda n: ({"n": (n_train or 4 * n_test) + n_test} if n.startswith("synthetic")
                   else {"max_traj": max_traj}),
        required=1, what="transfer_suite")
    base_hp = copy.deepcopy(_cfg.HP)
    rows = [GPE_ROW, GEO_ROW] + ([DIAG_ROW] if diagnostics else [])
    res = {"transfer": True, "rows": rows, "tasks": list(tasks), "kinds": list(kinds),
           "fractions": list(fractions), "seeds": list(seeds),
           "adapt_norm": bool(adapt_norm), "search": bool(search),
           "methods": {GPE_ROW: "GPE as published: fixed frequency ladder on raw "
                                "lon/lat, spatial only, no frame",
                       GEO_ROW: {t: describe_geo(t, similarity_extras) for t in tasks}},
           "results": {}, "skipped_datasets": skipped, "skipped_tasks": {}}
    for tname in tasks:
        domains = {}
        for name, trajs in raw.items():
            data = trajs
            try:
                if tname == "mode" and name == "geolife":
                    from ..data import load_dataset
                    data = load_dataset("geolife", (paths or {}).get("geolife"),
                                        max_traj=max_traj, labelled_only=True)
                task = TASKS[tname](dict(cfg)).prepare(data)
                if tname == "similarity":
                    task.metric = similarity_metric
                domains[name] = Domain(name, task)
            except (ValueError, FileNotFoundError) as e:
                res["skipped_tasks"][f"{name}/{tname}"] = str(e)
                print(f"  !! {tname} on {name}: {e}")
        if len(domains) < 2:
            msg = (f"needs at least two usable datasets, got {list(domains) or 'none'}")
            res["skipped_tasks"][f"*/{tname}"] = msg
            print(f"  !! SKIPPED {tname} transfer: {msg}")
            continue
        some = next(iter(domains.values())).task
        for kind in kinds:
            print(f"[transfer] {tname} / {kind}: " + ", ".join(
                f"{n} {len(d.train)}/{len(d.test)}" for n, d in domains.items())
                + f"; GEO = {describe_geo(tname, similarity_extras)}")
            geo_hp, info = base_hp, None
            if search:
                setting = f"{tname}/{kind}"
                try:
                    geo_hp, info = search_shared(
                        domains, kind, space, method=search_method,
                        n_trials=n_trials, search_seed=search_seed,
                        val_frac=val_frac, search_epochs=search_epochs,
                        search_seeds=search_seeds, sim_extras=similarity_extras,
                        cfg=cfg, log_path=search_dir / "trials.jsonl",
                        setting=setting, resume=resume)
                    _write_search_files(search_dir, setting, info, base_config, geo_hp)
                except RuntimeError as e:
                    print(f"  !! search skipped for {setting}: {e}; using the base "
                          "settings")
                    info = {"error": str(e)}
            runs = [run_seed(domains, kind, s, cfg, base_hp, geo_hp, fractions,
                             similarity_extras, adapt_norm, diagnostics) for s in seeds]
            r = _mean(runs)
            from .exp_gpe_suite import geo_layout
            r["geo_layout"] = geo_layout(None, geo_hp["encoders"])["text"]
            r["geo_settings"] = {k: v for sec in ("encoders", "model")
                                 for k, v in geo_hp[sec].items()
                                 if v != base_hp[sec].get(k)}
            if info is not None:
                r["search"] = info
            r.update(metric=some.metric, higher_is_better=some.higher,
                     datasets=list(domains),
                     n_train={n: len(d.train) for n, d in domains.items()},
                     n_test={n: len(d.test) for n, d in domains.items()})
            res["results"].setdefault(tname, {})[kind] = r
            save_results(res, out)
    if not res["results"]:
        raise FileNotFoundError("transfer_suite: no task had two usable datasets")
    _summary(res)
    save_results(res, out)
    return res


def _summary(res):
    print("\n==== transfer: in domain on the target -> zero-shot from the source ====")
    for tname, by_kind in res["results"].items():
        for kind, r in by_kind.items():
            m, hi = r["metric"], r["higher_is_better"]
            print(f"[{tname} / {kind}] {m} ({'higher' if hi else 'lower'} is better)")
            for key, pr in r["pairs"].items():
                ind = r["in_domain"][pr["dst"]]
                print(f"  {key:18s} " + "   ".join(
                    f"{row}: {ind[row][m]:.3f} -> {pr['zero_shot'][row][m]:.3f}"
                    for row in (GPE_ROW, GEO_ROW)))
