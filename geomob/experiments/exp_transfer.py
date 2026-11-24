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
    calibrated    (travel time only) zero-shot, with the predictions mapped
                  to minutes by a line fitted on `calib_n` labelled target
                  trips: separates "orders the trips right" from "knows the
                  city's speeds". rank_corr (Spearman) is reported throughout
    fine-tuned    the source model trained further on a budget of the
                  target's training split (`fractions`: a number of
                  trajectories if > 1, a share of the split if <= 1; default
                  50, 200, 1000 trajectories and 100 %); travel times are
                  standardised with that budget's labels
    from scratch  (only with --scratch) a fresh model trained on the same
                  budget of the target: says whether the source helped

Protocol (v0.9.8)
    frame_origin  "anchor" (default): GEO's frame sits at the city anchor, so
                  two datasets of the same city (T-Drive and GeoLife) share
                  it, origin AND orientation; between cities it is re-fitted.
    early stopping (default on): every training holds out a part of its own
                  budget (`stop_val_frac`, at least 8) and keeps the best
                  epoch on it; `min_steps` (default 200) guarantees enough
                  optimisation steps for small budgets.
    ft_shrink / ft_noise: shrink-and-perturb the source weights before
                  fine-tuning (Ash & Adams 2020); off by default (1.0 / 0.0).
    geo_step_scale ([encoders], default "median"): GEO's displacement channel
                  reads steps in units of the corpus' median step, rescaled
                  to the training corpus (undoes differences in sampling).
    eta_max_min   travel-time trips longer than this are dropped (180).

With --diagnostics the zero-shot step also reports GEO with its frame left at
the source city (same model, no extra training).

With --ablations drop|only|both, GEO is also trained with channels removed
(see geomob.ablation): one in-domain training per dataset and variant, scored
in domain and zero-shot on every pair (no fine-tuning for the variants).

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
    never used. GPE is not searched. `select_on = "zero_shot"` (or "both")
    scores each candidate instead by its zero-shot transfer between the
    validation parts (every ordered pair, frame re-fitted): no extra training.
    For travel time the score is the calibrated zero-shot one (a line fitted
    on `calib_n` labelled trips of the target's training part), and the
    target validation labels are read.
    Files in <out>_search/: trials.jsonl (resumable), search_results.csv,
    best_<task>_<model>.toml.

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

    def subset(self, budget, seed, min_n=32):
        """Indices of a training budget: a fraction of the training split if
        budget <= 1 (at least `min_n`), a number of trajectories if > 1."""
        n = len(self.train)
        if budget == 1.0:
            return np.arange(n)
        k = int(round(budget)) if budget > 1 else max(min_n, round(budget * n))
        k = int(min(n, k))
        return np.sort(np.random.default_rng(1000 + seed).permutation(n)[:k])


def budget_label(b):
    """'50' (trajectories) or '10%' (of the training split)."""
    return f"{int(round(b))}" if b > 1 else f"{100 * b:g}%"


def order_budgets(budgets):
    """Numbers of trajectories first (ascending), then fractions."""
    bs = {float(b) for b in budgets if b > 0}
    return tuple(sorted(b for b in bs if b > 1)) + tuple(sorted(b for b in bs if b <= 1))


# --------------------------------------------------------------------------
# train / score, one code path per task type
# --------------------------------------------------------------------------
def _tokens(tok, trajs):
    return [tok(t) for t in trajs]


def _split_budget(idx, seed, cfg):
    """A labelled budget -> (training part, validation part). The validation
    part (for early stopping) comes out of the budget itself: a run with 50
    target trajectories has 50, not more."""
    idx = np.asarray(idx)
    lo = int(cfg.get("min_val", 8))
    if not cfg.get("early_stopping", True) or len(idx) < 3 * lo:
        return idx, idx[:0]
    n_val = int(np.clip(round(cfg.get("stop_val_frac", 0.1) * len(idx)), lo,
                        cfg.get("max_val", 1000)))
    perm = np.random.default_rng(2000 + seed).permutation(len(idx))
    return np.sort(idx[perm[n_val:]]), np.sort(idx[perm[:n_val]])


def _n_epochs(n, epochs, cfg, contrastive):
    """At least `min_steps` optimisation steps: with 50 trajectories, 5
    epochs are 5 steps, and both fine-tuning and training from scratch would
    stop almost where they started."""
    bs = cfg["bs"]
    if contrastive:
        steps = max(1, n // max(min(bs, max(n // 2, 2)), 1))
    else:
        steps = max(1, int(np.ceil(n / min(bs, 64))))
    need = int(np.ceil(cfg.get("min_steps", 0) / steps))
    return int(min(max(epochs, need), cfg.get("max_epochs", 1000)))


def _shrink_perturb(model, cfg, seed):
    """Ash & Adams (2020): theta <- shrink * theta + noise * N(0, 1) before
    fine-tuning, which removes most of the generalisation penalty of
    warm-starting. Learned wavelengths are left alone."""
    lam, sig = float(cfg.get("ft_shrink", 1.0)), float(cfg.get("ft_noise", 0.0))
    if lam == 1.0 and sig == 0.0:
        return
    import torch
    g = torch.Generator().manual_seed(3000 + seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "front." in name:
                continue
            p.mul_(lam)
            if sig:
                p.add_(sig * torch.randn(p.shape, generator=g).to(p.device))


class _Best:
    """Keeps the weights of the best epoch on the validation part."""

    def __init__(self, score_fn, higher, epochs, max_evals=50):
        self.score_fn, self.higher, self.epochs = score_fn, higher, epochs
        self.every = max(1, epochs // max_evals)
        self.v = self.state = None
        self.ep = epochs

    def __call__(self, model, ep):
        if (ep + 1) % self.every and ep + 1 != self.epochs:
            return
        v = self.score_fn(model)
        if np.isfinite(v) and (self.v is None or (v > self.v if self.higher else v < self.v)):
            self.v, self.ep = v, ep + 1
            self.state = copy.deepcopy(model.state_dict())

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)
        return model


def train(dom, tok, idx, kind, seed, cfg, init=None, adapt_norm=True):
    """Train on dom.train[idx]; from `init` (a state from another dataset)
    when given. With early stopping (default) a part of idx is held out and
    the best epoch on it is kept. Returns a state dict."""
    tr, va = _split_budget(idx, seed, cfg)
    trajs = [dom.train[i] for i in tr]
    epochs = cfg["ft_epochs"] if init is not None else cfg["epochs"]
    lr = cfg["ft_lr"] if init is not None else cfg["lr"]
    dev = cfg["device"]
    if dom.task.name == "similarity":
        from .exp_similarity import evaluate, train_contrastive
        epochs = _n_epochs(len(trajs), epochs, cfg, True)
        model = copy.deepcopy(init["model"]) if init is not None else None
        if model is not None:
            _shrink_perturb(model, cfg, seed)
        vt = [dom.train[i] for i in va]
        best = (_Best(lambda M: evaluate(M, tok, vt, dev)["MRR"], True, epochs)
                if len(vt) else None)
        model = train_contrastive(tok, trajs, kind=kind, epochs=epochs,
                                  bs=min(cfg["bs"], max(len(trajs) // 2, 2)),
                                  lr=lr, device=dev, seed=seed, model=model,
                                  on_epoch=best)
        if best is not None:
            best.restore(model)
        return {"model": model, "epochs": epochs,
                "best_epoch": best.ep if best is not None else epochs}
    import torch
    from . import exp_downstream as D
    raw = getattr(tok, "raw_cols", None)
    X = _tokens(tok, trajs)
    eta = dom.task.name == "eta"
    if init is None:
        S = D.Standardiser().fit(X, raw)
        if eta:
            y = dom.y_train[tr]
            mu, sd = float(y.mean()), float(y.std() + 1e-6)
            model = D._head(tok, 1, kind, seed)
        else:
            mu = sd = None
            model = D._head(tok, len(dom.classes), kind, seed)
    else:
        model = copy.deepcopy(init["model"])
        _shrink_perturb(model, cfg, seed)
        mu, sd = init["mu"], init["sd"]
        if eta:
            # target labels are in use from here on, so they set the scale
            # (only zero-shot must read out on the source's scale)
            y = dom.y_train[tr]
            mu, sd = float(y.mean()), float(y.std() + 1e-6)
        S = (D.Standardiser().fit(_tokens(tok, dom.train), raw) if adapt_norm
             else init["S"])
    Xs = S(X)
    if eta:
        yt = torch.from_numpy(((dom.y_train[tr] - mu) / sd).astype(np.float32))
        loss = lambda M, x, m, b: torch.nn.functional.mse_loss(
            M(x, m).squeeze(-1), yt[b].to(x.device))
    else:
        yt = torch.from_numpy(dom.y_train[tr].astype(np.int64))
        loss = lambda M, x, m, b: torch.nn.functional.cross_entropy(
            M(x, m), yt[b].to(x.device))
    epochs = _n_epochs(len(tr), epochs, cfg, False)
    best = None
    if len(va):
        Xv = S(_tokens(tok, [dom.train[i] for i in va]))
        if eta:
            yv = (dom.y_train[va] - mu) / sd
            best = _Best(lambda M: float(np.mean(np.abs(
                D._apply(M, Xv, dev).squeeze(-1) - yv))), False, epochs)
        else:
            yv = dom.y_train[va]
            best = _Best(lambda M: D.classification_metrics(
                D._apply(M, Xv, dev), yv, len(dom.classes))[0]["macro_F1"], True, epochs)
    D._fit(model, Xs, loss, epochs, min(cfg["bs"], 64), lr, dev, seed, on_epoch=best)
    if best is not None:
        best.restore(model)
    return {"model": model, "S": S, "mu": mu, "sd": sd, "epochs": epochs,
            "best_epoch": best.ep if best is not None else epochs}


def score(dom, state, tok, adapt_norm=True, calib=None):
    """Metrics of `state` on dom.test, read through tokenizer `tok`.
    ETA with `calib` (indices of labelled target training trips): the
    predictions are mapped to minutes by a least-squares line fitted on
    those trips only (two numbers learned from the target, nothing else)."""
    if dom.task.name == "similarity":
        from .exp_similarity import evaluate
        return evaluate(state["model"], tok, dom.test, dom.task.cfg["device"])
    from . import exp_downstream as D
    S = state["S"]
    if adapt_norm and state.get("_foreign"):
        S = D.Standardiser().fit(_tokens(tok, dom.train), getattr(tok, "raw_cols", None))
    dev = dom.task.cfg["device"]
    out = D._apply(state["model"], S(_tokens(tok, dom.test)), dev)
    if dom.task.name == "eta":
        out = out.squeeze(-1)
        if calib is not None and len(calib) >= 2:
            oc = D._apply(state["model"], S(_tokens(tok, [dom.train[i] for i in calib])),
                          dev).squeeze(-1)
            a, b = np.linalg.lstsq(np.c_[oc, np.ones_like(oc)], dom.y_train[calib],
                                   rcond=None)[0]
            return D.regression_metrics(a * out + b, dom.y_test)
        return D.regression_metrics(out * state["sd"] + state["mu"], dom.y_test)
    return D.classification_metrics(out, dom.y_test, len(dom.classes))[0]


# --------------------------------------------------------------------------
# one task x model x seed
# --------------------------------------------------------------------------
def _tok_for(method, dom, sim_extras, src_tok=None, refit=True, origin=None):
    """A tokenizer for `dom`. GEO: the source tokenizer (same wavelengths,
    possibly learned) with its frame re-fitted on dom's city (kept when both
    datasets sit at the same city anchor; see FramedTokenizer.refit).
    `origin`: frame origin rule (transfer_suite default: the city anchor)."""
    if src_tok is not None:
        t = copy.deepcopy(src_tok)
        if method != "gpe" and refit:           # GEO and its ablations
            t.refit(dom.city)
        return t
    if method == "gpe":
        t = gpe_paper()
    elif method.startswith("abl:"):
        from .exp_robust import ablate
        t = ablate(geo_full(dom.task.name, sim_extras, origin), method[4:])
    else:
        t = geo_full(dom.task.name, sim_extras, origin)
    t.fit(dom.city)
    return t


def run_seed(domains, kind, seed, cfg, base_hp, geo_hp, fractions, sim_extras,
             adapt_norm, diagnostics, scratch=False, ablations="none"):
    from .exp_robust import ablation_row, ablation_variants
    rows = {"gpe": GPE_ROW, "geo": GEO_ROW}
    hp_of = {"gpe": base_hp, "geo": geo_hp}
    some = next(iter(domains.values())).task.name
    abl = [f"abl:{vk}" for vk in ablation_variants(some, sim_extras, ablations, geo_hp)]
    for m in abl:                       # in domain + zero-shot only
        rows[m], hp_of[m] = ablation_row(m[4:]), geo_hp
    names = list(domains)
    out = {"in_domain": {}, "pairs": {}}
    if scratch:
        out["scratch"] = {}
    states, toks = {}, {}
    # 1) full training on every dataset: the source models and the in-domain
    #    references at once
    for n in names:
        dom = domains[n]
        out["in_domain"][n] = {}
        if scratch:
            out["scratch"][n] = {}
        for m in ("gpe", "geo", *abl):
            with using(hp_of[m]):
                t0 = time.perf_counter()
                tok = _tok_for(m, dom, sim_extras, origin=cfg.get("frame_origin"))
                st = train(dom, tok, dom.subset(1.0, seed), kind, seed, cfg)
                states[n, m], toks[n, m] = st, tok
                out["in_domain"][n][rows[m]] = score(dom, st, tok)
                # 2) optional baseline: from scratch on fractions of this dataset
                if scratch and m in ("gpe", "geo"):
                    out["scratch"][n][rows[m]] = []
                    for f in fractions:
                        if f == 1.0:
                            out["scratch"][n][rows[m]].append(
                                out["in_domain"][n][rows[m]])
                            continue
                        t2 = _tok_for(m, dom, sim_extras, origin=cfg.get("frame_origin"))
                        s2 = train(dom, t2, dom.subset(f, seed), kind, seed, cfg)
                        out["scratch"][n][rows[m]].append(score(dom, s2, t2))
            print(f"    {n:8s} {m}: in domain {_fmt(out['in_domain'][n][rows[m]], dom)}"
                  f"  ({time.perf_counter() - t0:.0f}s)")
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
                    if T.task.name == "eta" and cfg.get("calib_n"):
                        pr.setdefault("calibrated", {})[rows[m]] = score(
                            T, st, tok, adapt_norm,
                            calib=T.subset(float(cfg["calib_n"]), seed))
                    if m == "geo" and diagnostics:
                        t_src = _tok_for(m, T, sim_extras, src_tok=toks[src, m],
                                         refit=False)
                        pr["zero_shot"][DIAG_ROW] = score(T, st, t_src, adapt_norm)
                    pr["finetune"][rows[m]] = []
                    for f in fractions:
                        ft = train(T, tok, T.subset(f, seed), kind, seed, cfg,
                                   init=states[src, m], adapt_norm=adapt_norm)
                        pr["finetune"][rows[m]].append(score(T, ft, tok))
            for m in abl:                       # ablations: zero-shot only
                with using(hp_of[m]):
                    tok = _tok_for(m, T, sim_extras, src_tok=toks[src, m])
                    st = dict(states[src, m], _foreign=True)
                    pr["zero_shot"][rows[m]] = score(T, st, tok, adapt_norm)
            out["pairs"][key] = pr
            cal = pr.get("calibrated")
            print(f"    {key:18s} zero-shot  GPE {_fmt(pr['zero_shot'][GPE_ROW], T)}"
                  f"  GEO {_fmt(pr['zero_shot'][GEO_ROW], T)}"
                  + (f"   calibrated GPE {_fmt(cal[GPE_ROW], T)}  GEO "
                     f"{_fmt(cal[GEO_ROW], T)}" if cal else "")
                  + "   | fine-tuned (GPE/GEO) "
                  + "  ".join(f"{budget_label(f)}: {_fmt(pr['finetune'][GPE_ROW][i], T)}/"
                              f"{_fmt(pr['finetune'][GEO_ROW][i], T)}"
                              for i, f in enumerate(fractions)))
    out["token_dims"] = {GPE_ROW: toks[names[0], "gpe"].dim,
                         GEO_ROW: toks[names[0], "geo"].dim}
    if abl:
        out["ablation_rows"] = [rows[m] for m in abl]
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
                  log_path, setting, resume=True, select_on="in_domain",
                  adapt_norm=True):
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
    sig = {"setting": setting, "datasets": sorted(vals),
           "epochs": scfg["epochs"], "seeds": list(search_seeds),
           "val_frac": val_frac, "sim_extras": sim_extras,
           "n_val": {n: len(v.test) for n, v in vals.items()},
           "base": base_hp,
           # trials run under another protocol are not reused
           "protocol": {k: cfg.get(k) for k in ("frame_origin", "early_stopping",
                                                "stop_val_frac", "min_steps")}}
    if select_on != "in_domain":            # trials also need zero-shot scores
        sig["zero_shot"] = {"adapt_norm": bool(adapt_norm),
                            "calib_n": cfg.get("calib_n")}
    signature = json.dumps(sig, sort_keys=True, default=float)
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
                per, zs = {}, {}
                with using(hp):
                    for sd in search_seeds:
                        fitted = {}
                        for n, v in vals.items():
                            tok = _tok_for("geo", v, sim_extras,
                                           origin=cfg.get("frame_origin"))
                            st = train(v, tok, v.subset(1.0, sd), kind, sd, scfg)
                            per.setdefault(n, []).append(score(v, st, tok)[m])
                            fitted[n] = (tok, st)
                        if select_on == "in_domain":
                            continue
                        for src, (stok, st) in fitted.items():
                            for dst, v in vals.items():
                                if dst == src or (v.task.name == "mode" and
                                                  vals[src].classes != v.classes):
                                    continue
                                t2 = _tok_for("geo", v, sim_extras, src_tok=stok)
                                # travel time: calibrated zero-shot (raw zero-shot
                                # MAE mostly measures the cities' trip durations)
                                cal = (v.subset(float(cfg["calib_n"]), sd)
                                       if v.task.name == "eta" and cfg.get("calib_n")
                                       else None)
                                zs.setdefault(f"{src}→{dst}", []).append(score(
                                    v, dict(st, _foreign=True), t2, adapt_norm,
                                    calib=cal)[m])
                per = {n: float(np.mean(x)) for n, x in per.items()}
                rec.update(status="ok", per_dataset=per)
                if zs:
                    rec["zero_shot"] = {k: float(np.mean(x)) for k, x in zs.items()}
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
                  + ("  | zero-shot " + "  ".join(f"{k} {x:.4f}" for k, x in
                                                   rec["zero_shot"].items())
                     if rec.get("zero_shot") else "")
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
    pairs = sorted({k for t in ok for k in (t.get("zero_shot") or {})})
    ref_zs = {k: (base["zero_shot"][k] if base and k in (base.get("zero_shot") or {})
                  else float(np.median([t["zero_shot"][k] for t in ok
                                        if k in (t.get("zero_shot") or {})])))
              for k in pairs}

    def ratio(v, r0):
        return (v / r0) if hi else (r0 / v)
    for t in ok:
        ind = [ratio(t["per_dataset"][n], ref[n]) for n in vals
               if ref[n] and t["per_dataset"][n]]
        z = [ratio(t["zero_shot"][k], ref_zs[k]) for k in pairs
             if ref_zs[k] and (t.get("zero_shot") or {}).get(k)]
        t["value_in_domain"] = float(np.mean(ind)) if ind else float("nan")
        t["value_zero_shot"] = float(np.mean(z)) if z else float("nan")
        t["value"] = {"in_domain": t["value_in_domain"],
                      "zero_shot": t["value_zero_shot"],
                      "both": float(np.mean(ind + z)) if ind + z else float("nan")
                      }[select_on]
    ok = [t for t in ok if np.isfinite(t["value"])] or ok
    best = max(ok, key=lambda t: t["value"])
    best_hp = copy.deepcopy(base_hp)
    for k, v in best["params"].items():
        best_hp[_where(k)][k] = v
    print(f"  -> best {best['name']} (selected on {select_on}): mean ratio to base "
          f"{best['value']:.4f} (in domain {best['value_in_domain']:.4f}"
          + (f", zero-shot {best['value_zero_shot']:.4f}" if pairs else "")
          + f")  {best['params']}")
    for t in trials:
        t.pop("signature", None)
    info = {"metric": m, "higher_is_better": hi, "select_on": select_on,
            "selected_on": {
                "in_domain": "mean over datasets of validation score / base score",
                "zero_shot": "mean over ordered pairs of zero-shot validation "
                             "score / base score",
                "both": "mean of both kinds of ratio"}[select_on],
            "epochs": scfg["epochs"], "seeds": list(search_seeds),
            "n_val": {n: len(v.test) for n, v in vals.items()},
            "n_combinations": total, "reference": ref,
            "best": {k: best.get(k) for k in ("name", "params", "value",
                                               "value_in_domain", "value_zero_shot",
                                               "per_dataset", "zero_shot",
                                               "geo_layout")},
            "trials": [{k: t.get(k) for k in ("name", "params", "status", "reason",
                                                "value", "value_in_domain",
                                                "value_zero_shot", "per_dataset",
                                                "zero_shot", "geo_layout",
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
                cfg, best_hp, info["best"], info.get("selected_on",
                                                     "mean validation score / base score"))
    path = search_dir / "search_results.csv"
    names = sorted(info["reference"])
    pairs = sorted({k for t in info["trials"] for k in (t.get("zero_shot") or {})})
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["setting", "name", "status", "selection_value",
                        "ratio_in_domain", "ratio_zero_shot"]
                       + [f"{info['metric']}_{n}" for n in names]
                       + [f"zero_shot_{k.replace(chr(8594), '_to_')}" for k in pairs]
                       + ["params", "geo_layout", "seconds"])
        for t in info["trials"]:
            pdv = t.get("per_dataset") or {}
            zsv = t.get("zero_shot") or {}
            w.writerow([setting, t["name"], t["status"], t.get("value"),
                        t.get("value_in_domain"), t.get("value_zero_shot")]
                       + [pdv.get(n) for n in names] + [zsv.get(k) for k in pairs]
                       + [json.dumps(t["params"]), t.get("geo_layout", ""),
                          t.get("seconds")])


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(datasets=("tdrive", "porto", "roma", "ais", "geolife"), paths=None,
        tasks=("similarity", "eta"), kinds=("lstm",),
        fractions=(50, 200, 1000, 1.0), similarity_extras="none",
        similarity_metric="KP@10", adapt_norm=True, diagnostics=False,
        n_train=None, n_test=2000, max_traj=20000, epochs=5, ft_epochs=None,
        bs=128, lr=1e-3, ft_lr=None, seeds=(0,), device="cpu", out=None,
        scratch=False, ablations="none", frame_origin="anchor",
        early_stopping=True, stop_val_frac=0.1, min_steps=200, ft_shrink=1.0,
        ft_noise=0.0, calib_n=50, eta_max_min=180.0,
        search=False, space=None, base_config=None, search_method="grid",
        n_trials=20, search_seed=0, search_epochs=None, search_seeds=(0,),
        val_frac=0.2, search_dir=None, resume=True, select_on="in_domain"):
    from pathlib import Path
    fractions = order_budgets(fractions)
    if select_on not in ("in_domain", "zero_shot", "both"):
        raise SystemExit("transfer_suite: select_on must be in_domain, zero_shot or both")
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
    if any(1 < f < 2 for f in fractions):
        raise SystemExit("transfer_suite: fractions are in (0, 1] (share of the "
                         "training split) or >= 2 (number of trajectories)")
    if frame_origin not in ("anchor", "median", "mean"):
        raise SystemExit("transfer_suite: frame_origin must be anchor, median or mean")
    cfg = dict(epochs=epochs, ft_epochs=ft_epochs or epochs, bs=bs, lr=lr,
               ft_lr=ft_lr or lr, device=device, n_train=n_train, n_test=n_test,
               frame_origin=frame_origin, early_stopping=bool(early_stopping),
               stop_val_frac=stop_val_frac, min_steps=int(min_steps or 0),
               ft_shrink=ft_shrink, ft_noise=ft_noise, calib_n=int(calib_n or 0),
               eta_max_min=eta_max_min)
    raw, skipped = load_available(
        datasets, paths or {},
        lambda n: ({"n": (n_train or 4 * n_test) + n_test} if n.startswith("synthetic")
                   else {"max_traj": max_traj}),
        required=1, what="transfer_suite")
    base_hp = copy.deepcopy(_cfg.HP)
    rows = [GPE_ROW, GEO_ROW] + ([DIAG_ROW] if diagnostics else [])
    from .exp_robust import ablation_row, ablation_variants
    for t in tasks:
        for vk in ablation_variants(t, similarity_extras, ablations):
            if ablation_row(vk) not in rows:
                rows.append(ablation_row(vk))
    res = {"transfer": True, "rows": rows, "tasks": list(tasks), "kinds": list(kinds),
           "fractions": list(fractions),
           "budget_labels": [budget_label(f) for f in fractions],
           "seeds": list(seeds),
           "adapt_norm": bool(adapt_norm), "search": bool(search),
           "protocol": {k: cfg[k] for k in ("frame_origin", "early_stopping",
                                            "stop_val_frac", "min_steps",
                                            "ft_shrink", "ft_noise", "calib_n",
                                            "eta_max_min")},
           "geo_step_scale": base_hp["encoders"].get("geo_step_scale", "median"),
           "scratch": bool(scratch), "ablations": ablations,
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
                        setting=setting, resume=resume, select_on=select_on,
                        adapt_norm=adapt_norm)
                    _write_search_files(search_dir, setting, info, base_config, geo_hp)
                except RuntimeError as e:
                    print(f"  !! search skipped for {setting}: {e}; using the base "
                          "settings")
                    info = {"error": str(e)}
            runs = [run_seed(domains, kind, s, cfg, base_hp, geo_hp, fractions,
                             similarity_extras, adapt_norm, diagnostics, scratch,
                             ablations)
                    for s in seeds]
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
                if "calibrated" in pr:
                    print(f"  {'':18s} " + "   ".join(
                        f"{row}: calibrated {pr['calibrated'][row][m]:.3f}, rank "
                        f"corr. {ind[row].get('rank_corr', float('nan')):.3f} -> "
                        f"{pr['zero_shot'][row].get('rank_corr', float('nan')):.3f}"
                        for row in (GPE_ROW, GEO_ROW)))
            abl = [row for row in res.get("rows", []) if row.startswith(GEO_ROW + " ")
                   and row in next(iter(r["in_domain"].values()))]
            if not abl:
                continue
            print(f"  channel ablation, % change from {GEO_ROW} (negative = worse):")

            def pct(v, f):
                return 100 * ((v - f) if hi else (f - v)) / max(abs(f), 1e-12)
            cols = list(r["in_domain"]) + list(r["pairs"])
            print(f"    {'':26s}" + "".join(f"{c:>16s}" for c in cols))
            for row in abl:
                vals = [pct(r["in_domain"][d][row][m], r["in_domain"][d][GEO_ROW][m])
                        for d in r["in_domain"]]
                vals += [pct(pr["zero_shot"][row][m], pr["zero_shot"][GEO_ROW][m])
                         for pr in r["pairs"].values()]
                print(f"    {row[len(GEO_ROW):].strip():26s}"
                      + "".join(f"{v:+16.1f}" for v in vals))
