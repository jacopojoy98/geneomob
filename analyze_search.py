#!/usr/bin/env python3
"""Analyse a hyperparameter-search table and suggest where to search next.

    python analyze_search.py search_results.csv
    python analyze_search.py search_results.csv --metric metric:MRR -o analysis/

Input: the `search_results.csv` written by `geomob search` (any table with one
row per trial, one column per hyperparameter and a score column works). The
script does not import geomob.

What it does
  1. Importance. Splits the variance of the score across the hyperparameters
     (main effects) and their pairs (two-way interactions), as in an ANOVA /
     fANOVA. "eta^2 = 0.40" means that hyperparameter alone explains 40% of
     the spread between trials.
  2. Noise. Each trial is a single training run, so part of the spread is
     luck. The residual of an additive + pairwise model estimates that noise;
     a difference between two settings counts as real only if it is larger
     than about two standard errors.
  3. Suggestions. For each hyperparameter: drop it (no real effect), refine
     around an interior optimum, or extend the range when the best value sits
     on the edge of what was tried. Hard constraints of GEO are respected
     (location dims a multiple of 4 and within the budget left by the angular
     harmonics, minimum wavelength below maximum).
  4. Output: a printed report, report.md, figures, and next_search_space.toml
     ready to paste into geomob.toml.

Dependencies: numpy, pandas, matplotlib (scipy optional).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"MR", "MAE", "RMSE", "MAPE", "seconds"}
KNOWN = {"rank", "name", "token_dim", "geo_layout", "seconds"}
ACCENT, GREY, REF = "#0072B2", "#9A9A9A", "#D55E00"


# ==========================================================================
# loading
# ==========================================================================
def _is_listy(s: pd.Series) -> bool:
    return s.dropna().astype(str).str.strip().str.startswith("[").all()


def load(path, metric=None, lower_is_better=None):
    df = pd.read_csv(path)
    cols = list(df.columns)
    if metric is None:                              # geomob: column after "name"
        metric = cols[cols.index("name") + 1] if "name" in cols else cols[0]
    if metric not in df:
        raise SystemExit(f"metric column '{metric}' not found; columns: {cols}")
    short = metric.split(":")[-1]
    higher = not (short in LOWER_IS_BETTER) if lower_is_better is None else not lower_is_better

    # hyperparameters: everything that is not bookkeeping or a metric
    hp = [c for c in cols if c not in KNOWN and c != metric
          and not c.startswith("metric:") and df[c].nunique(dropna=True) > 1]

    is_ref = df["rank"].astype(str).eq("ref") if "rank" in df else pd.Series(False, index=df.index)
    is_base = df["name"].astype(str).eq("base") if "name" in df else pd.Series(False, index=df.index)
    refs = df[is_ref | is_base]
    trials = df[~(is_ref | is_base)].dropna(subset=[metric]).copy()

    # derived numeric features, for trends (never used as ANOVA factors)
    derived = {}
    if "geo_layout" in trials:
        for part in ("loc", "radial", "angular"):
            v = trials["geo_layout"].astype(str).str.extract(rf"{part} (\d+)")[0]
            if v.notna().all():
                derived[f"{part}_dims"] = v.astype(int)
    for c in hp:
        if _is_listy(trials[c]):
            lists = trials[c].apply(json.loads)
            derived[f"{c}:count"] = lists.apply(len)
            derived[f"{c}:max"] = lists.apply(lambda x: max(x) if x else 0)
    for k, v in derived.items():
        trials[k] = v
    return trials, refs, metric, higher, hp, list(derived)


# ==========================================================================
# 1-2. importance and noise
# ==========================================================================
def _levels(s: pd.Series):
    """Levels in a natural order: numeric ascending, lists by length then
    content, otherwise as strings."""
    vals = s.dropna().unique().tolist()
    if s.dtype.kind in "if":
        return sorted(vals)
    if _is_listy(s):
        return sorted(vals, key=lambda v: (len(json.loads(v)), json.loads(v)))
    return sorted(map(str, vals))


def eta_squared(df, y, hp):
    """Main-effect and pairwise-interaction shares of the total variance."""
    yv = df[y].to_numpy(float)
    sst = float(((yv - yv.mean()) ** 2).sum())
    main = {}
    for c in hp:
        g = df.groupby(c)[y]
        main[c] = float((g.count() * (g.mean() - yv.mean()) ** 2).sum()) / sst
    inter = {}
    for a, b in itertools.combinations(hp, 2):
        g = df.groupby([a, b])[y]
        ss_cells = float((g.count() * (g.mean() - yv.mean()) ** 2).sum())
        inter[(a, b)] = max(ss_cells / sst - main[a] - main[b], 0.0)
    return main, inter, sst


def design(df, hp, pairs):
    """One-hot design matrix: intercept, main effects, chosen interactions."""
    blocks = [np.ones((len(df), 1))]
    for c in hp:
        d = pd.get_dummies(df[c].astype(str), drop_first=True).to_numpy(float)
        blocks.append(d)
    for a, b in pairs:
        da = pd.get_dummies(df[a].astype(str), drop_first=True).to_numpy(float)
        db = pd.get_dummies(df[b].astype(str), drop_first=True).to_numpy(float)
        blocks.append(np.einsum("ni,nj->nij", da, db).reshape(len(df), -1))
    return np.hstack(blocks)


def noise_sigma(df, y, hp):
    """Residual std of an additive + all-pairs model (an upper bound on the
    trial-to-trial noise: it also absorbs higher-order interactions)."""
    yv = df[y].to_numpy(float)
    for pairs in (list(itertools.combinations(hp, 2)), []):
        X = design(df, hp, pairs)
        rank = np.linalg.matrix_rank(X)
        dof = len(yv) - rank
        if dof >= max(10, 0.15 * len(yv)):
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            r = yv - X @ beta
            return float(np.sqrt((r @ r) / dof)), ("main + pairwise" if pairs else "main effects"), dof
    return float(np.std(yv, ddof=1)), "none (too few trials: raw std)", len(yv) - 1


def level_table(df, y, c, sigma, higher):
    lv = _levels(df[c])
    rows = []
    for v in lv:
        s = df.loc[df[c] == v, y]
        rows.append({"value": v, "n": len(s), "mean": s.mean(),
                     "best": s.max() if higher else s.min(),
                     "se": sigma / math.sqrt(len(s))})
    t = pd.DataFrame(rows)
    order = t["mean"].rank(ascending=not higher, method="first").astype(int)
    t["rank"] = order
    return t


# ==========================================================================
# 3. suggestions
# ==========================================================================
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return f"{v:.1f}"
    return str(v)


def _geo_budget(df):
    h = int(df["token_dim"].mode()[0]) if "token_dim" in df else 128
    return h


MIN_SHARE = 0.02      # below this share of the variance a knob is not worth searching


def suggest(df, y, c, table, eta, clear, higher, hp_best, h, min_share=MIN_SHARE):
    """One recommendation dict for hyperparameter c."""
    best = table.sort_values("rank").iloc[0]
    lv = list(table["value"])
    nums = [_num(v) for v in lv]
    numeric = all(n is not None for n in nums)
    i = lv.index(best["value"])
    rec = {"param": c, "best": best["value"], "eta2": eta, "clear": clear}

    if eta < min_share:
        why = (f"explains only {100 * eta:.1f}% of the spread"
               + (" (detectable, but too small to be worth searching)" if clear
                  else " and is not distinguishable from noise"))
        rec.update(action="fix", why=why, values=[best["value"]])
        return rec

    if numeric:
        geometric = all(n > 0 for n in nums) and (max(nums) / min(nums) >= 4
                                                  or c.endswith("_m"))
        if c == "geo_loc_dims":
            geometric = False
        at_low, at_high = i == 0, i == len(lv) - 1
        if len(lv) >= 2:
            step = (nums[1] / nums[0] if geometric else nums[1] - nums[0]) if at_low else \
                   (nums[-1] / nums[-2] if geometric else nums[-1] - nums[-2])
        else:
            step = 2.0 if geometric else 1.0
        b = nums[i]
        if at_high or at_low:
            sgn = 1 if at_high else -1
            if geometric:
                new = [b, b * step ** sgn, b * step ** (2 * sgn)]
            else:
                new = [b, b + sgn * step, b + 2 * sgn * step]
            rec.update(action="extend", why=f"best value is the {'largest' if at_high else 'smallest'} "
                       f"tried: the optimum may lie {'above' if at_high else 'below'} the range",
                       values=new)
        else:
            lo, hi = nums[i - 1], nums[i + 1]
            mid = (lambda a, b_: math.sqrt(a * b_)) if geometric else (lambda a, b_: (a + b_) / 2)
            rec.update(action="refine", why="optimum is inside the range: search more finely around it",
                       values=[mid(lo, b), b, mid(b, hi)])
        rec["values"] = _constrain(c, rec["values"], hp_best, h)
        return rec

    # list-valued (e.g. harmonics): extrapolate along the trend in length
    if _is_listy(df[c]):
        lists = [json.loads(v) for v in lv]
        bl = json.loads(best["value"])
        longest = max(lists, key=len)
        if bl == longest and len(bl) >= 2:
            nxt = bl + [bl[-1] + (bl[-1] - bl[-2])]
            rec.update(action="extend", why="the longest list tried is best: try one more element",
                       values=[json.dumps(bl), json.dumps(nxt)])
        else:
            rec.update(action="refine", why="keep the best and its neighbours",
                       values=[json.dumps(x) for x in lists if abs(len(x) - len(bl)) <= 1])
        rec["values"] = _constrain(c, rec["values"], hp_best, h)
        return rec

    rec.update(action="refine", why="categorical: keep the best two",
               values=list(table.sort_values("rank")["value"][:2]))
    return rec


def _couple_geo_budget(recs, df, h):
    """GEO splits a fixed budget h between location (4 per scale), angular
    harmonics (2 each) and the radial part (the rest). If more location dims
    is the way up but the cap is set by the harmonics, and the harmonics
    themselves hardly matter, free their budget: offer a shorter list too.
    Combinations that do not fit are skipped by `geomob search` itself."""
    by = {r["param"]: r for r in recs}
    loc, harm = by.get("geo_loc_dims"), by.get("geo_harmonics")
    notes = []
    if not loc or not harm or loc["action"] != "extend":
        return notes
    best_h = json.loads(harm["best"]) if isinstance(harm["best"], str) else [1, 4]
    cap = (h - 2 * len(best_h)) // 4 * 4
    if max(float(v) for v in loc["values"]) < cap:
        return notes
    short = best_h[:2]
    new_cap = (h - 2 * len(short)) // 4 * 4
    if new_cap <= cap:
        return notes
    start = int(float(loc["best"]))
    loc["values"] = list(range(start, new_cap + 1, 4))
    if harm["action"] == "fix":
        harm["action"] = "explore"
        harm["values"] = [json.dumps(short), json.dumps(best_h)]
        harm["why"] = (f"hardly matters itself, but every harmonic costs 2 dims that "
                       f"location could use: with {short} location can go up to "
                       f"{new_cap}, with {best_h} only to {cap}")
    notes.append(f"Location dims are capped at {cap} by the {len(best_h)} harmonics; "
                 f"values above {cap} are only buildable with the shorter list "
                 f"{short} (geomob search skips the combinations that do not fit).")
    return notes


def _constrain(c, values, hp_best, h):
    """GEO's hard constraints, so every suggested value can actually be built."""
    out = []
    for v in values:
        if c == "geo_loc_dims":
            harm = hp_best.get("geo_harmonics")
            n_h = len(json.loads(harm)) if isinstance(harm, str) and harm.startswith("[") else 2
            top = (h - 2 * n_h) // 4 * 4
            v = int(round(float(v) / 4.0)) * 4
            v = max(4, min(v, top))
        elif c == "geo_harmonics":
            lst = json.loads(v)
            loc = _num(hp_best.get("geo_loc_dims"))
            if loc and loc > 0 and h - loc - 2 * len(lst) < 0:
                continue                      # no room left for these harmonics
        elif c.endswith("_lam_min_m"):
            mx = _num(hp_best.get(c.replace("_min_", "_max_")))
            v = float(v)
            if mx and v >= mx:
                continue
            v = float(f"{v:.3g}")
        elif c.endswith("_lam_max_m"):
            mn = _num(hp_best.get(c.replace("_max_", "_min_")))
            v = float(v)
            if mn and v <= mn:
                continue
            v = float(f"{v:.3g}")
        elif isinstance(v, float):
            v = float(f"{v:.4g}")
        if v not in out:
            out.append(v)
    return out


# ==========================================================================
# figures
# ==========================================================================
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "legend.frameon": False})


def fig_importance(main, inter, sigma, sst, n, out, top_inter=6):
    import matplotlib.pyplot as plt
    items = sorted(main.items(), key=lambda kv: kv[1])
    ints = sorted(inter.items(), key=lambda kv: kv[1])[-top_inter:]
    labels = [k for k, _ in items] + [f"{a} x {b}" for (a, b), _ in ints]
    vals = [v for _, v in items] + [v for _, v in ints]
    kinds = ["main"] * len(items) + ["inter"] * len(ints)
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(vals) + 1.0))
    y = np.arange(len(vals))
    for yi, j in zip(y, order):
        ax.barh(yi, vals[j], height=0.62, color=ACCENT if kinds[j] == "main" else "#9CC3E0",
                edgecolor="none")
        ax.text(vals[j] + 0.005, yi, f"{100 * vals[j]:.1f}%", va="center", fontsize=6.5,
                color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[j] for j in order], fontsize=7)
    noise_share = sigma ** 2 * (n - 1) / sst
    ax.axvline(noise_share, color=GREY, ls="--", lw=0.9)
    ax.text(noise_share, len(vals) - 0.4, " noise level", color="#555555", fontsize=6.5,
            va="bottom")
    ax.set_xlabel("share of the variance between trials (eta$^2$)")
    ax.set_title("What drives the score: main effects (dark) and pairwise "
                 "interactions (light)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_importance.png")
    plt.close(fig)


def fig_marginals(df, y, hp, tables, refs, higher, out, metric):
    import matplotlib.pyplot as plt
    n = len(hp)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.3 * nrow + 0.4), squeeze=False,
                             sharey=True)
    rng = np.random.default_rng(0)
    for k, c in enumerate(hp):
        ax = axes[k // ncol][k % ncol]
        t = tables[c]
        for j, v in enumerate(t["value"]):
            s = df.loc[df[c] == v, y].to_numpy()
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(s)), s, s=5, color=GREY,
                       alpha=0.55, linewidths=0)
        x = np.arange(len(t))
        ax.errorbar(x, t["mean"], yerr=1.96 * t["se"], color=ACCENT, lw=1.6,
                    marker="o", ms=4.5, capsize=2.5, zorder=3)
        best_j = int(np.where(t["rank"] == 1)[0][0])
        ax.plot(best_j, t["mean"].iloc[best_j], marker="o", ms=8, mfc="none",
                mec=ACCENT, mew=1.2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(v) for v in t["value"]], fontsize=6.5,
                           rotation=20 if any(len(_short(v)) > 7 for v in t["value"]) else 0)
        ax.set_title(c, fontsize=7.5)
        for _, r in refs.iterrows():
            ax.axhline(r[y], color=REF if r["rank"] == "ref" else "#555555",
                       ls="--" if r["rank"] == "ref" else ":", lw=0.9)
        if k % ncol == 0:
            arrow = r"$\uparrow$" if higher else r"$\downarrow$"
            ax.set_ylabel(f"{metric} {arrow}")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    # direct labels for the reference lines, once
    ax0 = axes[0][ncol - 1] if n >= ncol else axes[0][0]
    for _, r in refs.iterrows():
        ax0.annotate(str(r["name"]), xy=(1.0, r[y]), xycoords=("axes fraction", "data"),
                     xytext=(3, 0), textcoords="offset points", fontsize=6, va="center",
                     color=REF if r["rank"] == "ref" else "#555555", annotation_clip=False)
    fig.suptitle("Score by hyperparameter value: grey = single trials, blue = mean "
                 "with 95% interval, ring = best value", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_marginals.png")
    plt.close(fig)


def fig_interaction(df, y, a, b, higher, out, metric):
    import matplotlib.pyplot as plt
    la, lb = _levels(df[a]), _levels(df[b])
    M = np.array([[df.loc[(df[a] == u) & (df[b] == v), y].mean() for v in lb] for u in la])
    fig, ax = plt.subplots(figsize=(0.9 * len(lb) + 2.2, 0.45 * len(la) + 1.4))
    cmap = "Blues" if higher else "Blues_r"
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(len(la)):
        for j in range(len(lb)):
            if np.isfinite(M[i, j]):
                dark = (M[i, j] - lo) / max(hi - lo, 1e-12) > 0.6
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if dark == higher else "#1A1A1A")
    ax.set_xticks(range(len(lb)))
    ax.set_xticklabels([_short(v) for v in lb], fontsize=6.5)
    ax.set_yticks(range(len(la)))
    ax.set_yticklabels([_short(v) for v in la], fontsize=6.5)
    ax.set_xlabel(b)
    ax.set_ylabel(a)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label=f"mean {metric}")
    ax.set_title("Strongest interaction: mean score per combination", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_interaction.png")
    plt.close(fig)


def _short(v):
    s = str(v)
    if s.endswith(".0"):
        s = s[:-2]
    return s.replace(" ", "")


# ==========================================================================
# report
# ==========================================================================
def to_toml_value(c, v):
    if isinstance(v, str) and v.startswith("["):
        return v
    if c == "geo_loc_dims":
        return str(int(float(v)))
    f = _num(v)
    return repr(float(f)) if f is not None else json.dumps(str(v))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("--metric", default=None,
                   help="score column (default: the one geomob ranked by)")
    p.add_argument("--lower-is-better", action="store_true", default=None,
                   help="force the direction (default: from the metric name)")
    p.add_argument("-o", "--out", default="search_analysis")
    p.add_argument("--top", type=int, default=10, help="rows in the top-trials table")
    p.add_argument("--min-share", type=float, default=MIN_SHARE,
                   help="minimum share of variance for a hyperparameter to be "
                        "searched further (default 0.02)")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, refs, y, higher, hp, derived = load(a.csv, a.metric, a.lower_is_better)
    if len(hp) == 0 or len(df) < 4:
        raise SystemExit("need several trials that vary at least one hyperparameter")

    main_eta, inter_eta, sst = eta_squared(df, y, hp)
    sigma, model, dof = noise_sigma(df, y, hp)
    tables = {c: level_table(df, y, c, sigma, higher) for c in hp}
    best_row = df.sort_values(y, ascending=not higher).iloc[0]
    hp_best = {c: best_row[c] for c in hp}
    h = _geo_budget(df)

    recs = []
    for c in hp:
        t = tables[c].sort_values("rank")
        clear = False
        if len(t) > 1:
            d = abs(t["mean"].iloc[0] - t["mean"].iloc[1])
            se = math.hypot(t["se"].iloc[0], t["se"].iloc[1])
            clear = d > 2 * se
        recs.append(suggest(df, y, c, tables[c], main_eta[c], clear, higher, hp_best,
                            h, a.min_share))
    recs.sort(key=lambda r: -r["eta2"])
    notes = _couple_geo_budget(recs, df, h)

    top_pairs = sorted(inter_eta.items(), key=lambda kv: -kv[1])
    strong_pairs = [(k, v) for k, v in top_pairs if v >= 0.03]

    # ---------------------------------------------------------------- figures
    try:
        _style()
        fig_importance(main_eta, inter_eta, sigma, sst, len(df), out)
        fig_marginals(df, y, hp, tables, refs, higher, out, y.split(":")[-1])
        if top_pairs:
            (pa, pb), _ = top_pairs[0]
            fig_interaction(df, y, pa, pb, higher, out, y.split(":")[-1])
        figs_ok = True
    except Exception as e:  # noqa: BLE001 - the text report still matters
        print(f"(figures skipped: {type(e).__name__}: {e})")
        figs_ok = False

    # ----------------------------------------------------------------- report
    L = []
    w = L.append
    better = "higher" if higher else "lower"
    w(f"# Hyperparameter search analysis\n")
    w(f"File: `{a.csv}`  \nScore: **{y}** ({better} is better), {len(df)} trials, "
      f"{len(hp)} hyperparameters: {', '.join(hp)}.\n")
    vals = df[y]
    w("## Overview\n")
    w(f"- Spread between trials: {vals.min():.4f} to {vals.max():.4f} "
      f"(std {vals.std():.4f}).")
    w(f"- Estimated noise of a single trial: sigma = {sigma:.4f} "
      f"(residual of a {model} model, {dof} degrees of freedom; an upper bound).")
    for _, r in refs.iterrows():
        gap = (best_row[y] - r[y]) * (1 if higher else -1)
        verdict = ("ahead of" if gap > 2 * sigma else "behind" if gap < -2 * sigma
                   else "indistinguishable from")
        w(f"- Best trial ({best_row['name']}, {best_row[y]:.4f}) is {verdict} "
          f"**{r['name']}** ({r[y]:.4f}; difference {gap:+.4f}, noise 2 sigma = {2 * sigma:.4f}).")
    n_close = int(((vals - best_row[y]).abs() <= 2 * sigma).sum())
    w(f"- {n_close} trial(s) are within 2 sigma of the best: they cannot be "
      f"told apart from it with one run each.\n")

    w("## Which hyperparameters matter\n")
    w(f"A hyperparameter is worth searching if it explains at least "
      f"{100 * a.min_share:.0f}% of the variance. 'Detectable' means the best value "
      f"beats the runner-up by more than two standard errors; with many trials "
      f"per value even negligible differences become detectable.\n")
    w("| Hyperparameter | Share of variance | Best value | Detectable | Worth searching |")
    w("|---|---:|---|---|---|")
    for r in recs:
        w(f"| {r['param']} | {100 * r['eta2']:.1f}% | {_short(r['best'])} | "
          f"{'yes' if r['clear'] else 'no'} | "
          f"{'yes' if r['eta2'] >= a.min_share else 'no'} |")
    resid_share = sigma ** 2 * (len(df) - 1) / sst
    w(f"\nNoise accounts for roughly {100 * resid_share:.1f}% of the variance. "
      f"Strongest interactions: " + (", ".join(
          f"{a_} x {b_} ({100 * v:.1f}%)" for (a_, b_), v in top_pairs[:3]) or "none") + ".\n")

    w("## Per value\n")
    for c in [r["param"] for r in recs]:
        t = tables[c]
        w(f"**{c}**\n")
        w("| value | trials | mean | 95% interval | best |")
        w("|---|---:|---:|---|---:|")
        for _, r in t.iterrows():
            w(f"| {_short(r['value'])} | {int(r['n'])} | {r['mean']:.4f} | +/-{1.96 * r['se']:.4f} | "
              f"{r['best']:.4f} |")
        w("")

    if derived:
        w("## Trends in derived quantities\n")
        w("Rank correlation (Spearman) of the score with quantities computed "
          "from the hyperparameters:\n")
        for d in derived:
            rho = df[[d, y]].rank().corr().iloc[0, 1]
            if np.isfinite(rho):
                direction = "more is better" if (rho > 0) == higher else "less is better"
                w(f"- {d}: rho = {rho:+.2f} ({direction if abs(rho) > 0.2 else 'no clear trend'})")
        w("")

    w(f"## Top {a.top} trials\n")
    show = df.sort_values(y, ascending=not higher).head(a.top)
    w("| trial | " + y + " | " + " | ".join(hp) + " |")
    w("|---|---:|" + "---|" * len(hp))
    for _, r in show.iterrows():
        w(f"| {r['name']} | {r[y]:.4f} | " + " | ".join(_short(r[c]) for c in hp) + " |")
    w("")

    w("## Where to search next\n")
    for r in recs:
        w(f"- **{r['param']}**: {r['action']} -> {', '.join(_short(v) for v in r['values'])}. "
          f"{r['why'][0].upper() + r['why'][1:]}.")
    for n_ in notes:
        w(f"- {n_}")
    if strong_pairs:
        w("- Interacting pairs (search them jointly, not one at a time): "
          + ", ".join(f"{a_} x {b_}" for (a_, b_), _ in strong_pairs[:3]) + ".")
    explore = [r for r in recs if r["action"] != "fix"]  # extend / refine / explore
    fixed = [r for r in recs if r["action"] == "fix"]
    n_comb = int(np.prod([max(len(r["values"]), 1) for r in explore])) if explore else 1
    sec = float(df["seconds"].median()) if "seconds" in df else float("nan")
    w(f"- Suggested grid: {n_comb} combinations"
      + (f", about {n_comb * sec / 3600:.1f} h at {sec:.0f} s per trial" if np.isfinite(sec) else "")
      + ".")
    w(f"- Before trusting a winner, re-run the top 5 with 3-5 seeds: differences "
      f"below {2 * sigma:.4f} are within single-run noise.\n")

    toml = ["# suggested by analyze_search.py -- paste into geomob.toml", "",
            "[search_space]"]
    for r in explore:
        toml.append(f"{r['param']} = [" + ", ".join(to_toml_value(r["param"], v)
                                                     for v in r["values"]) + "]")
    if fixed:
        toml += ["", "# no measurable effect: fixed at the best value tried",
                 "# (merge these lines into the [encoders] section you already have)",
                 "[encoders]"]
        for r in fixed:
            toml.append(f"{r['param']} = {to_toml_value(r['param'], r['best'])}")
    (out / "next_search_space.toml").write_text("\n".join(toml) + "\n")

    if figs_ok:
        w("## Figures\n")
        w("- `fig_importance.png`: share of the variance per hyperparameter and pair; "
          "the dashed line is the noise level.")
        w("- `fig_marginals.png`: score per value of each hyperparameter, with the "
          "reference rows as horizontal lines.")
        w("- `fig_interaction.png`: mean score for each combination of the "
          "strongest interacting pair.\n")
    text = "\n".join(L)
    (out / "report.md").write_text(text)
    print(text)
    print(f"\nwrote {out}/report.md, {out}/next_search_space.toml"
          + (", fig_importance.png, fig_marginals.png, fig_interaction.png" if figs_ok else ""))


if __name__ == "__main__":
    main()
