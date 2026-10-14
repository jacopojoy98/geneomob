"""Cross-file comparison: results per dataset, and critical-difference diagrams.

These builders look at ALL the results files of a batch at once (the others
work one file at a time):

  fig_by_dataset_<task>     one panel per dataset, every encoder x backbone
  fig_similarity_by_dataset local similarity metrics, dataset by dataset
  fig_cd_<family>           critical-difference (CD) diagram of the methods'
                            average ranks (Demsar 2006)
  results_by_dataset.csv    the headline number of every task on every dataset
  cd_stats.json             ranks, test statistics and the blocks behind each CD

A CD diagram ranks the methods inside each *block* (one dataset x task x
backbone x metric), averages the ranks, and joins with a bar the methods whose
average ranks differ by less than the Nemenyi critical difference.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .builders_suite import TASK_METRICS, UNITS, _enc_color, _tx
from .style import HATCHES, WIDTH_2COL, annotate_bars, shorten

# Nemenyi critical values q_alpha / sqrt(2), alpha = 0.05, k = 2..10 (Demsar 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164}
TASK_TITLE = {"tul": "Trajectory-user linking", "eta": "Travel-time estimation",
              "mode": "Transportation-mode detection",
              "anomaly": "Anomaly detection (all injected types pooled)"}
MAIN = ("gpe", "gpe_ts", "geo", "geo_ts")       # ablation rows are not methods
SIM_METRICS = (("MR", False), ("MRR", True), ("MP", True), ("KP@10", True))


def split_inputs(datasets):
    """(downstream results by dataset, list of gpe_suite results)."""
    down, suites = {}, []
    for d in datasets:
        if "tasks" in d and "backbones" in d:
            down[d.get("dataset", f"run{len(down)}")] = d
        elif "parts" in d and "datasets" in d:
            suites.append(d)
    return down, suites


# ==========================================================================
# 1. every task on every dataset
# ==========================================================================
def _task_rows(task):
    """[(row label, getter(cell) -> value, metric label, higher-is-better)]"""
    if task == "anomaly":
        return [("latent kNN", lambda v: v["latent_knn"].get("all"), "AUROC", True),
                ("reconstruction", lambda v: v["reconstruction"].get("all"), "AUROC", True)]
    head, _, higher = TASK_METRICS[task]
    return [(None, lambda v, m=head: v.get(m), head + UNITS.get(head, ""), higher)]


def fig_by_dataset(down):
    figs = []
    tasks = [t for t in ("tul", "eta", "mode", "anomaly")
             if any(t in d["tasks"] for d in down.values())]
    for task in tasks:
        names = [n for n, d in down.items() if task in d["tasks"]]
        rows = _task_rows(task)
        fig, axes = plt.subplots(len(rows), len(names), squeeze=False,
                                 figsize=(max(2.3 * len(names), 3.4), 2.3 * len(rows) + 0.5))
        for r, (rlabel, get, mlabel, higher) in enumerate(rows):
            for c, name in enumerate(names):
                ax, d = axes[r][c], down[name]
                R = d["tasks"][task]["results"]
                keys = [k for k in R if k in MAIN] or list(R)
                kinds = d["backbones"]
                x = np.arange(len(keys))
                w = 0.8 / len(kinds)
                for j, kind in enumerate(kinds):
                    vals = [get(R[k][kind]) if kind in R[k] else np.nan for k in keys]
                    vals = [np.nan if v is None else v for v in vals]
                    bars = ax.bar(x + (j - (len(kinds) - 1) / 2) * w, vals, w,
                                  color=[_enc_color(d["encoders"][k], i)
                                         for i, k in enumerate(keys)],
                                  edgecolor="black", lw=0.3,
                                  hatch=HATCHES[j % len(HATCHES)], label=kind)
                    annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=4.5)
                if task == "anomaly":
                    ax.axhline(0.5, color="0.4", lw=0.7, ls=":")
                    ax.set_ylim(0, 1.08)
                ax.set_xticks(x)
                ax.set_xticklabels([shorten(d["encoders"][k], 16) for k in keys],
                                   rotation=25, ha="right", fontsize=5.5)
                if r == 0:
                    ax.set_title(name, fontsize=8)
                if c == 0:
                    arrow = r"$\uparrow$" if higher else r"$\downarrow$"
                    ax.set_ylabel((f"{rlabel}\n" if rlabel else "") + f"{mlabel} {arrow}")
                if r == 0 and c == len(names) - 1 and len(kinds) > 1:
                    from matplotlib.patches import Patch
                    ax.legend(handles=[Patch(facecolor="0.85", edgecolor="black", lw=0.3,
                                             hatch=HATCHES[j % len(HATCHES)], label=kd)
                                       for j, kd in enumerate(kinds)],
                              fontsize=5.5, title="backbone", title_fontsize=5.5)
        fig.suptitle(f"{TASK_TITLE[task]}, by dataset", fontsize=9)
        fig.tight_layout()
        figs.append((f"fig_by_dataset_{task}", fig))
    return figs


def fig_similarity_by_dataset(suites):
    figs = []
    for si, res in enumerate(suites):
        lg = res["parts"].get("local_global")
        if not lg:
            continue
        ds = res["datasets"]
        kinds = list(lg)
        fig, axes = plt.subplots(len(kinds), 4, squeeze=False,
                                 figsize=(WIDTH_2COL, 2.0 * len(kinds) + 0.6))
        for r, kind in enumerate(kinds):
            encs = lg[kind]
            for c, (m, higher) in enumerate(SIM_METRICS):
                ax = axes[r][c]
                x = np.arange(len(ds))
                w = 0.8 / len(encs)
                for j, (e, row) in enumerate(encs.items()):
                    vals = [row["per_dataset"][d]["local"][m] for d in ds]
                    bars = ax.bar(x + (j - (len(encs) - 1) / 2) * w, vals, w,
                                  color=_enc_color(row["name"], j), edgecolor="black",
                                  lw=0.3, hatch=HATCHES[j % len(HATCHES)],
                                  label=row["name"])
                    annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=4.5,
                                  rotation=90 if len(ds) * len(encs) > 8 else 0)
                ax.set_xticks(x)
                ax.set_xticklabels(ds, rotation=25, ha="right", fontsize=6)
                arrow = r"$\uparrow$" if higher else r"$\downarrow$"
                ax.set_title(f"{m} {arrow}", fontsize=7)
                ax.margins(y=0.18)
                if c == 0:
                    ax.set_ylabel(kind)
        h_, l_ = axes[0][0].get_legend_handles_labels()
        fig.suptitle("Local trajectory similarity, by dataset", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.legend(h_, l_, loc="upper right", ncol=len(l_), fontsize=6,
                   frameon=False, bbox_to_anchor=(0.995, 0.995))
        figs.append(("fig_similarity_by_dataset" + (f"_{si + 1}" if si else ""), fig))
    return figs


def overview_rows(down, suites):
    """Long-format rows: one headline number per (family, dataset, task,
    backbone, metric, method, seed). When a results file kept its per-seed
    values there is one row per seed; older files give a single row with
    seed = None holding the mean."""
    rows = []

    def add(**kw):
        rows.append(kw)

    for name, d in down.items():
        for task, t in d["tasks"].items():
            for rlabel, get, mlabel, higher in _task_rows(task):
                for key, bb in t["results"].items():
                    for kind, v in bb.items():
                        reps = [(r.get("seed"), r) for r in v.get("_seeds", [])] or [(None, v)]
                        for seed, cell in reps:
                            try:
                                val = get(cell)
                            except (KeyError, TypeError):
                                val = None
                            if val is None:
                                continue
                            add(family="downstream", dataset=name, task=task,
                                backbone=kind,
                                metric=mlabel + (f" ({rlabel})" if rlabel else ""),
                                higher_is_better=higher, method_key=key,
                                method=d["encoders"].get(key, key), seed=seed,
                                value=float(val))
    for res in suites:
        lg = res["parts"].get("local_global") or {}
        for kind, encs in lg.items():
            for e, row in encs.items():
                for src, pd_ in row["per_dataset"].items():
                    reps = [(r.get("seed"), r) for r in pd_.get("seeds", [])] or [(None, pd_)]
                    for seed, cell in reps:
                        for m, higher in SIM_METRICS:
                            add(family="similarity", dataset=src, task="similarity",
                                backbone=kind, metric=m, higher_is_better=higher,
                                method_key=e, method=row["name"], seed=seed,
                                value=float(cell["local"][m]))
                        for dst, g in cell["global"].items():
                            add(family="similarity", dataset=f"{src}->{dst}",
                                task="zero-shot similarity", backbone=kind,
                                metric="MRR", higher_is_better=True,
                                method_key=e, method=row["name"], seed=seed,
                                value=float(list(g.values())[-1]))
    return rows


def _setting(r):
    return (r["family"], r["dataset"], r["task"], r["backbone"], r["metric"])


def mean_rows(rows):
    """Collapse seeds: one row per setting and method, value = mean."""
    acc = {}
    for r in rows:
        acc.setdefault(_setting(r) + (r["method_key"],), []).append(r)
    out = []
    for rs in acc.values():
        m = dict(rs[0])
        m["value"] = float(np.mean([x["value"] for x in rs]))
        m["n_seeds"] = len(rs)
        m["seed"] = None
        out.append(m)
    return out


def write_overview_csv(rows, path):
    cols = ["family", "dataset", "task", "backbone", "metric",
            "higher_is_better", "method", "seed", "value"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def tex_by_dataset(rows):
    """One LaTeX table per task: methods x backbone down, datasets across."""
    out = []
    groups = {}
    for r in mean_rows(rows):
        if r["family"] != "downstream":
            continue
        groups.setdefault((r["task"], r["metric"], r["higher_is_better"]), []).append(r)
    for (task, metric, higher), rs in groups.items():
        dsets = list(dict.fromkeys(r["dataset"] for r in rs))
        meths = list(dict.fromkeys((r["method"], r["backbone"]) for r in rs
                                   if r["method_key"] in MAIN))
        val = {(r["method"], r["backbone"], r["dataset"]): r["value"] for r in rs}
        best = {d: (max if higher else min)(
            [val[(m, k, d)] for m, k in meths if (m, k, d) in val] or [np.nan])
            for d in dsets}
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{{_tx(TASK_TITLE.get(task, task))}: {_tx(metric)} "
                 rf"on each dataset ({'higher' if higher else 'lower'} is better; "
                 r"best per dataset in bold).}",
                 r"\begin{tabular}{ll" + "r" * len(dsets) + "}", r"\toprule",
                 "Encoder & Backbone & " + " & ".join(_tx(d) for d in dsets) + r" \\",
                 r"\midrule"]
        for m, k in meths:
            cells = []
            for d in dsets:
                v = val.get((m, k, d))
                s = "--" if v is None else f"{v:.3f}"
                if v is not None and np.isclose(v, best[d]):
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            lines.append(f"{_tx(m)} & {k} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)


# ==========================================================================
# 2. critical-difference diagrams
# ==========================================================================
def make_blocks(rows, family, task=None, methods=None):
    """block label -> {method: score oriented so that higher is better}."""
    blocks = {}
    for r in rows:
        if r["family"] != family or (task and r["task"] != task):
            continue
        if methods and r["method_key"] not in methods:
            continue
        b = f"{r['dataset']} | {r['task']} | {r['backbone']} | {r['metric']}"
        if r.get("seed") is not None:
            b += f" | seed {r['seed']}"
        blocks.setdefault(b, {})[r["method"]] = (
            r["value"] if r["higher_is_better"] else -r["value"])
    if not blocks:
        return {}, []
    # methods present in the largest number of blocks define the comparison;
    # blocks missing one of them are dropped (ranks need complete blocks)
    counts = {}
    for sc in blocks.values():
        for m in sc:
            counts[m] = counts.get(m, 0) + 1
    top = max(counts.values())
    meths = [m for m in counts if counts[m] == top]
    blocks = {b: {m: sc[m] for m in meths} for b, sc in blocks.items()
              if all(m in sc and np.isfinite(sc[m]) for m in meths)}
    return blocks, meths


def _rank_desc(v):
    """Rank 1 = best (largest); ties share the average rank."""
    from scipy.stats import rankdata
    return rankdata(-np.asarray(v, float), method="average")


def cd_stats(blocks, meths, alpha_q=Q05):
    N, k = len(blocks), len(meths)
    S = np.array([[sc[m] for m in meths] for sc in blocks.values()], float)
    ranks = np.vstack([_rank_desc(row) for row in S])
    avg = ranks.mean(0)
    out = {"n_blocks": N, "k": k, "methods": meths,
           "avg_rank": {m: float(a) for m, a in zip(meths, avg)},
           "wins": {m: int((ranks[:, i] == 1).sum()) for i, m in enumerate(meths)},
           "tied_blocks": int(sum(len(set(np.round(row, 12))) < k for row in S)),
           "blocks": list(blocks)}
    out["cd"] = float(alpha_q[k] * np.sqrt(k * (k + 1) / (6.0 * N))) if k in alpha_q else None
    try:
        if k >= 3:
            from scipy.stats import friedmanchisquare
            st, p = friedmanchisquare(*[S[:, i] for i in range(k)])
            out["test"], out["p_value"] = "Friedman", float(p)
        else:
            # Sign test, not Wilcoxon: blocks mix metrics on different scales
            # (a rank near 1, a rate in [0, 1], minutes), and Wilcoxon ranks
            # the SIZE of the differences across blocks, so the large-scale
            # metrics would decide it. The sign test only uses who won each
            # block, which is also all the CD diagram itself uses.
            from scipy.stats import binomtest
            w0 = int((S[:, 0] > S[:, 1]).sum())
            w1 = int((S[:, 1] > S[:, 0]).sum())
            out["ties"] = int(N - w0 - w1)
            if w0 + w1 == 0:
                raise ValueError("all blocks tied")
            p = binomtest(w0, w0 + w1, 0.5).pvalue
            out["test"] = (f"sign test ({max(w0, w1)}-{min(w0, w1)}"
                           + (f", {out['ties']} ties" if out["ties"] else "") + ")")
            out["p_value"] = float(p)
    except ValueError:                      # e.g. all differences are zero
        out["test"], out["p_value"] = "n/a (no differences)", float("nan")
    return out


def _cliques(order, avg, cd):
    """Maximal runs of rank-sorted methods spanning less than cd."""
    spans, n = [], len(order)
    for i in range(n):
        j = i
        while j + 1 < n and avg[order[j + 1]] - avg[order[i]] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in spans):
            spans.append((i, j))
    return spans


def fig_cd(stats, title, name):
    meths, k, N = stats["methods"], stats["k"], stats["n_blocks"]
    avg = stats["avg_rank"]
    order = sorted(meths, key=lambda m: avg[m])
    cd = stats["cd"]
    spans = _cliques(order, avg, cd) if cd else []

    n_left = (k + 1) // 2
    depth = max(n_left, k - n_left)
    fig, ax = plt.subplots(figsize=(WIDTH_2COL * 0.8, 1.25 + 0.27 * depth
                                    + 0.14 * len(spans)))
    lo, hi = 1.0, float(k)
    side = 0.75 * (hi - lo) + 0.5          # room for the labels on each side
    ax.set_xlim(lo - side, hi + side)
    y_axis = 0.0
    ax.plot([lo, hi], [y_axis, y_axis], color="black", lw=1.0)
    for t in range(1, k + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.12], color="black", lw=0.8)
        ax.text(t, y_axis + 0.2, str(t), ha="center", va="bottom", fontsize=7)
    gap = 0.04 * (hi - lo) + 0.08
    ax.text(lo - gap, y_axis, r"better $\leftarrow$", ha="right", va="center",
            fontsize=6.5, color="0.35", style="italic")
    ax.text(hi + gap, y_axis, r"$\rightarrow$ worse", ha="left", va="center",
            fontsize=6.5, color="0.35", style="italic")
    ax.text((lo + hi) / 2, y_axis + 0.62, "average rank (1 = best)", ha="center",
            va="bottom", fontsize=6.5)

    # the critical difference itself, drawn to scale above the axis
    if cd:
        shown = min(cd, hi - lo)
        y_cd = y_axis + 1.25
        ax.plot([lo, lo + shown], [y_cd, y_cd], color="black", lw=1.6)
        for xx in (lo, lo + shown):
            ax.plot([xx, xx], [y_cd - 0.07, y_cd + 0.07], color="black", lw=1.0)
        ax.text(lo + shown / 2, y_cd + 0.12,
                f"CD = {cd:.2f}" + (" (longer than the axis)" if cd > hi - lo else ""),
                ha="center", va="bottom", fontsize=6.5)

    # bars joining methods that are NOT significantly different
    y0 = y_axis - 0.22
    for s, (i, j) in enumerate(spans):
        y = y0 - 0.16 * s
        ax.plot([avg[order[i]] - 0.04, avg[order[j]] + 0.04], [y, y],
                color="black", lw=2.6, solid_capstyle="butt")
    base = y0 - 0.16 * max(len(spans), 1) - 0.2

    # method labels: best half on the left, the rest on the right
    for idx, m in enumerate(order):
        left = idx < n_left
        level = idx if left else k - 1 - idx
        y = base - 0.36 * level
        x_end = ax.get_xlim()[0] + 0.15 if left else ax.get_xlim()[1] - 0.15
        col = _enc_color(m, idx)
        ax.plot([avg[m], avg[m], x_end], [y_axis, y, y], color="0.25", lw=0.7)
        # same family, same colour; the "+time+speed" variant is hollow so
        # identity never rests on colour alone
        hollow = "+time" in m
        ax.plot([avg[m]], [y_axis], marker="o", ms=5.5, zorder=5,
                markerfacecolor="white" if hollow else col,
                markeredgecolor=col if hollow else "white",
                markeredgewidth=1.4 if hollow else 0.8)
        ax.plot([x_end + (0.0 if left else 0.0)], [y], marker="o", ms=4, zorder=5,
                markerfacecolor="white" if hollow else col, markeredgecolor=col,
                markeredgewidth=1.0, clip_on=False)
        ax.text(x_end + (0.0 if left else 0.0), y + 0.09,
                f"{m}  ({avg[m]:.2f})" if left else f"({avg[m]:.2f})  {m}",
                ha="left" if left else "right", va="bottom", fontsize=6.5)
    ax.set_ylim(base - 0.36 * (depth - 1) - 0.25, y_axis + (1.75 if cd else 1.0))
    ax.axis("off")

    p = stats["p_value"]
    ptxt = "n/a" if not np.isfinite(p) else ("< 0.001" if p < 1e-3 else f"= {p:.3f}")
    verdict = ""
    if np.isfinite(p) and p >= 0.05:
        verdict = "; overall difference not significant"
    few = "  [few blocks: low power]" if N < 2 * k + 2 else ""
    ns, sp = stats.get("n_settings", N), stats.get("seeds_per_setting", 1)
    what = (f"{N} blocks ({ns} settings x {sp:g} seeds)" if sp and sp > 1
            else f"{N} blocks")
    ax.set_title(f"{title}\n{what}, {stats['test']} p {ptxt}{verdict}{few}",
                 fontsize=7.5)
    fig.tight_layout()
    return (name, fig)


def build_cd(rows, min_blocks=3):
    """All CD figures the batch supports, plus the statistics behind them."""
    figs, allstats, notes = [], {}, []
    specs = [("similarity", None, None, "Trajectory similarity (local + zero-shot)",
              "fig_cd_similarity"),
             ("downstream", None, MAIN, "Downstream tasks, all datasets",
              "fig_cd_downstream")]
    tasks = list(dict.fromkeys(r["task"] for r in rows if r["family"] == "downstream"))
    specs += [("downstream", t, MAIN, f"{TASK_TITLE.get(t, t)}", f"fig_cd_downstream_{t}")
              for t in tasks]
    for family, task, methods, title, name in specs:
        blocks, meths = make_blocks(rows, family, task, methods)
        if len(meths) < 2:
            continue
        if len(blocks) < min_blocks:
            notes.append(f"{name}: only {len(blocks)} block(s), need {min_blocks}; skipped")
            continue
        if len(meths) not in Q05:
            notes.append(f"{name}: {len(meths)} methods, CD table covers 2-10; skipped")
            continue
        st = cd_stats(blocks, meths)
        st["n_settings"] = len({b.split(" | seed ")[0] for b in blocks})
        st["seeds_per_setting"] = round(len(blocks) / max(st["n_settings"], 1), 2)
        allstats[name] = st
        figs.append(fig_cd(st, title, name))
    return figs, allstats, notes


def tex_cd(allstats):
    if not allstats:
        return ""
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Average ranks behind the critical-difference diagrams "
             r"(rank 1 = best in a block; a block is one dataset, task, backbone "
             r"and metric). Blocks sharing a dataset are not independent, so the "
             r"tests are indicative.}",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Comparison & Method & Avg. rank & Wins & Blocks \\", r"\midrule"]
    for name, st in allstats.items():
        label = name.replace("fig_cd_", "").replace("_", " ")
        for i, m in enumerate(sorted(st["methods"], key=lambda m: st["avg_rank"][m])):
            lines.append((_tx(label) if i == 0 else "") + f" & {_tx(m)} & "
                         f"{st['avg_rank'][m]:.2f} & {st['wins'][m]} & "
                         + (str(st["n_blocks"]) if i == 0 else "") + r" \\")
        p = st["p_value"]
        lines.append(rf"\multicolumn{{5}}{{r}}{{\footnotesize {st['test']} "
                     + (f"$p={p:.3f}$" if np.isfinite(p) else "n/a")
                     + (rf", CD$_{{0.05}}={st['cd']:.2f}$" if st["cd"] else "")
                     + r"} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ==========================================================================
# 3. GEO against GPE, paired over seeds
# ==========================================================================
PAIRS = (("geo", "gpe", ""), ("geo_ts", "gpe_ts", " +time+speed"))


def paired_tests(rows):
    """For every setting and each matched pair (GEO vs GPE with the same extra
    channels): the per-seed differences, their mean as a relative improvement,
    a 95% interval and a paired t-test over seeds (Holm-corrected across all
    settings). Seeds share splits and initialisation order, so they pair."""
    from scipy import stats as st
    cell = {}
    for r in rows:
        cell.setdefault(_setting(r), {}).setdefault(r["method_key"], {})[r["seed"]] = r
    out = []
    for setting, meths in cell.items():
        for a, b, tag in PAIRS:
            if a not in meths or b not in meths:
                continue
            seeds = sorted(set(meths[a]) & set(meths[b]), key=lambda x: (x is None, x))
            if not seeds:
                continue
            ra = meths[a][seeds[0]]
            sign = 1.0 if ra["higher_is_better"] else -1.0
            va = np.array([meths[a][sd]["value"] for sd in seeds])
            vb = np.array([meths[b][sd]["value"] for sd in seeds])
            d = sign * (va - vb)
            scale = 100.0 / max(abs(vb.mean()), 1e-12)
            n = len(d)
            rec = {"family": setting[0], "dataset": setting[1], "task": setting[2],
                   "backbone": setting[3], "metric": setting[4], "pair": "GEO vs GPE" + tag,
                   "n_seeds": n, "geo_mean": float(va.mean()), "gpe_mean": float(vb.mean()),
                   "rel_improvement_pct": float(d.mean() * scale),
                   "geo_better_seeds": int((d > 0).sum()),
                   "ci95_pct": float("nan"), "p_value": float("nan")}
            if n >= 3:
                sd_ = d.std(ddof=1)
                if sd_ > 0:
                    rec["ci95_pct"] = float(st.t.ppf(0.975, n - 1) * sd_ / np.sqrt(n) * scale)
                    rec["p_value"] = float(st.ttest_rel(sign * va, sign * vb).pvalue)
                else:                       # identical difference in every seed
                    rec["ci95_pct"] = 0.0
                    rec["p_value"] = 0.0 if d.mean() != 0 else 1.0
            out.append(rec)
    # Holm correction over every test that could be run
    idx = [i for i, r in enumerate(out) if np.isfinite(r["p_value"])]
    order = sorted(idx, key=lambda i: out[i]["p_value"])
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out[i]["p_value"]))
        out[i]["p_holm"] = float(running)
    for r in out:
        r.setdefault("p_holm", float("nan"))
        r["significant"] = bool(np.isfinite(r["p_holm"]) and r["p_holm"] < 0.05)
    return out


def fig_paired(tests, family, name, title):
    T = [t for t in tests if t["family"] == family]
    if not T:
        return None
    T.sort(key=lambda t: (t["task"], t["pair"], t["dataset"], t["backbone"], t["metric"]))
    n = len(T)
    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 0.9 + 0.17 * n))
    y = np.arange(n)[::-1]
    x = np.array([t["rel_improvement_pct"] for t in T])
    ci = np.array([t["ci95_pct"] for t in T])
    lim = np.nanpercentile(np.abs(np.r_[x, x + np.nan_to_num(ci), x - np.nan_to_num(ci)]), 95)
    lim = max(float(lim) * 1.25, 1.0)
    ax.axvline(0, color="black", lw=0.8)
    for i, t in enumerate(T):
        col = "#0072B2" if x[i] > 0 else "#D55E00"
        if np.isfinite(ci[i]):
            ax.plot([x[i] - ci[i], x[i] + ci[i]], [y[i], y[i]], color=col, lw=1.0,
                    solid_capstyle="butt")
        ax.plot(np.clip(x[i], -lim, lim), y[i], marker="o", ms=4.5,
                markerfacecolor=col if t["significant"] else "white",
                markeredgecolor=col, markeredgewidth=1.0, zorder=3)
        if abs(x[i]) > lim:                     # off the axis: say the value
            ax.annotate(f"{x[i]:+.0f}%", (np.sign(x[i]) * lim, y[i]), fontsize=5,
                        textcoords="offset points",
                        xytext=(-4 if x[i] > 0 else 4, 3),
                        ha="right" if x[i] > 0 else "left")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t['dataset']} | {t['task']}{t['pair'][10:]} | "
                        f"{t['backbone']} | {t['metric']}" for t in T], fontsize=5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("relative improvement of GEO over GPE, %   "
                  r"(GPE better $\leftarrow$ 0 $\rightarrow$ GEO better)")
    ax.grid(axis="y", visible=False)
    wins = int((x > 0).sum())
    sg = sum(1 for t in T if t["significant"] and t["rel_improvement_pct"] > 0)
    sl = sum(1 for t in T if t["significant"] and t["rel_improvement_pct"] < 0)
    ns = sorted({t["n_seeds"] for t in T})
    seeds_txt = f"{ns[0]}" if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
    note = ("" if max(ns) >= 3 else
            "  [fewer than 3 seeds: no intervals or tests]")
    fig.suptitle(f"{title}: GEO ahead in {wins} of {n} settings; significantly "
                 f"better in {sg}, significantly worse in {sl}\n"
                 f"{seeds_txt} seed(s) per setting; bars = 95% interval over seeds; "
                 f"filled = significant after Holm correction{note}", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.12 / fig.get_figheight()))
    return (name, fig)


def write_paired_csv(tests, path):
    cols = ["family", "dataset", "task", "backbone", "metric", "pair", "n_seeds",
            "geo_mean", "gpe_mean", "rel_improvement_pct", "ci95_pct",
            "geo_better_seeds", "p_value", "p_holm", "significant"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in tests:
            w.writerow([t[c] for c in cols])


def tex_paired(tests):
    if not tests:
        return ""
    groups = {}
    for t in tests:
        groups.setdefault((t["family"], t["task"], t["pair"]), []).append(t)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{GEO against GPE with matched channels, paired over seeds. "
             r"A setting is one dataset, sequence model and metric. `Sig.' counts "
             r"settings where the paired $t$-test over seeds is significant at 5\% "
             r"after Holm correction across all settings.}",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Task & Pair & Settings & GEO ahead & Sig. better & Sig. worse & "
             r"Median gain (\%) \\", r"\midrule"]
    for (fam, task, pair), ts in groups.items():
        x = np.array([t["rel_improvement_pct"] for t in ts])
        lines.append(
            f"{_tx(task)} & {_tx(pair)} & {len(ts)} & {int((x > 0).sum())} & "
            f"{sum(t['significant'] and t['rel_improvement_pct'] > 0 for t in ts)} & "
            f"{sum(t['significant'] and t['rel_improvement_pct'] < 0 for t in ts)} & "
            f"{np.median(x):+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ==========================================================================
# 4. relocated-city test
# ==========================================================================
# Rows of the main figure: colour = encoding (GPE orange, GEO blue), line =
# frame (dotted, hollow = left where it was trained; solid, filled = re-fitted
# on the moved city without labels). Built by _main_rows from what the
# results file contains.
_ORANGE, _BLUE = "#D55E00", "#0072B2"
MAIN_STYLE = {
    "gpe_fixed": dict(color=_ORANGE, ls=":", marker="o", mfc="white", lw=1.3),
    "gpe_refit": dict(color=_ORANGE, ls="-", marker="o", lw=1.6),
    "geo_fixed": dict(color=_BLUE, ls=":", marker="s", mfc="white", lw=1.3),
    "geo_refit": dict(color=_BLUE, ls="-", marker="s", lw=1.9),
}
MAIN_LABEL = {
    "gpe_fixed": "GPE, as trained (absolute lon/lat)",
    "gpe_refit": "GPE, in the re-fitted frame (diagnostic)",
    "geo_fixed": "GEO, frame left at the training city",
    "geo_refit": "GEO, frame re-fitted on the moved city",
}

# Every row, for the appendix figure
RELOC_LABEL = {
    "GPE": "GPE, as trained",
    "GPE + space shift": "GPE + bounding-box remap (ablation)",
    "GEO (fixed frame)": "GEO, frame left at the training city",
    "GEO (origin re-fit)": "GEO, origin re-fitted (axes kept)",
    "GEO (canonical frame)": "GEO, origin + axes re-fitted",
    "GPE in canonical frame": "GPE in GEO's re-fitted frame (diagnostic)",
}
RELOC_STYLE = {
    "GPE":                    dict(color=_ORANGE, ls=":", marker="o", mfc="white", lw=1.3),
    "GPE + space shift":      dict(color="#E69F00", ls="--", marker="P", lw=1.1),
    "GEO (fixed frame)":      dict(color=_BLUE, ls=":", marker="s", mfc="white", lw=1.3),
    "GEO (origin re-fit)":    dict(color="#56B4E9", ls="--", marker="D", lw=1.1),
    "GEO (canonical frame)":  dict(color=_BLUE, ls="-", marker="s", lw=1.9),
    "GPE in canonical frame": dict(color=_ORANGE, ls="-", marker="o", lw=1.6),
}
TASK_NAME = {"similarity": "trajectory similarity (control)",
             "eta": "travel-time estimation",
             "mode": "transport-mode detection"}


def _main_rows(res):
    """[(key, results row)] for the four lines of the main figure, using the
    best re-fitted GEO row the file has."""
    have = set(res["rows"])
    pick = [("gpe_fixed", "GPE"), ("geo_fixed", "GEO (fixed frame)"),
            ("geo_refit", next((r for r in ("GEO (canonical frame)", "GEO (origin re-fit)")
                                if r in have), None)),
            ("gpe_refit", "GPE in canonical frame")]
    return [(k, r) for k, r in pick if r in have]


def _reloc_panels(res):
    """task -> [(dataset, kind, data)]"""
    out = {}
    for name, bk in res["datasets"].items():
        for kind, bt in bk.items():
            for tname, d in bt.items():
                out.setdefault(tname, []).append((name, kind, d))
    return out


def _n_seeds(res):
    return len(res.get("seeds") or [0])


def _plot_sweeps(res, tname, panels, lines, title):
    """One row of panels per dataset x model, one column per sweep. `lines` is
    [(results row, label, style)]. Each panel starts with the value with the
    city in place (left of the thin rule), then the sweep; shaded bands are
    +-1 std over seeds."""
    sweeps = list(panels[0][2]["sweeps"])
    fig, axes = plt.subplots(len(panels), len(sweeps), squeeze=False,
                             figsize=(WIDTH_2COL, 1.75 * len(panels) + 1.25),
                             sharey="row")
    for r, (name, kind, d) in enumerate(panels):
        metric, hi = d["metric"], d["higher_is_better"]
        for c, sweep in enumerate(sweeps):
            ax, sw = axes[r][c], d["sweeps"][sweep]
            x = np.arange(len(sw["values"])) + 1.0
            for j, (row, label, st) in enumerate(lines):
                cells = sw["rows"].get(row)
                if not cells:
                    continue
                st = dict(st)
                mfc = st.pop("mfc", st["color"])
                y = np.array([cell[metric] for cell in cells])
                ax.plot(x, y, ms=3.6, label=label, mfc=mfc, mec=st["color"],
                        mew=0.9, zorder=3, **st)
                if metric + "_std" in cells[0]:
                    e = np.array([cell[metric + "_std"] for cell in cells])
                    ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.13,
                                    lw=0, zorder=1)
                ip = d.get("in_place", {}).get(row)
                if ip is not None:          # the city where it is
                    dx = (j - (len(lines) - 1) / 2) * 0.12
                    ax.errorbar([dx], [ip[metric]], yerr=[ip.get(metric + "_std", 0.0)],
                                fmt=st["marker"], ms=4.2, color=st["color"], mfc=mfc,
                                mec=st["color"], mew=0.9, capsize=1.5, elinewidth=0.8,
                                zorder=4)
            ax.axvline(0.5, color="0.6", lw=0.6)
            ax.set_xlim(-0.6, len(x) + 0.5)
            ax.set_xticks(np.r_[0.0, x])
            labels = sw.get("labels") or [f"{v:g}" for v in sw["values"]]
            ax.set_xticklabels(["in\nplace"] + list(labels),
                               fontsize=6 if len(labels) < 6 else 5.5)
            ax.grid(axis="y", color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            if sweep == "latitude":
                ax.set_xlabel(f"latitude of the city centre ({res['units'][sweep]}; "
                              f"real: {d['centre_lonlat'][1]:.0f})", fontsize=6)
            else:
                ax.set_xlabel(f"{sweep} ({res['units'][sweep]})", fontsize=6)
            if r == 0:
                ax.set_title(sweep, fontsize=7.5)
            if c == 0:
                arrow = r"$\uparrow$" if hi else r"$\downarrow$"
                ax.set_ylabel(f"{name} / {kind}\n{metric}{UNITS.get(metric, '')} {arrow}",
                              fontsize=6.5)
    h_, l_ = axes[0][0].get_legend_handles_labels()
    ns = _n_seeds(res)
    fig.suptitle(f"{title}" + (f"  ({ns} seeds, band $\\pm$1 std)" if ns > 1 else ""),
                 fontsize=8.5)
    H = fig.get_figheight()
    fig.texts[-1].set_y(1 - 0.08 / H)
    fig.texts[-1].set_va("top")
    fig.tight_layout()
    n_leg_rows = int(np.ceil(len(h_) / 2))
    fig.subplots_adjust(top=1 - (0.55 + 0.16 * n_leg_rows) / H)
    fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=6.4, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.30 / H), handlelength=3.4,
               columnspacing=2.5)
    return fig


def fig_relocate(res):
    """Main figure per task: four lines (encoding x frame); the appendix
    variant fig_relocate_all_<task> keeps every row."""
    figs = []
    main = _main_rows(res)
    # legend order: with ncol=2 matplotlib fills column by column, so this
    # gives rows = frame (fixed / re-fitted), columns = encoding (GPE / GEO)
    order = [k for k in ("gpe_fixed", "gpe_refit", "geo_fixed", "geo_refit")
             if k in dict(main)]
    lines = [(dict(main)[k], MAIN_LABEL[k], MAIN_STYLE[k]) for k in order]
    every = [(r, RELOC_LABEL.get(r, r),
              RELOC_STYLE.get(r, dict(color="black", ls="-", marker="o", lw=1)))
             for r in res["rows"]]
    for tname, panels in _reloc_panels(res).items():
        t = TASK_NAME.get(tname, tname)
        figs.append((f"fig_relocate_{tname}", _plot_sweeps(
            res, tname, panels, lines,
            f"Relocated-city test, {t}")))
        figs.append((f"fig_relocate_all_{tname}", _plot_sweeps(
            res, tname, panels, every,
            f"Relocated-city test, {t}, every variant")))
    s = fig_relocate_summary(res)
    if s:
        figs.append(s)
    return figs


def _worst(d, row, metric, hi):
    """Worst mean value of `row` over every move of every sweep, and where."""
    best = None
    for sweep, sw in d["sweeps"].items():
        for v, cell in zip(sw["values"], sw["rows"].get(row) or []):
            val = cell[metric]
            if best is None or (val < best[0] if hi else val > best[0]):
                best = (val, sweep, v)
    return best


def _change_pct(in_place, worst, hi):
    """Signed relative change, negative = worse."""
    if not in_place:
        return float("nan")
    ch = (worst - in_place) / abs(in_place) * 100.0
    return ch if hi else -ch


def _pct(ch):
    return "0%" if abs(ch) < 0.5 else f"{ch:+.0f}%"


def fig_relocate_summary(res):
    """Dumbbells: one row of panels per dataset / model, one column per task;
    in each panel every method's score with the city in place (filled) and
    its worst score over every move (hollow ring), with the relative change.
    When the two coincide the marker is a ring with a dot: nothing was lost."""
    main = _main_rows(res)
    if not main:
        return None
    short = {"gpe_fixed": "GPE, as trained", "gpe_refit": "GPE, re-fitted frame",
             "geo_fixed": "GEO, fixed frame", "geo_refit": "GEO, re-fitted frame"}
    items = [(dict(main)[k], short[k], MAIN_STYLE[k])
             for k in ("geo_refit", "geo_fixed", "gpe_refit", "gpe_fixed")
             if k in dict(main)]
    return _dumbbells(res, items, "fig_relocate_summary",
                      "Relocated-city test, summary")


def _dumbbells(res, items, fig_name, title, worst_label="worst over every move",
               in_label="city in place"):
    """items: [(results row, short label, style)], top to bottom."""
    panels = _reloc_panels(res)
    tasks = [t for t in ("eta", "mode", "similarity") if t in panels]
    tasks += [t for t in panels if t not in tasks]
    rows = sorted({(n, k) for t in tasks for n, k, _ in panels[t]})
    fig, axes = plt.subplots(len(rows), len(tasks), squeeze=False,
                             figsize=(WIDTH_2COL, 0.82 * len(rows) + 1.0))
    metric_of = {}
    for c, tname in enumerate(tasks):
        got = {(n, k): d for n, k, d in panels[tname]}
        for r, key in enumerate(rows):
            ax = axes[r][c]
            d = got.get(key)
            if d is None:
                ax.axis("off")
                continue
            m, hi = d["metric"], d["higher_is_better"]
            metric_of[tname] = (m, hi)
            for i, (row, _lab, st) in enumerate(items):
                ip = d["in_place"].get(row, {}).get(m)
                w = _worst(d, row, m, hi)
                if ip is None or w is None:
                    continue
                y = len(items) - 1 - i
                ax.plot([ip, w[0]], [y, y], color=st["color"], lw=1.2, ls=":",
                        zorder=2)
                ax.plot([w[0]], [y], marker=st["marker"], ms=5.6, mfc="white",
                        mec=st["color"], mew=1.0, zorder=3, ls="none")
                ax.plot([ip], [y], marker=st["marker"], ms=3.4, color=st["color"],
                        zorder=4, ls="none")
                ch = _change_pct(ip, w[0], hi)
                ax.annotate(_pct(ch), xy=(1.02, y), xycoords=("axes fraction", "data"),
                            fontsize=5.8, va="center",
                            color="#B00000" if ch < -5 else "0.4")
            ax.set_ylim(-0.6, len(items) - 0.4)
            ax.set_yticks(range(len(items)))
            ax.set_yticklabels([it[1] for it in items[::-1]] if c == 0 else [],
                               fontsize=6)
            ax.tick_params(axis="x", labelsize=5.8, pad=1.5)
            ax.grid(axis="x", color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            ax.margins(x=0.15)
            if c == 0:
                ax.set_title(f"{key[0]} / {key[1]}", loc="left", fontsize=6.8,
                             fontweight="bold", x=-0.62, pad=2)
    for c, tname in enumerate(tasks):       # metric under the bottom panel
        if tname in metric_of:
            m, hi = metric_of[tname]
            axes[-1][c].set_xlabel(f"{m}{UNITS.get(m, '')}, "
                                   f"{'higher' if hi else 'lower'} is better",
                                   fontsize=6.2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color="0.3", ls="none", ms=3.4,
                label=in_label),
         Line2D([], [], marker="o", mfc="white", mec="0.3", ls="none", ms=5.6,
                label=worst_label),
         Line2D([], [], marker="o", mfc="0.3", mec="0.3", ls="none", ms=0,
                label="% = change, negative = worse")]
    ns = _n_seeds(res)
    fig.suptitle(title + (f" (mean of {ns} seeds)" if ns > 1 else ""),
                 fontsize=8.5)
    H = fig.get_figheight()
    fig.texts[-1].set_y(1 - 0.06 / H)
    fig.texts[-1].set_va("top")
    fig.tight_layout(w_pad=2.6, h_pad=0.9)
    fig.subplots_adjust(top=1 - 0.78 / H, left=0.17, right=0.95)
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.22 / H), handletextpad=0.3)
    for c, tname in enumerate(tasks):       # task names over the columns
        bb = axes[0][c].get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.24 / H, TASK_NAME.get(tname, tname),
                 ha="center", va="bottom", fontsize=7.5)
    return (fig_name, fig)


def tex_relocate(res):
    out = []
    for tname, panels in _reloc_panels(res).items():
        metric, hi = panels[0][2]["metric"], panels[0][2]["higher_is_better"]
        pick = min if hi else max
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{Relocated-city test, {_tx(tname)} ({_tx(metric)}, "
                 rf"{'higher' if hi else 'lower'} is better): value with the test "
                 r"city in place, the worst value along each sweep, and the "
                 r"change from in place to the worst move overall (negative = "
                 r"worse). All rows are zero-shot; the four main rows come first.}",
                 r"\begin{tabular}{llrrrrr}", r"\toprule",
                 r"Dataset & Method & In place & Latitude & Rotation & Translation"
                 r" & $\Delta$ (\%) \\",
                 r"\midrule"]
        main = [r for _, r in _main_rows(res)]
        rows = main + [r for r in res["rows"] if r not in main]
        for name, kind, d in panels:
            for i, row in enumerate(rows):
                if row not in d["in_place"]:
                    continue
                cells = [f"{pick(c[metric] for c in d['sweeps'][sw]['rows'][row]):.3f}"
                         if sw in d["sweeps"] else "--"
                         for sw in ("latitude", "rotation", "translation")]
                ip = d["in_place"][row][metric]
                w = _worst(d, row, metric, hi)
                ch = f"{_change_pct(ip, w[0], hi):+.1f}" if w else "--"
                ch = "0.0" if ch in ("+0.0", "-0.0") else ch
                if i == len(main) and main:
                    lines.append(r"\cmidrule(l){2-7}")
                lines.append((f"{_tx(name)} ({kind})" if i == 0 else "")
                             + f" & {_tx(RELOC_LABEL.get(row, row))} & {ip:.3f} & "
                             + " & ".join(cells) + f" & {ch}" + r" \\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines += [r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)


# ==========================================================================
# 5. robustness: GEO (all channels) vs GPE as published, under rigid motions
# ==========================================================================
ROBUST_STYLE = {
    "GPE (paper)": MAIN_STYLE["gpe_fixed"],
    "GPE (paper) in GEO frame": MAIN_STYLE["gpe_refit"],
    "GEO (full, fixed frame)": MAIN_STYLE["geo_fixed"],
    "GEO (full)": MAIN_STYLE["geo_refit"],
}
ROBUST_SHORT = {
    "GPE (paper)": "GPE, as published",
    "GPE (paper) in GEO frame": "GPE in GEO's frame",
    "GEO (full, fixed frame)": "GEO, frame not re-fitted",
    "GEO (full)": "GEO, all channels",
}


def _robust_labels(res):
    city = res.get("scope", "city") == "city"
    return {
        "GPE (paper)": "GPE as published (raw lon/lat, spatial only)",
        "GPE (paper) in GEO frame": "GPE in GEO's re-fitted frame (diagnostic)",
        "GEO (full, fixed frame)": "GEO, all channels, frame not re-fitted (diagnostic)",
        "GEO (full)": ("GEO, all channels, frame re-fitted without labels" if city
                       else "GEO, all channels (training frame)"),
    }


def fig_robust(res):
    labels = _robust_labels(res)
    order = [r for r in ("GPE (paper)", "GPE (paper) in GEO frame",
                         "GEO (full, fixed frame)", "GEO (full)") if r in res["rows"]]
    lines = [(r, labels[r], ROBUST_STYLE[r]) for r in order]
    how = ("test city moved as one rigid body" if res.get("scope", "city") == "city"
           else "each test trip moved on its own")
    figs = []
    for tname, panels in _reloc_panels(res).items():
        t = TASK_NAME.get(tname, tname)
        figs.append((f"fig_robust_{tname}", _plot_sweeps(
            res, tname, panels, lines, f"Robustness, {t}: {how}")))
    items = [(r, ROBUST_SHORT[r], ROBUST_STYLE[r]) for r in order[::-1]]
    figs.append(_dumbbells(res, items, "fig_robust_summary",
                           "Robustness to translations and rotations",
                           worst_label="worst over every transformation",
                           in_label="untransformed test set"))
    return figs


def tex_robust(res):
    out = []
    sweeps = [s for s in ("translation", "rotation", "combined")]
    labels = _robust_labels(res)
    for tname, panels in _reloc_panels(res).items():
        metric, hi = panels[0][2]["metric"], panels[0][2]["higher_is_better"]
        pick = min if hi else max
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{Robustness to rigid motions, {_tx(tname)} "
                 rf"({_tx(metric)}, {'higher' if hi else 'lower'} is better; "
                 rf"{_tx(res.get('scope', 'city'))} scope): value with nothing "
                 r"moved, worst value over each family of transformations, and "
                 r"the change from in place to the worst one (negative = worse). "
                 r"GEO uses every channel; GPE is the published method.}",
                 r"\begin{tabular}{llrrrrr}", r"\toprule",
                 r"Dataset & Method & In place & Translation & Rotation & "
                 r"Transl.\,+\,rot. & $\Delta$ (\%) \\",
                 r"\midrule"]
        rows = [r for r in ("GEO (full)", "GPE (paper)", "GEO (full, fixed frame)",
                            "GPE (paper) in GEO frame") if r in res["rows"]]
        for name, kind, d in panels:
            for i, row in enumerate(rows):
                cells = [f"{pick(c[metric] for c in d['sweeps'][sw]['rows'][row]):.3f}"
                         if sw in d["sweeps"] else "--" for sw in sweeps]
                ip = d["in_place"][row][metric]
                w = _worst(d, row, metric, hi)
                ch = f"{_change_pct(ip, w[0], hi):+.1f}" if w else "--"
                ch = "0.0" if ch in ("+0.0", "-0.0") else ch
                if i == 2:
                    lines.append(r"\cmidrule(l){2-7}")
                lines.append((f"{_tx(name)} ({kind})" if i == 0 else "")
                             + f" & {_tx(labels[row])} & {ip:.3f} & "
                             + " & ".join(cells) + f" & {ch}" + r" \\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines += [r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)


# ==========================================================================
# entry point used by the figure driver
# ==========================================================================
def build(datasets, out_dir):
    """Returns ([(name, fig)], [latex tables]). Writes the CSV / JSON side
    files into out_dir. `datasets` is the list of parsed results JSONs."""
    down, suites = split_inputs(datasets)
    if not down and not suites:
        return [], []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = overview_rows(down, suites)
    figs = []
    if down:
        figs += fig_by_dataset(down)
    figs += fig_similarity_by_dataset(suites)
    cdf, stats, notes = build_cd(rows)
    figs += cdf
    for n in notes:
        print(f"  - CD: {n}")
    tests = paired_tests(rows)
    for fam, nm, ttl in (("similarity", "fig_paired_similarity", "Similarity"),
                         ("downstream", "fig_paired_downstream", "Downstream tasks")):
        f = fig_paired(tests, fam, nm, ttl)
        if f:
            figs.append(f)
    write_overview_csv(rows, out_dir / "results_by_dataset.csv")
    if tests:
        write_paired_csv(tests, out_dir / "paired_tests.csv")
    if stats:
        (out_dir / "cd_stats.json").write_text(json.dumps(stats, indent=2))
    return figs, [tex_by_dataset(rows), tex_cd(stats), tex_paired(tests)]
