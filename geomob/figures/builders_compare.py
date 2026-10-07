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
    backbone, metric, method)."""
    rows = []
    for name, d in down.items():
        for task, t in d["tasks"].items():
            for rlabel, get, mlabel, higher in _task_rows(task):
                for key, bb in t["results"].items():
                    for kind, v in bb.items():
                        val = get(v)
                        if val is None:
                            continue
                        rows.append({
                            "family": "downstream", "dataset": name, "task": task,
                            "backbone": kind,
                            "metric": mlabel + (f" ({rlabel})" if rlabel else ""),
                            "higher_is_better": higher, "method_key": key,
                            "method": d["encoders"].get(key, key),
                            "value": float(val)})
    for res in suites:
        lg = res["parts"].get("local_global") or {}
        for kind, encs in lg.items():
            for e, row in encs.items():
                for src, pd_ in row["per_dataset"].items():
                    for m, higher in SIM_METRICS:
                        rows.append({
                            "family": "similarity", "dataset": src, "task": "similarity",
                            "backbone": kind, "metric": m, "higher_is_better": higher,
                            "method_key": e, "method": row["name"],
                            "value": float(pd_["local"][m])})
                    for dst, g in pd_["global"].items():
                        rows.append({
                            "family": "similarity", "dataset": f"{src}->{dst}",
                            "task": "zero-shot similarity", "backbone": kind,
                            "metric": "MRR", "higher_is_better": True,
                            "method_key": e, "method": row["name"],
                            "value": float(list(g.values())[-1])})
    return rows


def write_overview_csv(rows, path):
    cols = ["family", "dataset", "task", "backbone", "metric",
            "higher_is_better", "method", "value"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def tex_by_dataset(rows):
    """One LaTeX table per task: methods x backbone down, datasets across."""
    out = []
    groups = {}
    for r in rows:
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
    ax.set_title(f"{title}\n{N} blocks, {stats['test']} p {ptxt}{verdict}{few}",
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
    write_overview_csv(rows, out_dir / "results_by_dataset.csv")
    if stats:
        (out_dir / "cd_stats.json").write_text(json.dumps(stats, indent=2))
    return figs, [tex_by_dataset(rows), tex_cd(stats)]
