"""Figures and LaTeX tables for the two comparison runners.

`gpe_suite` results are laid out as GPE's own tables (4, 6, 8, 9) so the two
papers can be read side by side; `downstream` results get one figure per task.
Tables are returned as LaTeX strings and written to `tables.tex` by the driver.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import (GREY, HATCHES, MARKERS, PALETTE, WIDTH_2COL,
                    annotate_bars, color_for, shorten)

TASK_METRICS = {                     # headline, secondary, higher-is-better
    "tul": ("acc@1", "macro_F1", True),
    "eta": ("MAE", "MAPE", False),
    "mode": ("macro_F1", "acc@1", True),
}
UNITS = {"MAE": " (min)", "RMSE": " (min)", "MAPE": " (%)"}


def _enc_color(name, i):
    base = "GPE" if name.startswith("GPE") else "GEO(i)+disp" if name.startswith("GEO") else name
    return color_for(base, i)


# ==========================================================================
# downstream
# ==========================================================================
def fig_downstream(res):
    figs = []
    enc_names = res["encoders"]
    for task, d in res["tasks"].items():
        if task == "anomaly":
            figs += _fig_downstream_anomaly(res, d)
            continue
        head, sec, higher = TASK_METRICS[task]
        R = d["results"]
        keys = list(R)
        kinds = res["backbones"]
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.6))
        for ax, metric in zip(axes, (head, sec)):
            x = np.arange(len(keys))
            w = 0.8 / len(kinds)
            for j, kind in enumerate(kinds):
                vals = [R[k][kind].get(metric, np.nan) for k in keys]
                err = [R[k][kind].get(metric + "_std", 0.0) for k in keys]
                bars = ax.bar(x + (j - (len(kinds) - 1) / 2) * w, vals, w,
                              yerr=err if any(err) else None, capsize=2,
                              color=[_enc_color(enc_names[k], i) for i, k in enumerate(keys)],
                              edgecolor="black", lw=0.3, hatch=HATCHES[j % len(HATCHES)],
                              label=kind)
                annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5)
            for bi, (bname, b) in enumerate(d["baselines"].items()):
                v = b if not isinstance(b, dict) else b.get(metric)
                if v is not None and np.isfinite(v):
                    ax.axhline(v, color=GREY, lw=0.8, ls=["--", ":", "-."][bi % 3])
                    ax.annotate(bname, (len(keys) - 0.5, v), fontsize=5.5,
                                color="0.35", ha="right", va="bottom")
            ax.set_xticks(x)
            ax.set_xticklabels([shorten(_task_label(enc_names[k], task), 18) for k in keys],
                               rotation=18, ha="right", fontsize=6)
            arrow = r"$\uparrow$" if (higher if metric == head else metric != "MAPE") else r"$\downarrow$"
            if metric in ("MAE", "RMSE", "MAPE"):
                arrow = r"$\downarrow$"
            ax.set_ylabel(metric + UNITS.get(metric, "") + " " + arrow)
            ax.legend(fontsize=6, title="backbone", title_fontsize=6)
        axes[0].set_title({"tul": "Trajectory-user linking",
                           "eta": "Travel-time estimation",
                           "mode": "Transportation-mode detection"}[task]
                          + f" ({res.get('dataset', '')})")
        fig.tight_layout()
        figs.append((f"fig_downstream_{task}", fig))
        if task == "mode" and d["meta"].get("per_class_f1"):
            figs.append(_fig_mode_per_class(res, d))
    return figs


def _fig_mode_per_class(res, d):
    pc = d["meta"]["per_class_f1"]
    classes = d["meta"]["classes"]
    rows = list(pc)
    M = np.array([[pc[r].get(c, np.nan) for c in classes] for r in rows]
                 + [[d["meta"]["hand_crafted_per_class"].get(c, np.nan) for c in classes]])
    labels = [f"{res['encoders'][r.split('/')[0]]} / {r.split('/')[1]}" for r in rows] \
        + ["[baseline] hand-crafted kinematics"]
    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 0.3 * len(labels) + 1.4))
    im = ax.imshow(M, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] < 0.2 or M[i, j] > 0.85 else "black")
    ax.axhline(len(rows) - 0.5, color="black", lw=1.0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1")
    ax.set_title("Mode detection, per-class F1")
    fig.tight_layout()
    return ("fig_downstream_mode_per_class", fig)


def _fig_downstream_anomaly(res, d):
    types = [t for t in d["meta"]["counts"] if t != "off_grid_control"]
    figs = []
    for score in ("latent_knn", "reconstruction", "pooled_tokens_no_model"):
        kinds = res["backbones"]
        fig, axes = plt.subplots(1, len(kinds), figsize=(WIDTH_2COL, 3.0),
                                 squeeze=False, sharey=True)
        for ax, kind in zip(axes[0], kinds):
            names, M = [], []
            for key, bb in d["results"].items():
                if kind in bb:
                    names.append(_ds_name(res, key))
                    M.append([bb[kind][score].get(t, np.nan) for t in types])
            for tname, tv in d["baselines"]["trivial"].items():
                names.append(tname)
                M.append([tv.get(t, np.nan) for t in types])
            M = np.array(M, dtype=float)
            im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.3, vmax=1.0, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                fontsize=5, color="white" if M[i, j] > 0.85 or M[i, j] < 0.42 else "black")
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=30,
                               ha="right", fontsize=6)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([shorten(n, 30) for n in names], fontsize=5.5)
            n_rep = sum(1 for n in names if not n.startswith("[trivial]"))
            ax.axhline(n_rep - 0.5, color="black", lw=1.0)
            ax.grid(False)
            ax.set_title(f"{kind}", fontsize=7)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="AUROC (0.5 = chance)")
        fig.suptitle(f"Anomaly detection, score = {score.replace('_', ' ')}", fontsize=8)
        figs.append((f"fig_downstream_anomaly_{score}", fig))
    return figs


def _ds_name(res, key):
    from ..experiments.exp_downstream import ENCODER_NAMES
    return res["encoders"].get(key) or ENCODER_NAMES.get(key, key)


# ==========================================================================
# gpe_suite
# ==========================================================================
def fig_gpe_suite(res):
    figs = []
    P = res["parts"]
    lg = P.get("local_global")
    if lg:
        figs += _fig_local(res, lg)
        figs += _fig_global(res, lg)
        figs += _fig_time(res, lg)
    if P.get("dims"):
        figs.append(_fig_dims(P["dims"]))
    if P.get("road"):
        figs.append(_fig_road(P["road"]))
    return figs


def _fig_local(res, lg):
    metrics = ["MR", "MRR", "MP", "KP@10"]
    kinds = list(lg)
    fig, axes = plt.subplots(1, len(metrics), figsize=(WIDTH_2COL, 2.3))
    for ax, m in zip(axes, metrics):
        encs = list(lg[kinds[0]])
        x = np.arange(len(kinds))
        w = 0.8 / len(encs)
        for i, e in enumerate(encs):
            vals = [lg[k][e]["local_avg"][m] for k in kinds]
            bars = ax.bar(x + (i - (len(encs) - 1) / 2) * w, vals, w,
                          color=_enc_color(lg[kinds[0]][e]["name"], i),
                          edgecolor="black", lw=0.3, hatch=HATCHES[i % len(HATCHES)],
                          label=lg[kinds[0]][e]["name"])
            annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(kinds, fontsize=6)
        ax.set_title(m + (r" $\downarrow$" if m == "MR" else r" $\uparrow$"), fontsize=7)
    axes[0].legend(fontsize=6)
    sk = res.get("skipped_datasets") or {}
    fig.suptitle(f"Local similarity, average over {', '.join(res['datasets'])} "
                 f"(GPE Tables 4 & 8)"
                 + (f"; missing: {', '.join(sk)}" if sk else ""), fontsize=8)
    fig.tight_layout()
    return [("fig_gpe_local", fig)]


def _fig_global(res, lg):
    """Train-on-row / test-on-column MRR matrix, one panel per encoder."""
    kind = "lstm" if "lstm" in lg else next(iter(lg))
    encs = lg[kind]
    ds = res["datasets"]
    if len(ds) < 2:
        return []
    fig, axes = plt.subplots(1, len(encs), figsize=(WIDTH_2COL, 2.2 + 0.25 * len(ds)),
                             squeeze=False)
    for ax, (e, r) in zip(axes[0], encs.items()):
        M = np.full((len(ds), len(ds)), np.nan)
        for i, src in enumerate(ds):
            M[i, i] = r["per_dataset"][src]["local"]["MRR"]
            for j, dst in enumerate(ds):
                if dst in r["per_dataset"][src]["global"]:
                    M[i, j] = list(r["per_dataset"][src]["global"][dst].values())[-1]
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        for i in range(len(ds)):
            for j in range(len(ds)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if M[i, j] < 0.6 else "black",
                            fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels(ds, rotation=30, ha="right", fontsize=6)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels(ds, fontsize=6)
        ax.set_xlabel("test")
        ax.set_ylabel("train")
        ax.grid(False)
        ax.set_title(r["name"], fontsize=7)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="MRR")
    fig.suptitle(f"Zero-shot cross-dataset MRR, {kind} (diagonal = local)", fontsize=8)
    return [("fig_gpe_global", fig)]


def _fig_time(res, lg):
    kind = "lstm" if "lstm" in lg else next(iter(lg))
    encs = lg[kind]
    names = [r["name"] for r in encs.values()]
    pos = [r["time_avg_s"]["position"] for r in encs.values()]
    mod = [r["time_avg_s"]["model"] for r in encs.values()]
    fig, ax = plt.subplots(figsize=(WIDTH_2COL * 0.5, 2.2))
    x = np.arange(len(names))
    ax.bar(x, pos, color=GREY, edgecolor="black", lw=0.3, label="position (fit)")
    ax.bar(x, mod, bottom=pos, color=PALETTE[0], edgecolor="black", lw=0.3,
           label="model training")
    for i in range(len(names)):
        ax.annotate(f"{pos[i]:.1f}+{mod[i]:.0f}", (i, pos[i] + mod[i]), ha="center",
                    va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6)
    ax.set_ylabel(r"seconds $\downarrow$")
    ax.legend(fontsize=6)
    ax.set_title("Training time (GPE Table 6, right)", fontsize=7)
    fig.tight_layout()
    return [("fig_gpe_time", fig)]


def _fig_dims(dd):
    fig, ax = plt.subplots(figsize=(WIDTH_2COL * 0.5, 2.2))
    for i, (name, row) in enumerate(dd["MRR"].items()):
        hs = sorted(int(h) for h in row)
        ax.plot(hs, [row[str(h)] for h in hs], marker=MARKERS[i], color=_enc_color(name, i),
                label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dd["dims"])
    ax.set_xticklabels([str(h) for h in dd["dims"]])
    ax.minorticks_off()
    ax.set_xlabel("embedding size h")
    ax.set_ylabel(r"local MRR $\uparrow$")
    # zoom to the data: with every value near 1 a 0..1 axis hides all
    # differences. The axis then does not start at 0 -- say so in the caption.
    vals = [v for row in dd["MRR"].values() for v in row.values()]
    lo, hi = min(vals), max(vals)
    pad = max(0.15 * (hi - lo), 0.005)
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad) + pad / 2)
    rows = list(dd["MRR"].items())
    for i, (name, row) in enumerate(rows):
        for h in sorted(int(h) for h in row):
            v = row[str(h)]
            # label above if this series is the highest at h, else below
            top = v >= max(r[str(h)] for _, r in rows if str(h) in r)
            ax.annotate(f"{v:.3f}", (h, v), textcoords="offset points",
                        xytext=(0, 6 if top else -11), ha="center",
                        fontsize=5.5, color=_enc_color(name, i))
    ax.legend(fontsize=6)
    ax.set_title(f"Embedding size, {dd['dataset']} (GPE Table 6, left)", fontsize=7)
    fig.tight_layout()
    return ("fig_gpe_dims", fig)


def _fig_road(rd):
    rows = rd["rows"]
    names = list(rows)
    measures = [m for m in rows[names[0]]]
    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 2.5))
    x = np.arange(len(measures))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [rows[n].get(m, np.nan) for m in measures]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w,
                      color=_enc_color(n, i) if not n.startswith("node2vec") else PALETTE[2],
                      edgecolor="black", lw=0.3, hatch=HATCHES[i % len(HATCHES)], label=n)
        annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in measures])
    ax.set_ylabel(r"hit ratio $\uparrow$")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=6, ncol=3)
    ax.set_title(f"Road network, supervised in ST2Vec, lambda={rd['lambda']} "
                 f"({rd['n_nodes']} vertices, {rd['graph']} graph; GPE Table 9)", fontsize=7)
    fig.tight_layout()
    return ("fig_gpe_road", fig)


# ==========================================================================
# LaTeX tables
# ==========================================================================
def _tx(x) -> str:
    """Escape LaTeX specials in free text (dataset names, labels)."""
    x = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#")):
        x = x.replace(a, b)
    return x


def _skipped_note(res):
    """Caption sentence naming the requested-but-missing datasets, so a table
    built from a partial run can never be read as the full comparison."""
    sk = res.get("skipped_datasets") or {}
    if not sk:
        return ""
    return (" Not included (data unavailable for this run): "
            + ", ".join(_tx(d) for d in sk) + ".")


def _task_label(name, task):
    """For ETA the '+time+speed' encoders carry only the departure time and no
    speed channel (both would leak the answer); label them accordingly."""
    return name.replace("+time+speed", "+departure time") if task == "eta" else name
def _fmt(v, best, lower=False, nd=3):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.{nd}f}"
    return rf"\textbf{{{s}}}" if best is not None and np.isclose(v, best) else s


def tex_gpe_suite(res):
    out = []
    lg = res["parts"].get("local_global")
    if lg:
        ds = res["datasets"]
        for kind, encs in lg.items():
            metrics = [("MR", True), ("MRR", False), ("MP", False), ("KP@10", False)]
            best = {m: (min if low else max)(r["local_avg"][m] for r in encs.values())
                    for m, low in metrics}
            cols = "l" + "r" * 4 + ("|" + "r" * len(ds) if len(ds) > 1 else "")
            lines = [r"\begin{table}[t]", r"\centering",
                     rf"\caption{{Local and global results, {kind} backbone, "
                     rf"averaged over {len(ds)} dataset(s): {', '.join(_tx(d) for d in ds)} "
                     r"(layout of GPE Table 4). "
                     + (r"Global columns are zero-shot MRR when training on the row "
                        r"encoder's datasets and testing on the named one, averaged over sources."
                        if len(ds) > 1 else
                        r"The zero-shot global block needs two or more datasets and is omitted.")
                     + _skipped_note(res) + "}",
                     rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
                     " & MR$\\downarrow$ & MRR$\\uparrow$ & MP$\\uparrow$ & KP$\\uparrow$"
                     + ("".join(f" & {_tx(d)}" for d in ds) if len(ds) > 1 else "") + r" \\",
                     r"\midrule"]
            for e, r in encs.items():
                a = r["local_avg"]
                row = _tx(r["name"]) + "".join(
                    " & " + _fmt(a[m], best[m], nd=3) for m, _ in metrics)
                if len(ds) > 1:
                    for dst in ds:
                        v = [list(r["per_dataset"][src]["global"][dst].values())[-1]
                             for src in ds if dst in r["per_dataset"][src]["global"]]
                        row += " & " + (f"{np.mean(v):.3f}" if v else "--")
                lines.append(row + r" \\")
            if "gpe" in encs and "geo" in encs:
                from ..experiments.exp_gpe_suite import improvement
                g, o = encs["gpe"]["local_avg"], encs["geo"]["local_avg"]
                lines.append(r"\midrule")
                lines.append("Improv. (GEO vs GPE)" + "".join(
                    f" & {improvement(o[m], g[m], m):+.1f}\\%" for m, _ in metrics)
                    + (" &" * len(ds) if len(ds) > 1 else "") + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
            out.append("\n".join(lines))
    rd = res["parts"].get("road")
    if rd:
        rows = rd["rows"]
        ms = list(next(iter(rows.values())))
        best = {m: max(r.get(m, -1) for r in rows.values()) for m in ms}
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{Road-network HR@{rd['k']}, supervised inside an ST2Vec-style "
                 rf"model with $\lambda={rd['lambda']}$ (layout of GPE Table 9).}}",
                 r"\begin{tabular}{l" + "r" * len(ms) + "}", r"\toprule",
                 " & " + " & ".join(m.split(" ", 1)[1] for m in ms) + r" \\", r"\midrule"]
        for n, r in rows.items():
            lines.append(_tx(n) + "".join(" & " + _fmt(r.get(m), best[m]) for m in ms) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)


def tex_downstream(res):
    out = [f"% task '{t}' was not run: {why}"
           for t, why in (res.get("skipped_tasks") or {}).items()]
    for task, d in res["tasks"].items():
        if task == "anomaly":
            continue
        head, sec, higher = TASK_METRICS[task]
        metrics = {"tul": ["acc@1", "acc@5", "macro_P", "macro_R", "macro_F1"],
                   "eta": ["MAE", "RMSE", "MAPE"],
                   "mode": ["acc@1", "macro_P", "macro_R", "macro_F1"]}[task]
        lower = {"MAE", "RMSE", "MAPE"}
        cells = []
        for key, bb in d["results"].items():
            for kind, v in bb.items():
                cells.append((res["encoders"][key], kind, v))
        best = {m: (min if m in lower else max)(c[2].get(m, np.nan) for c in cells)
                for m in metrics}
        title = {"tul": "Trajectory-user linking",
                 "eta": "Travel-time estimation (minutes)",
                 "mode": "Transportation-mode detection"}[task]
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{{title} on {_tx(res['dataset'])}.}}",
                 r"\begin{tabular}{ll" + "r" * len(metrics) + "}", r"\toprule",
                 "Encoder & Backbone & " + " & ".join(_tx(m.replace("_", " ")) for m in metrics) + r" \\",
                 r"\midrule"]
        for name, kind, v in cells:
            lines.append(f"{_tx(_task_label(name, task))} & {kind}" + "".join(
                " & " + _fmt(v.get(m), best[m]) + (f"$\\pm${v[m + '_std']:.3f}" if m + "_std" in v else "")
                for m in metrics) + r" \\")
        lines.append(r"\midrule")
        for bname, b in d["baselines"].items():
            if isinstance(b, dict):
                lines.append(f"\\textit{{{_tx(bname)}}} & --" + "".join(
                    " & " + (f"{b[m]:.3f}" if m in b else "--") for m in metrics) + r" \\")
            else:
                lines.append(f"\\textit{{{_tx(bname)}}} & --" + "".join(
                    " & " + (f"{b:.3f}" if m == head else "--") for m in metrics) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)
