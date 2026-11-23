"""Channel-ablation figures: which GEO channels matter for which task.

  fig_ablation_indomain   from downstream results (--ablations), all datasets
  fig_ablation_robust     from robust results (--ablations)
  fig_ablation_transfer   from transfer_suite results (--ablations)

Every panel is a heatmap: rows = ablated variants ("- location" = GEO with
the location channel removed, "location only" = GEO with only that channel),
columns = datasets (or source -> target pairs). A cell is the change of the
task metric relative to full GEO, in %, signed so that NEGATIVE = WORSE:
a red row is a channel the task relies on. Second line: +- one standard
deviation over seeds of that change (paired by seed where the per-seed
values were kept, otherwise from the two standard deviations).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..ablation import CHANNELS, LABEL
from .builders_compare import TASK_NAME, _tx
from .style import WIDTH_2COL

HEAD = {"similarity": ("KP@10", True), "tul": ("acc@1", True), "eta": ("MAE", False),
        "mode": ("macro_F1", True), "anomaly": ("AUROC (latent kNN)", True)}
TASKS = ("similarity", "tul", "eta", "mode", "anomaly")
TNAME = {**TASK_NAME, "similarity": "trajectory similarity",
         "tul": "trajectory-user linking", "anomaly": "anomaly detection"}


# --------------------------------------------------------------------------
# variant names
# --------------------------------------------------------------------------
def _order_key(kind, ch):
    return (0 if kind == "no" else 1, CHANNELS.index(ch))


def _from_vkey(vk):
    kind, ch = vk.split("_", 1)
    return kind, ch


def _from_row(row):
    """'GEO (full) - location' / 'GEO (full): location only' -> (kind, ch)"""
    inv = {v: k for k, v in LABEL.items()}
    if "− " in row:
        return "no", inv[row.split("− ", 1)[1].strip()]
    lab = row.split(": ", 1)[1].rsplit(" only", 1)[0]
    return "only", inv[lab]


def _label(kind, ch):
    return f"− {LABEL[ch]}" if kind == "no" else f"{LABEL[ch]} only"


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------
def _change(v, f, hi):
    if v is None or f is None or not np.isfinite(f) or f == 0:
        return np.nan
    return (v - f) / abs(f) * 100.0 * (1 if hi else -1)


def _cell(var, full, get, hi):
    """(mean % change, std) for a variant cell vs the full-GEO cell."""
    try:
        v, f = get(var), get(full)
    except (KeyError, TypeError):
        return np.nan, np.nan
    sv, sf = var.get("_seeds"), full.get("_seeds")
    if sv and sf and len(sv) == len(sf) and len(sv) > 1:
        try:
            ch = [_change(get(a), get(b), hi) for a, b in zip(sv, sf)]
            return float(np.nanmean(ch)), float(np.nanstd(ch))
        except (KeyError, TypeError):
            pass
    err = np.nan
    m = getattr(get, "metric", None)
    if m and (m + "_std") in var and (m + "_std") in full and f:
        err = 100.0 * np.hypot(var[m + "_std"], full[m + "_std"]) / abs(f)
    return _change(v, f, hi), err


def _getter(task):
    if task == "anomaly":
        def g(c):
            return c["latent_knn"]["all"]
        g.metric = None
        return g, True
    m, hi = HEAD[task]

    def g(c, m=m):
        return c[m]
    g.metric = m
    return g, hi


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def _heatmaps(panels, title, ncols=None):
    """panels: [{"title", "rows" [(kind, ch)], "cols" [str], "val", "err"}]"""
    panels = [p for p in panels if p["rows"] and p["cols"]]
    if not panels:
        return None
    ncols = ncols or min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    hmax = max(len(p["rows"]) for p in panels)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(WIDTH_2COL, nrows * (0.33 * hmax + 1.15) + 0.5))
    for k, p in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        V, E = np.asarray(p["val"], float), np.asarray(p["err"], float)
        lim = np.nanmax(np.abs(V)) if np.isfinite(V).any() else 1.0
        lim = max(lim, 1.0)
        cm = plt.get_cmap("RdBu").copy()
        cm.set_bad("0.9")
        ax.imshow(np.ma.masked_invalid(V), cmap=cm, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                v, e = V[i, j], E[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.5,
                            color="0.5")
                    continue
                dark = abs(v) > 0.6 * lim
                txt = f"{v:+.1f}" + (f"\n±{e:.1f}" if np.isfinite(e) else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                        linespacing=0.95, color="white" if dark else "black")
        ax.grid(False)
        ax.set_xticks(range(len(p["cols"])))
        ax.set_xticklabels(p["cols"], fontsize=5.8, rotation=25 if len(p["cols"]) > 3 else 0,
                           ha="right" if len(p["cols"]) > 3 else "center")
        ax.set_yticks(range(len(p["rows"])))
        ax.set_yticklabels([_label(*r) for r in p["rows"]], fontsize=6)
        ax.set_title(p["title"], fontsize=7)
    for k in range(len(panels), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(title + "\n% change from GEO with every channel (negative = worse)",
                 fontsize=8)
    fig.tight_layout()
    return fig


def _rows_sorted(rows):
    return sorted(set(rows), key=lambda r: _order_key(*r))


# --------------------------------------------------------------------------
# 1. in domain (downstream results)
# --------------------------------------------------------------------------
def indomain_panels(down):
    panels = []
    for task in TASKS:
        files = [(n, d) for n, d in down.items()
                 if task in d.get("tasks", {}) and d["tasks"][task].get("ablation")]
        if not files:
            continue
        get, hi = _getter(task)
        cols, cells, rows = [], {}, []
        for name, d in files:
            ab = d["tasks"][task]["ablation"]
            base = ab[d.get("ablation_base", "geo_ts")]
            kinds = d.get("backbones") or list(base)
            for kind in kinds:
                if kind not in base:
                    continue
                col = name if len(kinds) == 1 else f"{name}\n{kind}"
                cols.append(col)
                for key, bb in ab.items():
                    if "-" not in key or kind not in bb:
                        continue
                    r = _from_vkey(key.split("-", 1)[1])
                    rows.append(r)
                    cells[r, col] = _cell(bb[kind], base[kind], get, hi)
        rows = _rows_sorted(rows)
        V = [[cells.get((r, c), (np.nan, np.nan))[0] for c in cols] for r in rows]
        E = [[cells.get((r, c), (np.nan, np.nan))[1] for c in cols] for r in rows]
        m = HEAD[task][0]
        panels.append({"title": f"{TNAME.get(task, task)} ({m})", "task": task,
                       "rows": rows, "cols": cols, "val": V, "err": E})
    return panels


def fig_ablation_indomain(down):
    f = _heatmaps(indomain_panels(down), "Channel ablation, in domain")
    return [("fig_ablation_indomain", f)] if f else []


# --------------------------------------------------------------------------
# 2. robustness
# --------------------------------------------------------------------------
def _robust_worst(d, row, m, hi):
    vals = [c[m] for sw in d["sweeps"].values() for c in sw["rows"].get(row, [])]
    return (min(vals) if hi else max(vals)) if vals else None


def robust_panels(res):
    GEO = "GEO (full)"
    panels = []
    by_task = {}
    for name, bk in res["datasets"].items():
        for kind, bt in bk.items():
            for tname, d in bt.items():
                by_task.setdefault(tname, []).append((name, kind, d))
    multi = len(res.get("kinds", [])) > 1
    for tname in [t for t in TASKS if t in by_task] + [t for t in by_task if t not in TASKS]:
        items = by_task[tname]
        for view in ("in place", "worst over every move"):
            cols, cells, rows = [], {}, []
            for name, kind, d in items:
                abl = d.get("ablation_rows") or []
                if not abl:
                    continue
                m, hi = d["metric"], d["higher_is_better"]
                col = name if not multi else f"{name}\n{kind}"
                cols.append(col)
                for row in abl:
                    r = _from_row(row)
                    rows.append(r)
                    if view == "in place":
                        cells[r, col] = _cell(d["in_place"][row], d["in_place"][GEO],
                                              lambda c, m=m: c[m], hi)
                        # std from the two cells (no per-seed values here)
                        sv, sf = (d["in_place"][row].get(m + "_std"),
                                  d["in_place"][GEO].get(m + "_std"))
                        f = d["in_place"][GEO][m]
                        if sv is not None and sf is not None and f:
                            cells[r, col] = (cells[r, col][0],
                                             100.0 * np.hypot(sv, sf) / abs(f))
                    else:
                        cells[r, col] = (_change(_robust_worst(d, row, m, hi),
                                                 _robust_worst(d, GEO, m, hi), hi), np.nan)
            if not cols:
                continue
            rows = _rows_sorted(rows)
            V = [[cells.get((r, c), (np.nan, np.nan))[0] for c in cols] for r in rows]
            E = [[cells.get((r, c), (np.nan, np.nan))[1] for c in cols] for r in rows]
            m = items[0][2]["metric"]
            panels.append({"title": f"{TNAME.get(tname, tname)} ({m}), {view}",
                           "task": tname, "rows": rows, "cols": cols, "val": V, "err": E})
    return panels


def fig_ablation_robust(res):
    scope = res.get("scope", "city")
    f = _heatmaps(robust_panels(res), "Channel ablation under translations and rotations "
                  f"({'whole city moved' if scope == 'city' else 'each trip moved'})",
                  ncols=2)
    return [("fig_ablation_robust", f)] if f else []


# --------------------------------------------------------------------------
# 3. transfer
# --------------------------------------------------------------------------
def transfer_panels(res):
    GEO = "GEO (full)"
    panels = []
    for tname in [t for t in TASKS if t in res.get("results", {})]:
        for kind, r in res["results"][tname].items():
            abl = r.get("ablation_rows") or []
            if not abl:
                continue
            m, hi = r["metric"], r["higher_is_better"]
            get = lambda c, m=m: c[m]           # noqa: E731
            sfx = f" [{kind}]" if len(res.get("kinds", [])) > 1 else ""
            rows = _rows_sorted(_from_row(x) for x in abl)
            # in domain: one column per dataset
            cols = [n for n in r["datasets"] if all(x in r["in_domain"][n] for x in abl)]
            V, E = [], []
            for rr in rows:
                row = next(x for x in abl if _from_row(x) == rr)
                vals = [_cell(r["in_domain"][n][row], r["in_domain"][n][GEO], get, hi)
                        for n in cols]
                vals = [(c, _err(r["in_domain"][n][row], r["in_domain"][n][GEO], m))
                        for (c, _), n in zip(vals, cols)]
                V.append([v for v, _ in vals])
                E.append([e for _, e in vals])
            panels.append({"title": f"{TNAME.get(tname, tname)} ({m}), in domain{sfx}",
                           "task": tname, "rows": rows, "cols": cols, "val": V, "err": E})
            # zero-shot: one column per source -> target pair
            keys = [k for k, pr in r["pairs"].items()
                    if all(x in pr["zero_shot"] for x in abl)]
            V, E = [], []
            for rr in rows:
                row = next(x for x in abl if _from_row(x) == rr)
                V.append([_change(r["pairs"][k]["zero_shot"][row][m],
                                  r["pairs"][k]["zero_shot"][GEO][m], hi) for k in keys])
                E.append([_err(r["pairs"][k]["zero_shot"][row],
                               r["pairs"][k]["zero_shot"][GEO], m) for k in keys])
            panels.append({"title": f"{TNAME.get(tname, tname)} ({m}), zero-shot{sfx}",
                           "task": tname, "rows": rows,
                           "cols": [k.replace("→", "→\n") for k in keys],
                           "val": V, "err": E})
    return panels


def _err(var, full, m):
    sv, sf, f = var.get(m + "_std"), full.get(m + "_std"), full.get(m)
    if sv is None or sf is None or not f:
        return np.nan
    return 100.0 * np.hypot(sv, sf) / abs(f)


def fig_ablation_transfer(res):
    f = _heatmaps(transfer_panels(res), "Channel ablation across datasets", ncols=2)
    return [("fig_ablation_transfer", f)] if f else []


# --------------------------------------------------------------------------
# table (any of the panel lists above)
# --------------------------------------------------------------------------
def _t(s):
    return _tx(s).replace("\u2212", "$-$").replace("\u2192", r"$\to$")


def tex_ablation(panels, caption):
    if not panels:
        return ""
    out = []
    for p in panels:
        cols = [c.replace("\n", " ") for c in p["cols"]]
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{{caption} {_t(p['title'])}: change from GEO with every "
                 r"channel, in \% (negative = worse), $\pm$ one standard deviation "
                 r"over seeds.}", r"\small",
                 r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
                 "Variant & " + " & ".join(_t(c) for c in cols) + r" \\", r"\midrule"]
        for i, r in enumerate(p["rows"]):
            vals = []
            for j in range(len(cols)):
                v, e = p["val"][i][j], p["err"][i][j]
                vals.append("--" if not np.isfinite(v) else
                            f"{v:+.1f}" + (f" $\\pm$ {e:.1f}" if np.isfinite(e) else ""))
            lines.append(_t(_label(*r)) + " & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)
