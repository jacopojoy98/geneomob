"""Figures and tables for `transfer_suite` (train on one dataset, test on another).

  fig_transfer_suite_<task>[_<model>]  layout of fig_robust_<task>: one row
                                 per target, one panel per source; left of
                                 the rule the model trained on the target,
                                 right of it zero-shot and fine-tuned.
                                 GPE as published vs GEO (frame re-fitted)
  fig_transfer_suite_all_<task>  (only with --scratch)  rows = target, columns = source. In each
                                 panel: zero-shot at the left, then fine-tuning
                                 on growing fractions of the target's training
                                 data (solid, filled) against training from
                                 scratch on the same fraction (dotted, hollow);
                                 thin dashed lines = trained on the whole
                                 target (in domain). Orange GPE, blue GEO.
  fig_transfer_suite_matrix           zero-shot retention (% of the in-domain score
                                 kept) for every source x target, for GPE, for
                                 GEO, and GEO minus GPE
  fig_transfer_suite_summary          per pair and task: in domain on the target
                                 (dot) vs zero-shot from the source (ring)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .builders_compare import (MAIN_STYLE, TASK_NAME, _change_pct, _dumbbells,
                               _pct, _tx)
from .builders_suite import UNITS
from .style import WIDTH_2COL

GPE, GEO, DIAG = "GPE (paper)", "GEO (full)", "GEO (full, source frame)"
STYLE = {GPE: MAIN_STYLE["gpe_refit"], GEO: MAIN_STYLE["geo_refit"],
         DIAG: MAIN_STYLE["geo_fixed"]}
SCRATCH = {GPE: MAIN_STYLE["gpe_fixed"], GEO: MAIN_STYLE["geo_fixed"]}
SHORT = {GPE: "GPE, as published", GEO: "GEO, all channels",
         DIAG: "GEO, source frame"}


def _items(res):
    for tname, by_kind in res["results"].items():
        for kind, r in by_kind.items():
            yield tname, kind, r


def _retention(ind, zs, hi):
    """% of the in-domain score kept zero-shot (100 = nothing lost)."""
    if not ind or not zs:
        return np.nan
    return 100.0 * (zs / ind if hi else ind / zs)


def _ns(res):
    return len(res.get("seeds") or [0])


# --------------------------------------------------------------------------
# 1. curves: zero-shot -> fine-tuned, against from scratch
# --------------------------------------------------------------------------
def _fig_curves(res, tname, kind, r):
    names, fr = r["datasets"], r["fractions"] if "fractions" in r else res["fractions"]
    m, hi = r["metric"], r["higher_is_better"]
    n = len(names)
    fig, axes = plt.subplots(n, n, squeeze=False,
                             figsize=(WIDTH_2COL, 1.55 * n + 1.25), sharey="row")
    x_ft = np.arange(len(fr) + 1, dtype=float)          # 0 = zero-shot
    x_sc = x_ft[1:]
    ticks = ["zero-\nshot"] + [f"{100 * f:g}%" for f in fr]
    for i, dst in enumerate(names):
        ind = r["in_domain"][dst]
        for j, src in enumerate(names):
            ax = axes[i][j]
            ax.grid(axis="y", color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"source: {src}", fontsize=7.2)
            if j == 0:
                arrow = r"$\uparrow$" if hi else r"$\downarrow$"
                ax.set_ylabel(f"target: {dst}\n{m}{UNITS.get(m, '')} {arrow}",
                              fontsize=6.5)
            if src == dst:
                # the diagonal: what training on the target alone gives
                for row in (GPE, GEO):
                    st = SCRATCH[row]
                    cells = r["scratch"][dst][row]
                    y = np.array([c[m] for c in cells])
                    ax.plot(x_sc, y, ls=st["ls"], marker=st["marker"], color=st["color"],
                            mfc="white", mec=st["color"], ms=3.6, lw=1.2)
                    if m + "_std" in cells[0]:
                        e = np.array([c[m + "_std"] for c in cells])
                        ax.fill_between(x_sc, y - e, y + e, color=st["color"],
                                        alpha=0.12, lw=0)
                ax.text(0.03, 0.95, "no transfer:\ntarget only", transform=ax.transAxes,
                        fontsize=5.8, va="top", color="0.35")
            else:
                pr = r["pairs"].get(f"{src}→{dst}")
                if pr is None:
                    ax.text(0.5, 0.5, "skipped", transform=ax.transAxes,
                            ha="center", fontsize=6, color="0.5")
                else:
                    for row in (GPE, GEO):
                        st = STYLE[row]
                        cells = [pr["zero_shot"][row]] + pr["finetune"][row]
                        y = np.array([c[m] for c in cells])
                        ax.plot(x_ft, y, ls="-", marker=st["marker"], color=st["color"],
                                ms=3.6, lw=1.6, zorder=3)
                        if m + "_std" in cells[0]:
                            e = np.array([c.get(m + "_std", 0.0) for c in cells])
                            ax.fill_between(x_ft, y - e, y + e, color=st["color"],
                                            alpha=0.13, lw=0, zorder=1)
                        sc = SCRATCH[row]
                        ys = [c[m] for c in r["scratch"][dst][row]]
                        ax.plot(x_sc, ys, ls=":", marker=sc["marker"], color=sc["color"],
                                mfc="white", mec=sc["color"], ms=3.4, lw=1.1, zorder=2)
                    if DIAG in pr["zero_shot"]:
                        st = STYLE[DIAG]
                        ax.plot([0.0], [pr["zero_shot"][DIAG][m]], marker=st["marker"],
                                mfc="white", mec=st["color"], ms=4.2, ls="none", zorder=4)
            for row in (GPE, GEO):           # trained on the whole target
                ax.axhline(ind[row][m], color=STYLE[row]["color"], lw=0.7, ls="--",
                           alpha=0.7, zorder=0)
            ax.axvline(0.5, color="0.6", lw=0.6)
            ax.set_xlim(-0.5, len(fr) + 0.5)
            ax.set_xticks(x_ft)
            ax.set_xticklabels(ticks, fontsize=5.8)
            ax.tick_params(axis="y", labelsize=6)
            if i == n - 1:
                ax.set_xlabel("target training data used", fontsize=6)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=STYLE[GPE]["color"], marker="o", lw=1.6,
                label="GPE as published: source model, fine-tuned"),
         Line2D([], [], color=SCRATCH[GPE]["color"], marker="o", mfc="white", ls=":",
                lw=1.1, label="GPE: trained from scratch on the target"),
         Line2D([], [], color=STYLE[GEO]["color"], marker="s", lw=1.6,
                label="GEO, all channels: source model, fine-tuned"),
         Line2D([], [], color=SCRATCH[GEO]["color"], marker="s", mfc="white", ls=":",
                lw=1.1, label="GEO: trained from scratch on the target"),
         Line2D([], [], color="0.4", ls="--", lw=0.7,
                label="trained on the whole target (in domain)")]
    if any(DIAG in pr["zero_shot"] for pr in r["pairs"].values()):
        h.append(Line2D([], [], color=STYLE[DIAG]["color"], marker="s", mfc="white",
                        ls="none", label="GEO zero-shot, frame left at the source"))
    ns = _ns(res)
    t = TASK_NAME.get(tname, tname).replace(" (control)", "")
    fig.suptitle(f"Transfer across datasets, {t} ({kind})"
                 + (f"  ({ns} seeds, band $\\pm$1 std)" if ns > 1 else ""), fontsize=8.5)
    H = fig.get_figheight()
    fig.texts[-1].set_y(1 - 0.08 / H)
    fig.texts[-1].set_va("top")
    fig.tight_layout()
    rows_leg = int(np.ceil(len(h) / 2))
    fig.subplots_adjust(top=1 - (0.5 + 0.15 * rows_leg) / H)
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=6.1, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.28 / H), handlelength=3.0, columnspacing=2.0)
    return fig


# --------------------------------------------------------------------------
# 1b. main figure, in the layout of fig_robust_<task>
# --------------------------------------------------------------------------
MAIN = {GPE: dict(MAIN_STYLE["gpe_fixed"]), GEO: dict(MAIN_STYLE["geo_refit"])}
MAIN_LABEL = {GPE: "GPE as published (raw lon/lat, spatial only)",
              GEO: "GEO, all channels, frame re-fitted on the target city"}


def _fig_main(res, tname, kind, r):
    """One row per target, one panel per source. Left of the rule: the model
    trained on the target itself (in domain); right of it: the source model
    on the target, zero-shot and then fine-tuned on growing fractions of the
    target's training data. Bands: +-1 std over seeds."""
    names, m, hi = r["datasets"], r["metric"], r["higher_is_better"]
    fr = r.get("fractions", res.get("fractions", []))
    srcs = {t: [s for s in names if s != t and f"{s}\u2192{t}" in r["pairs"]]
            for t in names}
    targets = [t for t in names if srcs[t]]
    ncol = max(len(v) for v in srcs.values())
    fig, axes = plt.subplots(len(targets), ncol, squeeze=False,
                             figsize=(WIDTH_2COL, 1.75 * len(targets) + 1.25),
                             sharey="row")
    x = np.arange(len(fr) + 1, dtype=float) + 1.0       # zero-shot, fractions
    ticks = ["in\ndomain", "zero-\nshot"] + [f"{100 * f:g}%" for f in fr]
    for i, dst in enumerate(targets):
        ind = r["in_domain"][dst]
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(srcs[dst]):
                ax.axis("off")
                continue
            src = srcs[dst][j]
            pr = r["pairs"][f"{src}\u2192{dst}"]
            for k, row in enumerate((GPE, GEO)):
                st = dict(MAIN[row])
                mfc = st.pop("mfc", st["color"])
                cells = [pr["zero_shot"][row]] + list(pr["finetune"][row])
                y = np.array([c[m] for c in cells])
                ax.plot(x, y, ms=3.6, label=MAIN_LABEL[row], mfc=mfc,
                        mec=st["color"], mew=0.9, zorder=3, **st)
                if m + "_std" in cells[0]:
                    e = np.array([c.get(m + "_std", 0.0) for c in cells])
                    ax.fill_between(x, y - e, y + e, color=st["color"], alpha=0.13,
                                    lw=0, zorder=1)
                c0 = ind[row]                         # trained on the target
                ax.errorbar([(k - 0.5) * 0.12], [c0[m]], yerr=[c0.get(m + "_std", 0.0)],
                            fmt=st["marker"], ms=4.2, color=st["color"], mfc=mfc,
                            mec=st["color"], mew=0.9, capsize=1.5, elinewidth=0.8,
                            zorder=4)
            ax.axvline(0.5, color="0.6", lw=0.6)
            ax.set_xlim(-0.6, len(fr) + 1.5)
            ax.set_xticks(np.r_[0.0, x])
            ax.set_xticklabels(ticks, fontsize=6)
            ax.grid(axis="y", color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{src} \u2192 {dst}", fontsize=7.5)
            if i == len(targets) - 1:
                ax.set_xlabel("target training data used for fine-tuning", fontsize=6)
            if j == 0:
                arrow = r"$\uparrow$" if hi else r"$\downarrow$"
                ax.set_ylabel(f"target: {dst}\n{m}{UNITS.get(m, '')} {arrow}",
                              fontsize=6.5)
    h_, l_ = axes[0][0].get_legend_handles_labels()
    ns = _ns(res)
    t = TASK_NAME.get(tname, tname).replace(" (control)", "")
    fig.suptitle(f"Transfer across datasets, {t} ({kind}): trained on the source city"
                 + (f"  ({ns} seeds, band $\\pm$1 std)" if ns > 1 else ""),
                 fontsize=8.5)
    H = fig.get_figheight()
    fig.texts[-1].set_y(1 - 0.08 / H)
    fig.texts[-1].set_va("top")
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.80 / H)
    fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=6.4, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.30 / H), handlelength=3.4,
               columnspacing=2.5)
    return fig


# --------------------------------------------------------------------------
# 2. retention matrices
# --------------------------------------------------------------------------
def fig_transfer_matrix(res):
    blocks = list(_items(res))
    if not blocks:
        return None
    fig, axes = plt.subplots(len(blocks), 3, squeeze=False,
                             figsize=(WIDTH_2COL, 2.05 * len(blocks) + 0.6))
    for b, (tname, kind, r) in enumerate(blocks):
        names, m, hi = r["datasets"], r["metric"], r["higher_is_better"]
        n = len(names)
        R = {row: np.full((n, n), np.nan) for row in (GPE, GEO)}
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                for row in (GPE, GEO):
                    if src == dst:
                        R[row][i, j] = 100.0
                        continue
                    pr = r["pairs"].get(f"{src}→{dst}")
                    if pr:
                        R[row][i, j] = _retention(r["in_domain"][dst][row][m],
                                                  pr["zero_shot"][row][m], hi)
        D = R[GEO] - R[GPE]
        lim = np.nanmax(np.abs(D[~np.eye(n, dtype=bool)])) if n > 1 else 1.0
        lim = max(lim if np.isfinite(lim) else 1.0, 1.0)
        panels = [(R[GPE], "GPE as published", "cividis", 0, 100),
                  (R[GEO], "GEO, all channels", "cividis", 0, 100),
                  (D, "GEO − GPE (points)", "RdBu", -lim, lim)]
        for c, (A, ttl, cmap, lo, hi_) in enumerate(panels):
            ax = axes[b][c]
            A = A.copy()
            np.fill_diagonal(A, np.nan)          # the diagonal is 100 by definition
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("0.92")
            im = ax.imshow(np.clip(A, lo, hi_) if c < 2 else A, cmap=cm,
                           vmin=lo, vmax=hi_)
            for i in range(n):
                for j in range(n):
                    v = A[i, j]
                    if i == j:
                        txt = "in\ndomain"
                    elif np.isnan(v):
                        txt = ""
                    else:
                        txt = f"{v:+.0f}" if c == 2 else f"{v:.0f}"
                    frac = (v - lo) / (hi_ - lo) if np.isfinite(v) else 0.5
                    dark = (frac < 0.45) if c < 2 else (abs(v) > 0.6 * lim if np.isfinite(v) else False)
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=5.2 if i == j else 6.3,
                            color="0.45" if i == j else ("white" if dark else "black"))
            ax.grid(False)
            ax.set_xticks(range(n))
            ax.set_xticklabels(names, fontsize=6, rotation=30, ha="right")
            ax.set_yticks(range(n))
            ax.set_yticklabels(names if c == 0 else [], fontsize=6)
            if c == 0:
                ax.set_ylabel(f"{TASK_NAME.get(tname, tname).replace(' (control)', '')}"
                              f"\n({kind}, {m})\nsource", fontsize=6.5)
            if b == len(blocks) - 1:
                ax.set_xlabel("target", fontsize=6.5)
            if b == 0:
                ax.set_title(ttl, fontsize=7.2)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=5.5)
    ns = _ns(res)
    fig.suptitle("Zero-shot transfer: % of the in-domain score kept"
                 + (f" (mean of {ns} seeds)" if ns > 1 else ""), fontsize=8.5)
    fig.tight_layout()
    return ("fig_transfer_suite_matrix", fig)


# --------------------------------------------------------------------------
# 3. summary dumbbells
# --------------------------------------------------------------------------
def fig_transfer_summary(res):
    panels = {}
    for tname, kind, r in _items(res):
        for key, pr in r["pairs"].items():
            d = {"metric": r["metric"], "higher_is_better": r["higher_is_better"],
                 "ref": r["in_domain"][pr["dst"]], "zs": pr["zero_shot"]}
            panels.setdefault(tname, []).append((key, kind, d))
    if not panels:
        return None
    rows = [GEO] + ([DIAG] if DIAG in res.get("rows", []) else []) + [GPE]
    items = [(row, SHORT[row], STYLE[row] if row != DIAG else MAIN_STYLE["geo_fixed"])
             for row in rows]

    def pair_fn(d, row, m, hi):
        ref = d["ref"].get(row if row != DIAG else GEO, {}).get(m)
        zs = d["zs"].get(row, {}).get(m)
        return ref, zs

    return _dumbbells(res, items, "fig_transfer_suite_summary",
                      "Transfer across datasets: in domain vs zero-shot",
                      worst_label="zero-shot from the source dataset",
                      in_label="trained on the target (in domain)",
                      panels=panels, pair_fn=pair_fn,
                      task_names={"similarity": "trajectory similarity"})


def fig_transfer(res):
    figs = []
    multi = len(res.get("kinds", [])) > 1
    for tname, kind, r in _items(res):
        sfx = f"{tname}" + (f"_{kind}" if multi else "")
        figs.append((f"fig_transfer_suite_{sfx}", _fig_main(res, tname, kind, r)))
        if any(r.get("scratch", {}).values()):     # with the from-scratch baseline
            figs.append((f"fig_transfer_suite_all_{sfx}",
                         _fig_curves(res, tname, kind, r)))
    for f in (fig_transfer_matrix(res), fig_transfer_summary(res)):
        if f:
            figs.append(f)
    return figs


# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------
def tex_transfer(res):
    out = []
    for tname, kind, r in _items(res):
        m, hi = r["metric"], r["higher_is_better"]
        fr = r.get("fractions", res["fractions"])
        has_sc = any(r.get("scratch", {}).values())
        fcols = " & ".join(f"{100 * f:g}\\%" for f in fr)
        blocks = [("Fine-tuned", fcols)] + ([("Scratch", fcols)] if has_sc else [])
        blocks = [b for b in blocks if fr]
        head = "".join(rf" & \multicolumn{{{len(fr)}}}{{c}}{{{b}}}" for b, _ in blocks)
        sub = "".join(f" & {c}" for _, c in blocks)
        lines = [r"\begin{table}[t]", r"\centering",
                 rf"\caption{{Transfer across datasets, {_tx(tname)} ({kind}, "
                 rf"{_tx(m)}, {'higher' if hi else 'lower'} is better). In domain: "
                 r"trained on the target. Zero-shot: the source model on the "
                 r"target, without target labels. Fine-tuned / scratch: trained "
                 r"further from the source model / from scratch on that fraction "
                 r"of the target's training data. $\Delta$: zero-shot against in "
                 r"domain (negative = worse).}",
                 r"\small",
                 rf"\begin{{tabular}}{{ll rr r{'r' * len(fr) * len(blocks)}}}",
                 r"\toprule",
                 rf"Source$\to$target & Method & In domain & Zero-shot & $\Delta$ (\%)"
                 + head + r" \\"]
        if blocks:
            lines.append(rf" & & & & {sub} \\")
        lines += [r"\midrule"]
        for key, pr in r["pairs"].items():
            for i, row in enumerate((GEO, GPE)):
                ind = r["in_domain"][pr["dst"]][row][m]
                zs = pr["zero_shot"][row][m]
                ch = _change_pct(ind, zs, hi)
                cols = [f"{c[m]:.3f}" for c in pr["finetune"][row]]
                if has_sc:
                    cols += [f"{c[m]:.3f}" for c in r["scratch"][pr["dst"]][row]]
                lines.append((f"{_tx(pr['src'])}$\\to${_tx(pr['dst'])}" if i == 0 else "")
                             + f" & {_tx(SHORT[row])} & {ind:.3f} & {zs:.3f} & "
                             + f"{ch:+.1f}" + "".join(f" & {c}" for c in cols) + r" \\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines += [r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    return "\n".join(out)
