"""One builder per results file. Each takes the parsed JSON and returns
`(name, figure)` pairs, so `make_figures.py` can write them all out.

Every builder is defensive about missing keys: a partial results file (an
interrupted run, a subset of tokenizers) produces the figures it can and skips
the rest rather than failing.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from .style import (BLUE, GREY, HATCHES, LINESTYLES, MARKERS, PALETTE,
                    VERMILLION, WIDTH_1COL, WIDTH_2COL, annotate_bars,
                    color_for, shorten)

FLOOR = 1e-16   # so exact zeros are drawable on a log axis


def _vlabel(c: str) -> str:
    """'v=(1000.0, 0.0)' -> 'v = (1000, 0) m'; '|v|=50.0' -> '||v|| = 50 m'."""
    if c.startswith("v="):
        nums = c[2:].strip("()").split(",")
        return "(" + ", ".join(str(int(float(n))) for n in nums) + ") m"
    if c.startswith("|v|="):
        return r"$\|v\|$ = " + f"{float(c.split('=')[1]):g} m"
    return c


def _clip(v):
    return max(float(v), FLOOR) if v is not None else None


# ==========================================================================
# audit.json
# ==========================================================================
def fig_equivariance_error(res):
    """Translation and rotation equivariance error per encoder, log scale.

    The headline figure: it shows that GPE is already an exact translation GEO
    (so this work explains it rather than competing with it), and that the
    displacement channel is the one that adds exact rotation equivariance.
    """
    enc = res.get("encoders", {})
    rot = res.get("rotation", {})
    names = [n for n in enc if n in rot]
    if not names:
        return []
    order = sorted(names, key=lambda n: enc[n]["translation_err"])
    tr = [_clip(enc[n]["translation_err"]) for n in order]
    ro = [_clip(rot[n]) for n in order]

    # the displacement channel is rotation-only, so it gets its own bar
    extra = [k for k in rot if k.startswith("DisplacementGEO")]

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 2.5))
    x = np.arange(len(order) + len(extra))
    w = 0.38
    ax.bar(x[:len(order)] - w / 2, tr, w,
           color=[color_for(n, i) for i, n in enumerate(order)],
           edgecolor="black", linewidth=0.4)
    ax.bar(x[:len(order)] + w / 2, ro, w,
           color=[color_for(n, i) for i, n in enumerate(order)],
           edgecolor="black", linewidth=0.4, hatch="///", alpha=0.85)
    if extra:
        vals = [_clip(rot[k]) for k in extra]
        ax.bar(x[len(order):] + w / 2, vals, w,
               color=[color_for(k, i) for i, k in enumerate(extra)],
               edgecolor="black", linewidth=0.4, hatch="///", alpha=0.85)

    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 10, 10)
    ax.axhline(1e-10, color=GREY, lw=0.7, ls=":")
    ax.annotate("floating-point exact", (len(x) - 0.4, 1.5e-10), ha="right",
                va="bottom", fontsize=6, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([shorten(n, 22).replace("DisplacementGEO", "DispGEO")
                        for n in order + extra], rotation=20, ha="right")
    ax.set_ylabel(r"relative error  $\|f(g\cdot x)-\rho(g)f(x)\|$")
    # Colour already encodes the encoder, so the legend must not reuse it:
    # neutral proxies keep it about the group action only.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.75", edgecolor="black", lw=0.4,
                             label="translation"),
                       Patch(facecolor="0.75", edgecolor="black", lw=0.4,
                             hatch="///", label="rotation")],
              ncol=2, loc="upper left")
    ax.set_title("Encoder equivariance under translation and rotation")
    return [("fig_equivariance_error", fig)]


def fig_latitude(res):
    """Gram distortion under a metric-preserving move to another latitude.

    The instrument for H6. Solid lines are the encoder as trained; dashed
    lines re-fit the local frame, which is available to every metric encoder
    and to no angular one -- GPE's ladder is fixed in angular units worldwide.
    """
    lat = res.get("latitude", {})
    if not lat:
        return []
    targets = sorted(lat, key=float)
    xs = [float(t) for t in targets]
    keys = [k for k in lat[targets[0]]
            if k != "lon_stretch" and lat[targets[0]][k] is not None]
    base = [k for k in keys if "refit" not in k]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.6),
                                  gridspec_kw={"width_ratios": [2.4, 1]})
    n_refit = 0
    for i, k in enumerate(base):
        ys = [lat[t].get(k) for t in targets]
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], color=color_for(k, i),
                ls=LINESTYLES[i % len(LINESTYLES)], label=shorten(k, 22))
        rk = k + " [refit ref]"
        if rk in lat[targets[0]]:
            ax.plot(xs, [lat[t].get(rk) for t in targets], marker=".",
                    color=color_for(k, i), ls=":", lw=1.0, alpha=0.8)
            n_refit += 1
    ax.set_xlabel("latitude of the relocated corpus (deg)")
    ax.set_ylabel("Gram distortion of the code")
    ax.set_title("Same trip, different latitude")
    ax.legend(ncol=2, fontsize=6)
    if n_refit:
        # Every re-fitted metric code lands exactly on zero, so the dotted
        # lines coincide with the axis: say it rather than draw it on top of
        # the data.
        ax.set_ylim(-0.012, None)
        ax.annotate("metric codes fall to exactly 0 once the local frame is "
                    "re-fitted (dotted, on the axis)",
                    (0.5, 0.16), xycoords="axes fraction", ha="center",
                    fontsize=6, color="0.25",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.8", lw=0.4))
    src = res.get("ref_lat")
    if src is not None:
        ax.axvline(src, color=GREY, lw=0.7, ls="--")
        ax.annotate("source", (src, ax.get_ylim()[1]), fontsize=6,
                    color=GREY, ha="center", va="top")

    stretch = [lat[t].get("lon_stretch") for t in targets]
    ax2.plot(xs, stretch, marker="o", color="black")
    ax2.axhline(1.0, color=GREY, lw=0.7, ls=":")
    ax2.set_xlabel("latitude (deg)")
    ax2.set_ylabel(r"east-west stretch $\cos\phi_{src}/\cos\phi$")
    ax2.set_title("Confound being varied")
    fig.text(0.01, -0.06,
             "Dotted lines: the same encoder after re-fitting its local frame. "
             "Every corpus GPE evaluates sits at 40-42$^\\circ$N, so their "
             "cross-city experiments vary this factor by under 3%.",
             fontsize=6, color="0.3")
    fig.tight_layout()
    return [("fig_latitude", fig)]


def fig_noise(res):
    """Code stability under GPS jitter, with the wavelength scale annotated."""
    sat = res.get("noise_saturation", {})
    if not sat:
        return []
    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 2.4))
    for i, (name, curve) in enumerate(sat.items()):
        xs = sorted(float(k) for k in curve)
        ys = [curve[str(x)] if str(x) in curve else curve[f"{x:.1f}"]
              for x in xs]
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], color=color_for(name, i),
                ls=LINESTYLES[i % len(LINESTYLES)], label=shorten(name, 16))
    ax.axvline(15.0, color=GREY, lw=0.7, ls="--")
    ax.annotate("typical GPS error", (15.4, 0.02), fontsize=6, color=GREY,
                rotation=90, va="bottom")
    ax.set_xlabel(r"positional noise $\sigma$ (m)")
    ax.set_ylabel("cosine similarity to the clean code")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6)
    ax.set_title("Stability under GPS jitter")
    return [("fig_noise", fig)]


# ==========================================================================
# expa.json
# ==========================================================================
def fig_experiment_a(res):
    """Sample efficiency alongside equivariance and attribution error.

    Handles both shapes of results file: the old two-model synthetic version
    and the current one, where every tokenizer compared elsewhere is a row
    through the same backbone.
    """
    sizes = res.get("sizes")
    if not sizes:
        return []
    if "models" in res:
        return _fig_experiment_a_multi(res)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.5),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(sizes, res["geo_rmse"], marker="o", color=color_for("GEO(i)+disp"),
            label="GEO tokenizer + equivariant aggregator")
    ax.plot(sizes, res["base_rmse"], marker="s", ls="--", color=GREY,
            label="matched-capacity unconstrained baseline")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("training trajectories")
    ax.set_ylabel("validation RMSE")
    ax.set_title("Sample efficiency")
    ax.legend(fontsize=6)

    ax = axes[1]
    groups = [("equivariance_error", "prediction"),
              ("explanation_equivariance_error", "attribution")]
    labels, geo, base = [], [], []
    for key, lab in groups:
        if key in res:
            labels.append(lab)
            geo.append(_clip(res[key]["geo"]))
            base.append(_clip(res[key]["baseline"]))
    if labels:
        x = np.arange(len(labels))
        w = 0.36
        b1 = ax.bar(x - w / 2, geo, w, color=color_for("GEO(i)+disp"),
                    edgecolor="black", lw=0.4, label="GEO")
        b2 = ax.bar(x + w / 2, base, w, color=GREY, edgecolor="black", lw=0.4,
                    hatch="///", label="baseline")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("relative equivariance error")
        ax.set_title("Exactness under rotation")
        ax.legend(fontsize=6)
        for bars, vals in ((b1, geo), (b2, base)):
            annotate_bars(ax, bars, vals, fmt="{:.1e}", fontsize=5.5)
        ax.set_ylim(min(geo + base) / 30, max(geo + base) * 30)
    fig.tight_layout()
    return [("fig_experiment_a", fig)]


def _fig_experiment_a_multi(res):
    models = res["models"]
    sizes = res["sizes"]
    unit = res.get("unit", "")
    const = res.get("constant_predictor")
    failed = set(res.get("warning_failed_to_fit", []))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.7),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for i, (name, r) in enumerate(models.items()):
        errs = r["errors"]
        ax.plot(sizes[:len(errs)], errs, marker=MARKERS[i % len(MARKERS)],
                color=color_for(name, i), ls=LINESTYLES[i % len(LINESTYLES)],
                alpha=0.35 if name in failed else 1.0,
                label=shorten(name, 22) + (" (did not fit)" if name in failed
                                           else ""))
    if const is not None:
        ax.axhline(const, color=GREY, lw=0.9, ls="--")
        ax.annotate("constant predictor", (sizes[0], const), fontsize=6,
                    color=GREY, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("training trajectories")
    ax.set_ylabel(f"{res.get('task', 'task')} error ({unit})")
    ax.set_title(f"Sample efficiency \u2014 {res.get('dataset', '')}")
    ax.legend(fontsize=5.5, ncol=2)

    ax = axes[1]
    names = list(models)
    x = np.arange(len(names))
    w = 0.38
    eq = [_clip(models[n].get("equivariance_error")) for n in names]
    ex = [_clip(models[n].get("explanation_equivariance_error")) for n in names]
    ax.bar(x - w / 2, eq, w, color=[color_for(n, i) for i, n in enumerate(names)],
           edgecolor="black", lw=0.3)
    ax.bar(x + w / 2, ex, w, color=[color_for(n, i) for i, n in enumerate(names)],
           edgecolor="black", lw=0.3, hatch="///", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([shorten(n, 14) for n in names], rotation=25, ha="right",
                       fontsize=5.5)
    ax.set_ylabel("relative error under rotation")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.75", edgecolor="black", lw=0.3,
                             label="prediction"),
                       Patch(facecolor="0.75", edgecolor="black", lw=0.3,
                             hatch="///", label="attribution")],
              fontsize=6, ncol=2, loc="lower left")
    rho = "R" if res.get("task") == "destination" else "I"
    ax.set_title(rf"Exactness under rotation ($\rho={rho}$)")
    fig.tight_layout()
    return [(f"fig_experiment_a_{res.get('task', 'task')}", fig)]


# ==========================================================================
# tokens.json
# ==========================================================================
def fig_token_identity(res):
    """Token preservation under translation, split by what is being asked.

    Left: v is a lattice period, where an equivariant code must preserve the
    index exactly. Right: v is arbitrary, where nothing should -- the
    prediction the proposal originally got wrong, shown so the corrected claim
    is legible.
    """
    a = res.get("test_1_lattice_period", {})
    b = res.get("test_1b_arbitrary_v", {})
    if not a:
        return []
    methods = list(next(iter(a.values())).keys())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.5), sharey=True)

    for ax, data, title in ((axes[0], a, "$v$ is a lattice period"),
                            (axes[1], b, "$v$ arbitrary")):
        if not data:
            ax.set_visible(False)
            continue
        conds = list(data)
        x = np.arange(len(conds))
        w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            vals = [data[c].get(m, np.nan) for c in conds]
            bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w,
                          color=color_for(m, i), edgecolor="black", lw=0.3,
                          hatch=HATCHES[i % len(HATCHES)],
                          label=shorten(m, 20) if ax is axes[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([_vlabel(c) for c in conds], fontsize=6.5)
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("fraction of steps keeping their token")
    axes[0].axhline(1.0, color=GREY, lw=0.7, ls=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), fontsize=6,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))

    pt = res.get("test_1c_patchwork_tile_local_period")
    if pt:
        # Regime (iii)'s period is tile-local, so a single global v is the
        # wrong test for it; the correct per-tile figure goes in the margin.
        fig.text(0.01, -0.04,
                 f"Regime (iii) is tiled, so its period is the tile-local "
                 f"$\\lambda_k$: tested per tile it scores "
                 f"{pt['mean']:.3f} mean ({pt['min']:.3f} min) over "
                 f"{pt['n_tiles']} tiles, the shortfall falling entirely on "
                 f"tile boundaries.", fontsize=6, color="0.3")
    fig.tight_layout()
    return [("fig_token_identity", fig)]


def fig_gauge_defect(res):
    """Regime (ii): measured defect against the O(||v||^2) prediction."""
    curve = res.get("gauge_defect_curve")
    if not curve:
        return []
    xs = np.array(sorted(float(k) for k in curve))
    ys = np.array([curve[str(x)] if str(x) in curve else curve[f"{x:.1f}"]
                   for x in xs])
    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 2.4))
    ax.loglog(xs, ys, marker="o", color=color_for("GEO(i)+disp"),
              label="measured defect")
    tail = xs >= 10
    if tail.sum() >= 2:
        c = ys[tail][-1] / xs[tail][-1] ** 2
        ax.loglog(xs[tail], c * xs[tail] ** 2, ls="--", color=GREY,
                  label=r"$O(\|v\|^2)$ prediction")
    s_all = res.get("gauge_defect_loglog_slope_all")
    s_tail = res.get("gauge_defect_loglog_slope_tail")
    txt = []
    if s_tail is not None:
        txt.append(f"tail slope {s_tail:.2f}")
    if s_all is not None:
        txt.append(f"all points {s_all:.2f}")
    if txt:
        ax.annotate("  |  ".join(txt), (0.03, 0.95), xycoords="axes fraction",
                    fontsize=6.5, va="top")
    ax.axvspan(xs.min(), 10, color=GREY, alpha=0.12)
    ax.annotate("interpolation floor", (xs.min() * 1.2, ys.max()), fontsize=6,
                color="0.35", va="top")
    ax.set_xlabel(r"translation $\|v\|$ (m)")
    ax.set_ylabel("deviation from the gauge prediction")
    ax.set_title("Regime (ii) is first-order, as claimed")
    ax.legend(fontsize=6, loc="lower right")
    return [("fig_gauge_defect", fig)]


def fig_equivariant_codebook(res):
    """Does the group act on the DISCRETE tokens by a known permutation?

    Left: measured vs analytic agreement against translation magnitude. The
    prediction is a number to hit, not a bound: on-lattice translations must
    give exactly 1.0, off-lattice ones must match prod(1 - |r_a|).
    Right: held-out agreement of the induced index map, which separates a
    genuine group action from an empirical coincidence.
    """
    curve = res.get("test_2_equivariant_codebook")
    if not curve or "measured" not in next(iter(curve.values()), {}):
        return []
    mags = sorted(float(k) for k in curve)
    def at(m, f):
        return curve[str(m)][f] if str(m) in curve else curve[f"{m:g}"][f]
    meas = [at(m, "measured") for m in mags]
    pred = [at(m, "predicted") for m in mags]
    onlat = [at(m, "on_lattice") for m in mags]

    tr = res.get("test_3_permutation_transfer", {})
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.5),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    ax.plot(mags, pred, ls="--", color=GREY, marker="_",
            label=r"analytic $\prod_a(1-|r_a|)$")
    ax.plot(mags, meas, ls="none", marker="o", color=color_for("GEO(i)+disp"),
            label="measured")
    on = [m for m, o in zip(mags, onlat) if o]
    ax.plot(on, [1.0] * len(on), ls="none", marker="o", mfc="none", mew=1.2,
            ms=9, color=VERMILLION, label="on-lattice (must be exactly 1)")
    ax.set_xscale("log")
    ax.set_xlabel(r"translation $\|v\|$ (m)")
    ax.set_ylabel("permutation agreement")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("The group acts on tokens by a known permutation")

    ax = axes[1]
    if tr:
        key = sorted(tr, key=lambda k: float(k.split("=")[1]))[0]
        methods = list(tr[key])
        x = np.arange(len(methods))
        vals = [tr[key][m]["heldout_agreement"] for m in methods]
        bars = ax.bar(x, vals, 0.6,
                      color=[color_for(m, i) for i, m in enumerate(methods)],
                      edgecolor="black", lw=0.3)
        annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=6)
        for i, m in enumerate(methods):
            if "empirical_matches_analytic" not in tr[key][m]:
                ax.annotate("no analytic\npermutation", (i, vals[i]),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", fontsize=5.5, color="0.35")
            else:
                ax.annotate("fitted map $=$\nanalytic $\\rho(v)$", (i, vals[i]),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", fontsize=5.5,
                            color=color_for(m, i))
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("k-means on the same code",
                                      "k-means on\nthe same code")
                             .replace("equivariant codebook",
                                      "equivariant\ncodebook")
                             .replace("k-means on GPE", "k-means\non GPE")
                            for m in methods], fontsize=6)
        ax.set_ylim(0, 1.28)
        ax.set_ylabel("held-out agreement of the index map")
        ax.set_title(f"Index map generalisation ({key})")
    fig.tight_layout()
    return [("fig_equivariant_codebook", fig)]


# ==========================================================================
# probe.json
# ==========================================================================
def fig_probe(res):
    """Heat map of probe R^2: representation x mobility statistic.

    The point is the *pattern*, not the average. Each channel should lose
    exactly what it was designed to lose: an absolute-position code cannot
    recover path length, a displacement code cannot recover absolute position,
    and the concatenation recovers both.
    """
    reps = [k for k in res if k != "note"]
    if not reps:
        return []
    stats = [s for s in res[reps[0]] if res[reps[0]][s] is not None]
    M = np.array([[res[r].get(s) if res[r].get(s) is not None else np.nan
                   for s in stats] for r in reps], dtype=float)
    M = np.clip(M, -0.2, 1.0)

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 0.42 * len(reps) + 1.6))
    im = ax.imshow(M, cmap="RdYlBu", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([s.replace("_", " ") for s in stats], rotation=30,
                       ha="right")
    ax.set_yticks(range(len(reps)))
    ax.set_yticklabels([shorten(r, 24) for r in reps])
    ax.grid(False)
    for i in range(len(reps)):
        for j in range(len(stats)):
            if np.isfinite(M[i, j]):
                # white on the dark ends of RdYlBu, black in the pale middle
                dark = M[i, j] < 0.05 or M[i, j] > 0.80
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if dark else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"held-out probe $R^2$")
    ax.set_title("What survives tokenization")
    fig.tight_layout()
    return [("fig_probe", fig)]


# ==========================================================================
# bench_*.json
# ==========================================================================
def fig_similarity(res, name="fig_similarity"):
    """Trajectory-similarity metrics per tokenizer.

    MR is on its own axis because lower is better and its scale differs; the
    rank metrics share one.
    """
    methods = [k for k in res if isinstance(res[k], dict) and "MRR" in res[k]]
    if not methods:
        return []
    rank_keys = [k for k in ("MRR", "MP") if k in res[methods[0]]]
    rank_keys += [k for k in res[methods[0]] if k.startswith("KP@")]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.4),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(len(rank_keys))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [res[m].get(k, np.nan) for k in rank_keys]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w,
                      color=color_for(m, i), edgecolor="black", lw=0.3,
                      hatch=HATCHES[i % len(HATCHES)], label=shorten(m, 22))
        annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([k + r" $\uparrow$" for k in rank_keys])
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Similarity ranking")

    ax = axes[1]
    mr = [res[m].get("MR", np.nan) for m in methods]
    bars = ax.bar(range(len(methods)), mr,
                  color=[color_for(m, i) for i, m in enumerate(methods)],
                  edgecolor="black", lw=0.3)
    annotate_bars(ax, bars, mr, fmt="{:.2f}", fontsize=5.5)
    ax.axhline(1.0, color=GREY, lw=0.7, ls=":")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([shorten(m, 14) for m in methods], rotation=25,
                       ha="right", fontsize=6)
    ax.set_ylabel(r"mean rank $\downarrow$")
    ax.set_title("Mean rank (1 = perfect)")
    fig.tight_layout()
    return [(name, fig)]


# ==========================================================================
# transfer_*.json
# ==========================================================================
def fig_transfer(res, name="fig_transfer"):
    """Zero-shot, scale re-fit and fine-tuned MRR on the destination corpus.

    The mechanistic claim is not that one model transfers better in aggregate,
    but that for the GEO model re-fitting *only* the geography-dependent scale
    field closes most of the zero-shot-to-fine-tuned gap.
    """
    methods = [k for k in res if isinstance(res[k], dict) and "zero_shot" in res[k]]
    if not methods:
        return []
    regimes = [("in_domain", "in-domain"), ("zero_shot", "zero-shot"),
               ("refit_scale", "re-fit scale"), ("finetune", "fine-tune")]
    regimes = [r for r in regimes if r[0] in res[methods[0]]]

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 2.5))
    x = np.arange(len(regimes))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [res[m][k]["MRR"] for k, _ in regimes]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w,
                      color=color_for(m, i), edgecolor="black", lw=0.3,
                      hatch=HATCHES[i % len(HATCHES)], label=shorten(m, 22))
        annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5.5)
        gc = res[m].get("gap_closed_by_refit")
        if gc is not None:
            ax.annotate(f"gap closed by re-fit: {gc:.0%}",
                        (0.02, 0.94 - 0.07 * i), xycoords="axes fraction",
                        fontsize=6, color=color_for(m, i))
        if res[m].get("refit_changed_tokenizer") is False:
            ax.annotate("nothing to re-fit", (x[2] + (i - (len(methods) - 1) / 2) * w,
                                              vals[2]),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=5.5, color="0.35", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in regimes])
    ax.set_ylabel(r"MRR on destination corpus $\uparrow$")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.set_title(f"Transfer: {res.get('src', '?')} $\\rightarrow$ "
                 f"{res.get('dst', '?')}")
    return [(name, fig)]


# ==========================================================================
# arch.json  (GPE Sec 5.5, architectures)
# ==========================================================================
def fig_architectures(res):
    """Is the tokenizer's contribution a property of the representation, or an
    interaction with one sequence model?

    Grouped bars per backbone. If the ordering of tokenizers is stable across
    LSTM, Transformer and GNN, the word "tokenizer" is earned; if it flips,
    the effect belongs to the architecture and the claim must be narrowed.
    The Spearman correlations between backbones are printed on the figure so
    that judgement is not left to the eye.
    """
    cells = res.get("cells")
    if not cells:
        return []
    names, kinds = list(cells), res.get("kinds", [])
    has_tr = any("transfer" in cells[n].get(k, {}) for n in names for k in kinds)
    ncol = 2 if has_tr else 1

    fig, axes = plt.subplots(1, ncol, figsize=(WIDTH_2COL, 2.6), squeeze=False)
    panels = [("local", "local")] + ([("transfer", "transfer")] if has_tr else [])
    for ax, (field, title) in zip(axes[0], panels):
        x = np.arange(len(kinds))
        w = 0.8 / max(len(names), 1)
        for i, n in enumerate(names):
            vals = [cells[n].get(k, {}).get(field, {}).get("MRR", np.nan)
                    for k in kinds]
            bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w,
                          color=color_for(n, i), edgecolor="black", lw=0.3,
                          hatch=HATCHES[i % len(HATCHES)],
                          label=shorten(n, 20) if field == "local" else None)
            annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(kinds)
        ax.set_ylabel(r"MRR $\uparrow$")
        ax.set_ylim(0, 1.12)
        lat = res.get("transfer_lat")
        ax.set_title(title if field == "local"
                     else rf"transfer (relocated to {lat}$^\circ$)")
    axes[0][0].legend(fontsize=6, ncol=2, loc="upper left")

    ra = res.get("rank_agreement") or {}
    if ra:
        txt = "  |  ".join(f"{k}: $\\rho$={v:.2f}" for k, v in ra.items())
        fig.text(0.01, -0.04, "Rank agreement between backbones  " + txt,
                 fontsize=6, color="0.3")
    fig.tight_layout()
    tag = res.get("dataset", "").replace("/", "_") or "data"
    return [(f"fig_architectures_{tag}", fig)]


# ==========================================================================
# roadnet.json  (GPE Sec 5.5, road network)
# ==========================================================================
def fig_roadnetwork(res):
    """Vertex features on a road network, ranked against network-aware
    ground truths. The question is whether geometry (a position embedding)
    and topology (node2vec) are complementary, which the half-dimension
    concatenated rows test without simply adding capacity.
    """
    rows = res.get("rows")
    if not rows:
        return []
    names = list(rows)
    measures = [k for k in rows[names[0]] if k.startswith("HR@")]
    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 2.6))
    x = np.arange(len(measures))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [rows[n].get(m, np.nan) for m in measures]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w,
                      color=color_for(n, i), edgecolor="black", lw=0.3,
                      hatch=HATCHES[i % len(HATCHES)], label=shorten(n, 24))
        annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5,
                      rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("HR@", "HR@").replace(" ", "\n")
                        for m in measures])
    ax.set_ylabel(r"hit ratio $\uparrow$")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=6, ncol=3, loc="upper left")
    ax.set_title(f"Road-network vertex features "
                 f"({res.get('n_nodes', '?')} vertices, "
                 f"{res.get('graph', '?')} graph)")
    if res.get("graph") == "trajectory-induced":
        fig.text(0.01, -0.05,
                 "The network here is induced from the trajectories "
                 "themselves, not imported from OSM, so it shares its support "
                 "with the data the models see; pass --graph-path for a real "
                 "road graph.", fontsize=6, color="0.3")
    fig.tight_layout()
    tag = "osm" if res.get("graph") == "osm" else "induced"
    return [(f"fig_roadnetwork_{tag}", fig)]


# ==========================================================================
# anomaly.json
# ==========================================================================
def fig_anomaly(res):
    """Two panels: the per-type AUROC heat map, and the orientation control.

    The heat map carries the whole argument. Read it by row for what a
    representation can see, and by column for which channel a given anomaly
    needs. The `[trivial]` rows are the guard: wherever one of them matches
    the tokenizer, that column demonstrates nothing about the representation.
    """
    runs = res.get("runs")
    if not runs:
        return []
    sev = max(runs, key=lambda x: float(x))
    row = runs[sev]
    types = [t for t in res["types"] if t != "off_grid_control"]
    names = list(row)

    M = np.array([[row[n].get(t, np.nan) for t in types] for n in names],
                 dtype=float)
    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 0.32 * len(names) + 1.9))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=25,
                       ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([shorten(n, 30) for n in names], fontsize=6)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(types)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if M[i, j] > 0.85 or M[i, j] < 0.42
                        else "black")
    # separate the guard rows from the representations
    n_rep = sum(1 for n in names if not n.startswith("[trivial]"))
    if 0 < n_rep < len(names):
        ax.axhline(n_rep - 0.5, color="black", lw=1.1)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC (0.5 = chance)")
    ax.set_title(f"Anomaly detection by type, severity {sev}")
    fig.tight_layout()
    figs = [("fig_anomaly", fig)]

    ori = (res.get("prediction_check") or {}).get("orientation_sensitivity")
    if ori:
        fig2, ax2 = plt.subplots(figsize=(WIDTH_2COL, 2.4))
        ks = list(ori)
        x = np.arange(len(ks))
        w = 0.38
        a = [ori[k]["off_grid"] for k in ks]
        b = [ori[k]["control_90deg"] for k in ks]
        ax2.bar(x - w / 2, a, w, color=BLUE, edgecolor="black", lw=0.3,
                label=r"off-grid (45$^\circ$)")
        ax2.bar(x + w / 2, b, w, color=GREY, edgecolor="black", lw=0.3,
                hatch="///", label=r"control (90$^\circ$, still aligned)")
        ax2.axhline(0.5, color=GREY, lw=0.7, ls=":")
        ax2.set_xticks(x)
        ax2.set_xticklabels([shorten(k, 16) for k in ks], rotation=25,
                            ha="right", fontsize=5.5)
        ax2.set_ylabel("AUROC")
        ax2.legend(fontsize=6)
        ax2.set_title("Orientation control: both rotations move position "
                      "equally, only 45$^\circ$ breaks grid alignment")
        fig2.text(0.01, -0.05,
                  "Only a positive 45$^\circ$-minus-90$^\circ$ gap is evidence of "
                  "orientation sensitivity; a negative gap means the signal was "
                  "positional all along.", fontsize=6, color="0.3")
        fig2.tight_layout()
        figs.append(("fig_anomaly_orientation", fig2))
    return figs


# ==========================================================================
# lambda.json
# ==========================================================================
def fig_lambda(res):
    """HR@k against lambda for each time embedding, plus the period-shift test."""
    grid = res.get("grid")
    if not grid:
        return []
    lams = sorted((float(x) for x in grid), reverse=True)
    kinds = res["time_kinds"]
    kk = [k for k in grid[str(lams[0])][kinds[0]] if k.startswith("HR@")
          and "shift" not in k][0]

    has_shift = res.get("shift_test_h") is not None
    fig, axes = plt.subplots(1, 2 if has_shift else 1,
                             figsize=(WIDTH_2COL, 2.6), squeeze=False)
    ax = axes[0][0]
    for i, kind in enumerate(kinds):
        ys = [grid[str(l)][kind][kk] for l in lams]
        ax.plot(lams, ys, marker=MARKERS[i % len(MARKERS)],
                ls=LINESTYLES[i % len(LINESTYLES)], color=PALETTE[i % len(PALETTE)],
                label=kind)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\lambda$   (1 = purely spatial $\rightarrow$ 0 = purely temporal)")
    ax.set_ylabel(rf"{kk} $\uparrow$")
    ax.set_title(f"{res.get('measure','')} ground truth, "
                 f"temporal mode = {res.get('temporal_mode','')}")
    ax.legend(fontsize=6, title="time channel", title_fontsize=6)

    if has_shift:
        ax = axes[0][1]
        x = np.arange(len(kinds))
        w = 0.8 / max(len(lams), 1)
        for j, l in enumerate(lams):
            ys = [grid[str(l)][k].get("embedding_drift", np.nan) for k in kinds]
            ax.bar(x + (j - (len(lams) - 1) / 2) * w, ys, w,
                   color=PALETTE[j % len(PALETTE)], edgecolor="black", lw=0.3,
                   label=rf"$\lambda$={l:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(kinds, rotation=20, ha="right", fontsize=6)
        ax.set_ylabel("relative embedding drift")
        ax.set_title(rf"shift by {res['shift_test_h']:g} h "
                     rf"(a period of $C_{{24}}\times C_7$)")
        ax.legend(fontsize=5.5, ncol=2)
        fig.text(0.01, -0.05,
                 "A whole-week shift is a period of the cyclic group, so an "
                 "exactly periodic code is unchanged; any non-periodic (linear) "
                 "term drifts.", fontsize=6, color="0.3")
    fig.tight_layout()
    return [("fig_lambda", fig)]


# ==========================================================================
# tul.json
# ==========================================================================
def fig_tul(res):
    """Trajectory-user linking accuracy against the majority-class baseline."""
    models = res.get("models", {})
    if not models:
        return []
    names = list(models)
    keys = [k for k in models[names[0]] if k.startswith("acc@")]
    fig, ax = plt.subplots(figsize=(WIDTH_1COL * 1.4, 2.4))
    x = np.arange(len(names))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [models[n][k] for n in names]
        bars = ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w,
                      color=[color_for(n, j) for j, n in enumerate(names)],
                      edgecolor="black", lw=0.3,
                      hatch=HATCHES[i % len(HATCHES)], label=k)
        annotate_bars(ax, bars, vals, fmt="{:.2f}", fontsize=5.5)
    mb = res.get("majority_baseline")
    if mb is not None:
        ax.axhline(mb, color=GREY, lw=0.8, ls="--")
        ax.annotate(f"majority class ({mb:.2f})", (len(names) - 0.5, mb),
                    fontsize=6, color=GREY, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([shorten(n, 16) for n in names], rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=6, ncol=len(keys))
    ax.set_title(f"Trajectory-user linking ({res.get('n_users', '?')} users, "
                 f"split='{res.get('split', '?')}')")
    return [("fig_tul", fig)]
