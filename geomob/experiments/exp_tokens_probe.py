"""Experiment C -- token consistency of the location code.
Experiment F -- what the tokens preserve.

Experiment C is run in the corrected form described in `codebook.py`: the
falsifiable prediction is exact index preservation under *lattice-period*
translations, and predicted-permutation agreement under an equivariant
codebook, not index preservation under arbitrary v.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import save_results
from ..codebook import (EquivariantTorusCodebook, KMeansCodebook,
                        codebook_agreement_curve, equivariant_codebook_test,
                        fit_loglog_slope, gauge_defect_curve,
                        lattice_period_test, permutation_transfer_test)
from ..data import load_dataset, make_synthetic
from ..encoders import (GPE, DisplacementGEO, PatchworkLattice, RawXY,
                        SeparableWarp, TorusLattice)
from ..geo import lonlat_to_enu
from ..metrics import mobility_stats, pool_tokens, ridge_probe


# ==========================================================================
# Experiment C
# ==========================================================================
def run_tokens(dataset="synthetic", path=None, n_codes=512, seed=0, out=None):
    trajs = (make_synthetic(n=600, seed=seed)[0] if dataset == "synthetic"
             else load_dataset(dataset, path)[:600])
    lonlat = np.concatenate([t.lonlat for t in trajs])
    ref = lonlat.mean(0)

    lam = 1000.0
    regimes = {
        "(i) uniform lattice": TorusLattice(lams=(lam,), ref_deg=ref),
        "(ii) separable warp": SeparableWarp(ref_deg=ref).fit(lonlat),
        "(iii) patchwork": PatchworkLattice(ref_deg=ref).fit(lonlat),
        "naive raw xy": RawXY(normalize=True).fit(lonlat),
        "GPE": GPE(h=64),
    }

    res = {"lambda_m": lam, "test_1_lattice_period": {},
           "test_1b_arbitrary_v": {}, "test_2_equivariant_codebook": {},
           "codebook_perplexity": {}}

    books = {}
    for name, enc in regimes.items():
        cb = KMeansCodebook(n_codes, seed=seed).fit(enc.encode(lonlat))
        books[name] = cb
        res["codebook_perplexity"][name] = cb.perplexity(cb.encode(enc.encode(lonlat)))

    # ---- test 1: v IS a lattice period -> must be exactly 1.0 for (i)
    for v in [(lam, 0.0), (0.0, lam), (2 * lam, lam)]:
        res["test_1_lattice_period"][f"v={v}"] = {
            n: lattice_period_test(e, books[n], lonlat, v, ref)
            for n, e in regimes.items()}
    res["test_1_note"] = (
        "Regime (iii) is NOT expected to score 1.0 here: its period is the "
        "tile-local lambda_k, so a single global v is a period only in tiles "
        "where lambda_k divides it. The per-tile version below is the correct "
        "test for (iii), and it is the one that must come out exact.")

    # ---- test 1c: regime (iii) with the correct, tile-local period
    patch = regimes["(iii) patchwork"]
    tiles = patch.tile_ids(lonlat)
    cb = books["(iii) patchwork"]
    per_tile = []
    for key in np.unique(tiles, axis=0)[:40]:
        m = (tiles == key).all(1)
        if m.sum() < 20:
            continue
        lam_k = patch.lam_by_tile.get(tuple(key), patch.lam_default)
        per_tile.append(lattice_period_test(patch, cb, lonlat[m],
                                            (lam_k, 0.0), ref))
    res["test_1c_patchwork_tile_local_period"] = {
        "mean": float(np.mean(per_tile)), "min": float(np.min(per_tile)),
        "n_tiles": len(per_tile)}

    # ---- test 1b: arbitrary v -> nobody should score 1.0; this is the
    #      prediction the draft got wrong, kept here to make that explicit
    for mag in (50.0, 200.0, 500.0):
        v = (mag / np.sqrt(2), mag / np.sqrt(2))
        res["test_1b_arbitrary_v"][f"|v|={mag}"] = {
            n: lattice_period_test(e, books[n], lonlat, v, ref)
            for n, e in regimes.items()}

    # ---- test 2: equivariant codebook.
    # The claim is not that tokens are preserved but that the group acts on
    # them by a KNOWN permutation. On-lattice translations must agree at 1.0
    # exactly; off-lattice ones must match the analytic prediction
    # prod_a (1 - |r_a|), which is a number to hit, not a bound to stay under.
    bins = 16
    tpc = EquivariantTorusCodebook(lam=lam, bins=bins, ref_deg=ref)
    cell = lam / bins
    mags = [cell / 2, cell * 0.75, cell, cell * 1.5, cell * 2, cell * 4,
            cell * 8, lam]
    res["test_2_equivariant_codebook"] = codebook_agreement_curve(
        tpc, lonlat, mags, ref=ref)
    res["test_2_note"] = (
        f"codebook: {bins}x{bins} phase cells of {cell:.1f} m on a "
        f"{lam:.0f} m torus. 'on_lattice' translations are multiples of the "
        "cell size, where the permutation is exact.")

    # ---- test 3: does the induced index map generalise off the points it was
    # fitted on? This is what separates a group action from a coincidence.
    res["test_3_note"] = (
        "Read held-out agreement, not the generalisation gap: for a coarse "
        "k-means codebook and a small v the best empirical map is near the "
        "identity, which generalises well while agreeing only partially. Only "
        "the equivariant codebook admits an analytic permutation at all, and "
        "its empirically fitted map coincides with it.")
    res["test_3_permutation_transfer"] = {}
    for mag in (cell, cell * 4, lam):
        v = (float(mag), 0.0)
        res["test_3_permutation_transfer"][f"|v|={mag:.1f}"] = {
            "equivariant codebook": permutation_transfer_test(
                None, tpc, lonlat, v, ref, seed=seed),
            "k-means on the same code": permutation_transfer_test(
                regimes["(i) uniform lattice"], books["(i) uniform lattice"],
                lonlat, v, ref, seed=seed),
            "k-means on GPE": permutation_transfer_test(
                regimes["GPE"], books["GPE"], lonlat, v, ref, seed=seed)}

    # ---- regime (ii): is the defect really O(||v||^2)?
    vs = [(m, 0.0) for m in (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)]
    curve = gauge_defect_curve(regimes["(ii) separable warp"], lonlat, vs, ref)
    res["gauge_defect_curve"] = curve
    res["gauge_defect_loglog_slope_all"] = fit_loglog_slope(curve)
    res["gauge_defect_loglog_slope_tail"] = fit_loglog_slope(curve, min_v=10.0)
    res["gauge_defect_slope_prediction"] = 2.0
    res["gauge_defect_note"] = (
        "Prediction (5) is asymptotic. Below ~10 m the measurement sits on a "
        "floor set by the resolution of the interpolant for eta, so read the "
        "tail slope; the all-points slope understates it.")

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


# ==========================================================================
# Experiment F
# ==========================================================================
def run_probe(dataset="synthetic", path=None, n=1500, seed=0, out=None):
    """Probes for statistics that the construction should and should not lose.

    The sharp prediction: `centroid_dist` (which torus wrap a point is in) and
    `radius_of_gyration` degrade for wrapped codes, while `total_distance`,
    `mean_step` and heading do not. A representation that loses exactly what
    it was designed to lose, and nothing else, shows that pattern; an
    unconstrained embedding shows no particular pattern.
    """
    trajs = (make_synthetic(n=n, seed=seed)[0] if dataset == "synthetic"
             else load_dataset(dataset, path)[:n])
    lonlat = np.concatenate([t.lonlat for t in trajs])
    ref = lonlat.mean(0)

    reps = {
        "GEO(i)+disp": ("torus_disp", None),
        "GEO(i) only": ("torus", None),
        "disp only": ("disp", None),
        "GPE": ("gpe", None),
        "raw xy (upper bound)": ("raw", None),
    }
    torus = TorusLattice(ref_deg=ref)
    gpe = GPE(h=64)
    disp = DisplacementGEO()

    def featurise(tag, tr):
        xy = lonlat_to_enu(tr.lonlat, ref)[0]
        d = np.diff(xy, axis=0)
        if tag == "torus":
            return pool_tokens(torus.encode(tr.lonlat))
        if tag == "disp":
            return pool_tokens(disp.encode_disp(d))
        if tag == "torus_disp":
            return np.concatenate([pool_tokens(torus.encode(tr.lonlat)),
                                   pool_tokens(disp.encode_disp(d))])
        if tag == "gpe":
            return pool_tokens(gpe.encode(tr.lonlat))
        return pool_tokens(xy / 1000.0)

    stats = [mobility_stats(lonlat_to_enu(t.lonlat, ref)[0]) for t in trajs]
    keys = list(stats[0])
    Y = {k: np.array([s[k] for s in stats]) for k in keys}

    res = {"note": (
        "Statistics with near-zero variance in this corpus are reported as "
        "null rather than R^2 ~ 1; on synthetic data `stay_points` is usually "
        "one of them, since the generator produces no stays.")}
    for label, (tag, _) in reps.items():
        X = np.stack([featurise(tag, t) for t in trajs])
        res[label] = {k: ridge_probe(X, Y[k], seed=seed) for k in keys}
        print(label, {k: (round(v, 3) if v is not None else "degenerate")
                      for k, v in res[label].items()}, flush=True)

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


if __name__ == "__main__":
    run_tokens()
    run_probe()
