"""Encoder-level audit: the table that positions this work against GPE.

Four blocks, all cheap (no training), all run on the same trajectories:

  translation  -- is the encoder a translation GEO? (GPE: yes, analytically)
  rotation     -- is it an SO(2) GEO? (GPE: no; the displacement channel: yes)
  latitude     -- does it survive being moved to another latitude with its
                  metric shape preserved? This is the instrument for H6.
                  GPE's own datasets sit at 40-42 N, so their cross-city
                  experiments never vary cos(lat) by more than ~3%.
  noise        -- how fast does the code decorrelate under GPS jitter?
                  GPE's finest wavelength lands just above the noise floor,
                  which their own epsilon ablation hints at without naming.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import save_results
from ..data import load_dataset, make_synthetic
from ..encoders import (GPE, DisplacementGEO, GridEmbed, LearnedFourier, RawXY,
                        Space2Vec, TorusLattice)
from ..equivariance import (empirical_lipschitz, encoder_equivariance_error,
                            rotation_equivariance_error, saturation_curve)
from ..geo import (R_EARTH, enu_to_lonlat, lon_scale_ratio, lonlat_to_enu,
                   reproject_to_latitude)


def _build(ref_lat=41.15):
    return {
        "GPE(h=128)": GPE(h=128, eps_over_2pi=1e-6),
        "GPE(h=256)": GPE(h=256, eps_over_2pi=1e-6),
        "Space2Vec": Space2Vec(n_scales=8),
        "TorusLattice(i)": TorusLattice(),
        "RawXY": RawXY(normalize=False),
        "Grid(150m)": GridEmbed(cell_m=150.0),
        "Fourier/TriW": LearnedFourier(dim=128),
    }


def run(dataset="synthetic", path=None, n_points=4000, seed=0, out=None):
    if dataset == "synthetic":
        trajs = make_synthetic(n=400, seed=seed)[0]
    else:
        trajs = load_dataset(dataset, path)[:400]
    lonlat = np.concatenate([t.lonlat for t in trajs])[:n_points]
    ref_lat = float(lonlat[:, 1].mean())
    encs = _build(ref_lat)
    for e in encs.values():
        if hasattr(e, "fit") and e.__class__.__name__ == "RawXY":
            e.fit(lonlat)
        if hasattr(e, "ref_deg"):
            e.ref_deg = lonlat.mean(0)

    rng = np.random.default_rng(seed)
    res: dict = {
        "dataset": dataset, "ref_lat": ref_lat, "encoders": {},
        "translation_note": (
            "Raw coordinates are translation- and rotation-equivariant in the "
            "trivial affine/orthogonal sense, so they score well here. That is "
            "worth stating plainly rather than hiding: the case for the GEO "
            "tokenizer is not that symmetry is otherwise unobtainable, but "
            "that it is obtainable *together with* boundedness and multi-scale "
            "resolution -- which is what the transfer and probe experiments "
            "measure and what raw coordinates conspicuously lack."),
    }

    # ---------------------------------------------------------------- 1
    v = rng.normal(0, 500.0, size=2)          # metres
    xy, ref = lonlat_to_enu(lonlat)
    translate = lambda ll: enu_to_lonlat(lonlat_to_enu(ll, ref)[0] + v, ref)

    for name, enc in encs.items():
        rho = None
        if isinstance(enc, GPE):
            rho = enc.rho(v, ref_lat_deg=ref_lat)
        elif hasattr(enc, "rho"):
            try:
                rho = enc.rho(v)
            except Exception:
                rho = None
        r = encoder_equivariance_error(enc, lonlat, translate, rho=rho, seed=seed)
        res["encoders"].setdefault(name, {})["translation_err"] = r["error"]
        res["encoders"][name]["rho"] = r["rho"]
        res["encoders"][name]["dim"] = enc.dim
        try:
            res["encoders"][name]["lipschitz_per_m"] = enc.lipschitz()
        except Exception:
            res["encoders"][name]["lipschitz_per_m"] = None

    # ---------------------------------------------------------------- 2
    dxy = np.concatenate([np.diff(t.xy(ref), axis=0) for t in trajs])[:n_points]
    res["rotation"] = {
        "DisplacementGEO(m=1)": rotation_equivariance_error(
            DisplacementGEO(harmonics=(1,)), dxy, seed=seed),
        "DisplacementGEO(m=1,2,3)": rotation_equivariance_error(
            DisplacementGEO(harmonics=(1, 2, 3)), dxy, seed=seed),
    }
    for name, enc in encs.items():
        rot = lambda ll, th=0.7: enu_to_lonlat(
            lonlat_to_enu(ll, ref)[0] @ np.array(
                [[np.cos(0.7), np.sin(0.7)], [-np.sin(0.7), np.cos(0.7)]]), ref)
        res["rotation"][name] = encoder_equivariance_error(
            enc, lonlat, rot, rho=None, seed=seed)["error"]

    # ---------------------------------------------------------------- 3
    res["latitude"] = {}
    res["latitude_note"] = (
        "Gram distortion of the L2-normalised code under a metric-preserving "
        "move to another latitude. Codes of dimension < 4 (RawXY, Grid) give "
        "degenerate Gram statistics and are reported as null; read their "
        "latitude behaviour off the transfer experiment instead. The source "
        "latitude itself is included as a zero-check.")
    for target in (round(ref_lat, 2), 0.0, 20.0, 40.0, 60.0):
        moved = reproject_to_latitude(lonlat, target)
        row = {"lon_stretch": lon_scale_ratio(ref_lat, target)}
        for name, enc in encs.items():
            if enc.dim < 4:
                row[name] = None
                continue
            row[name] = _gram_distortion(enc, lonlat, moved)
            # A metric encoder can have its local frame re-fitted to the new
            # region -- that is exactly Experiment G's "re-fit only the
            # geography-dependent part". GPE has no such part: its frequency
            # ladder is fixed in angular units worldwide.
            if hasattr(enc, "ref_deg"):
                import copy as _copy
                e2 = _copy.deepcopy(enc)
                e2.ref_deg = moved.mean(0)
                row[name + " [refit ref]"] = _gram_distortion(enc, lonlat,
                                                              moved, e2)
        # the metric channel is latitude-invariant by construction
        g = DisplacementGEO()
        d0 = np.diff(lonlat_to_enu(lonlat, lonlat.mean(0))[0], axis=0)
        d1 = np.diff(lonlat_to_enu(moved, moved.mean(0))[0], axis=0)
        row["DisplacementGEO"] = float(
            np.abs(g.encode_disp(d0) - g.encode_disp(d1)).mean())
        res["latitude"][str(target)] = row

    # ---------------------------------------------------------------- 4
    res["noise_saturation"] = {n: saturation_curve(e, lonlat, seed=seed)
                               for n, e in encs.items()}
    res["empirical_lipschitz"] = {n: empirical_lipschitz(e, lonlat, seed=seed)
                                  for n, e in encs.items()}
    res["gpe_finest_wavelength_m"] = {
        "h=128": GPE(128).finest_wavelength_m(),
        "h=256": GPE(256).finest_wavelength_m()}

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


if __name__ == "__main__":
    run()


def _gram_distortion(enc, lonlat, moved, enc_moved=None):
    """Mean |Gram(enc(l)) - Gram(enc(l'))| for L2-normalised codes.

    An exactly equivariant code has an orthogonal rho, which preserves inner
    products, so the Gram matrix is invariant under the group action; any
    distortion is therefore a direct readout of broken equivariance.
    """
    E0 = enc.encode(lonlat)
    E1 = (enc_moved or enc).encode(moved)
    n0 = E0 / (np.linalg.norm(E0, axis=1, keepdims=True) + 1e-12)
    n1 = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.abs(n0 @ n0.T - n1 @ n1.T)))
