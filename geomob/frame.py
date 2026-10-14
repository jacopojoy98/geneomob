"""How GEO's local frame is chosen: origin, orientation, coordinates.

All controlled from `[encoders]`:

    frame_origin       "mean"    centre of mass of the training points (v<=0.8
                                 behaviour; moves with every sample)
                       "median"  geometric median: robust to far-away trips,
                                 and it moves exactly with the city when the city
                                 is shifted or turned
                       "anchor"  a fixed city centre: identical for training,
                                 validation, test and every run. The anchor is the
                                 nearest entry of CITY_ANCHORS (+ `frame_anchors`)
                                 within `frame_anchor_radius_km` of the data's
                                 median; otherwise it falls back to the median
    frame_orientation  "none"    x = east, y = north
                       "grid"    x along the dominant street direction, from the
                                 m = 4 harmonic of step headings; of the four
                                 grid-aligned directions (a square grid cannot
                                 tell them apart) the one nearest the principal
                                 axis is taken. Where there is no grid
                                 (strength < 0.02, e.g. Rome) it falls
                                 back to "pca"
                       "pca"     x along the longest spread of the points, pointed
                                 at the heavier tail
    geo_coords         "metric"  location channel in metres in that local frame
                       "lonlat"  location channel on raw longitude / latitude
                                 (x = R * lon, y = R * lat in radians, i.e.
                                 "metres at the equator"): GPE's coordinates with
                                 GEO's wavelengths. No origin, no orientation;
                                 east-west wavelengths shrink by cos(latitude).

The displacement and speed channels always use the local metric frame (only
its orientation matters for them).
"""
from __future__ import annotations

import numpy as np

from .geo import R_EARTH, lonlat_to_enu, rot2

# city centres (lon, lat) for frame_origin = "anchor"
CITY_ANCHORS = {
    "beijing": (116.3975, 39.9087),       # T-Drive, GeoLife
    "porto": (-8.6110, 41.1496),
    "rome": (12.4964, 41.9028),
    "san_francisco": (-122.4194, 37.7749),
    "singapore": (103.8198, 1.3521),
    "new_york": (-73.9857, 40.7484),
}

_warned: set = set()


def _hp():
    from .config import HP
    return HP["encoders"]


def _points(trajs_or_points):
    if isinstance(trajs_or_points, np.ndarray):
        return trajs_or_points
    return np.concatenate([t.lonlat for t in trajs_or_points])


def _haversine_km(a, b):
    a, b = np.deg2rad(a), np.deg2rad(b)
    d = np.sin((b[1] - a[1]) / 2) ** 2 + np.cos(a[1]) * np.cos(b[1]) * np.sin((b[0] - a[0]) / 2) ** 2
    return 2 * R_EARTH / 1000.0 * np.arcsin(np.sqrt(d))


def geometric_median(ll, iters=1000, tol_m=1e-5):
    """Point minimising the summed distance to all points (Weiszfeld), in the
    plane. Unlike a per-coordinate median it moves exactly with the data under
    rotations as well as translations, and like it, it ignores far outliers."""
    from .geo import enu_to_lonlat
    ref = np.median(ll, axis=0)
    xy = lonlat_to_enu(ll, ref)[0]
    z = np.median(xy, axis=0)
    for _ in range(iters):
        d = np.maximum(np.hypot(*(xy - z).T), 1e-6)
        z_new = (xy / d[:, None]).sum(0) / (1.0 / d).sum()
        if np.hypot(*(z_new - z)) < tol_m:
            z = z_new
            break
        z = z_new
    return enu_to_lonlat(z[None, :], ref)[0]


def fit_origin(trajs_or_points, rule=None):
    """The frame origin (lon, lat) under the configured rule."""
    ll = _points(trajs_or_points)
    rule = rule or _hp().get("frame_origin", "mean")
    if rule == "mean":
        return ll.mean(0)
    med = geometric_median(ll)
    if rule == "median":
        return med
    if rule == "anchor":
        anchors = dict(CITY_ANCHORS)
        anchors.update({k: tuple(v) for k, v in (_hp().get("frame_anchors") or {}).items()})
        radius = float(_hp().get("frame_anchor_radius_km", 75.0))
        best = min(anchors.items(), key=lambda kv: _haversine_km(med, np.asarray(kv[1])))
        if _haversine_km(med, np.asarray(best[1])) <= radius:
            return np.asarray(best[1], dtype=np.float64)
        key = (round(float(med[0]), 1), round(float(med[1]), 1))
        if key not in _warned:
            _warned.add(key)
            print(f"  [frame] no anchor within {radius:g} km of "
                  f"({med[0]:.3f}, {med[1]:.3f}); using the median (add one with "
                  f"[encoders] frame_anchors = {{name = [lon, lat]}})")
        return med
    raise ValueError(f"frame_origin must be mean, median or anchor, got {rule!r}")


def grid_angle(trajs, ref, min_step_m=5.0):
    """Dominant street direction in (-45, 45] degrees, from the 4th harmonic
    of step headings weighted by step length; and its strength in [0, 1]
    (0 = no preferred direction, 1 = every step on one grid)."""
    acc, wsum = 0j, 0.0
    for t in trajs:
        d = np.diff(lonlat_to_enu(t.lonlat, ref)[0], axis=0)
        r = np.hypot(d[:, 0], d[:, 1])
        keep = r >= min_step_m
        if not keep.any():
            continue
        psi = np.arctan2(d[keep, 1], d[keep, 0])
        acc += np.sum(r[keep] * np.exp(4j * psi))
        wsum += float(r[keep].sum())
    if wsum == 0:
        return 0.0, 0.0
    return float(np.angle(acc) / 4.0), float(abs(acc) / wsum)


def pca_angle(points, ref):
    xy = lonlat_to_enu(points, ref)[0]
    xy = xy - xy.mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    ax = v[:, np.argmax(w)]
    ang = float(np.arctan2(ax[1], ax[0]))
    if np.mean((xy @ ax) ** 3) < 0:
        ang += np.pi
    return ang, float(w.min() / max(w.max(), 1e-12))     # angle, isotropy


def fit_orientation(trajs, ref, rule=None):
    rule = rule or _hp().get("frame_orientation", "none")
    if rule == "none":
        return 0.0
    if rule == "grid":
        ang, strength = grid_angle(trajs, ref)
        if strength < 0.02:
            # no street grid to align to: use the principal axis, which turns
            # with the data. (Before v0.9.3 this returned 0, i.e. due east,
            # which does NOT turn with the city, so a rotated gridless city
            # got a different frame.)
            if "nogrid" not in _warned:        # once per run
                _warned.add("nogrid")
                print(f"  [frame] no street grid near ({ref[0]:.2f}, {ref[1]:.2f}) "
                      f"(strength {strength:.3f}); orienting by the principal axis")
            return pca_angle(_points(trajs), ref)[0]
        # a square grid fixes the axes only up to 90 degrees; pick, of the four
        # grid-aligned directions, the one closest to the principal axis
        # (which is defined up to 360 degrees), so a turned city gets a turned
        # frame rather than one with x and y swapped
        p, _ = pca_angle(_points(trajs), ref)
        cands = ang + np.pi / 2 * np.arange(4)
        diff = np.angle(np.exp(1j * (cands - p)))
        return float(cands[np.argmin(np.abs(diff))])
    if rule == "pca":
        return pca_angle(_points(trajs), ref)[0]
    raise ValueError(f"frame_orientation must be none, grid or pca, got {rule!r}")


def fit_frame(loc, trajs, force=False):
    """Set loc.ref_deg (origin) and loc.frame_angle (orientation) from data.
    `loc` is any channel with a `ref_deg` attribute."""
    if getattr(loc, "ref_deg", None) is None or force:
        loc.ref_deg = fit_origin(trajs)
        loc.frame_angle = fit_orientation(trajs, loc.ref_deg)
    return loc


def metric_xy(lonlat, loc=None):
    """Metres in the local frame of `loc` (origin + orientation)."""
    ref = getattr(loc, "ref_deg", None) if loc is not None else None
    xy = lonlat_to_enu(lonlat, ref)[0]
    ang = getattr(loc, "frame_angle", 0.0) if loc is not None else 0.0
    if ang:
        xy = xy @ rot2(-ang).T
    return xy


def lonlat_xy(lonlat):
    """GPE's coordinates in metres-at-the-equator: x = R lon, y = R lat."""
    return R_EARTH * np.deg2rad(np.asarray(lonlat, dtype=np.float64))


def describe(loc):
    if loc is None or getattr(loc, "ref_deg", None) is None:
        return "no frame"
    if getattr(loc, "coords", "metric") == "lonlat":
        return "raw longitude/latitude (no frame)"
    r = loc.ref_deg
    return (f"origin ({r[0]:.4f}, {r[1]:.4f}), "
            f"x-axis at {np.degrees(getattr(loc, 'frame_angle', 0.0)):+.1f} deg from east")
