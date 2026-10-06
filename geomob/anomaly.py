"""Channel-aligned anomaly injection.

Synthetic-anomaly benchmarks are usually weak, because the generator is under
the author's control and a detector can succeed by learning an artefact of the
injection rather than the anomaly. This module is built to avoid that in three
ways:

1. **Each anomaly type is tied to a channel of the tokenizer**, and the
   prediction of which channel should catch it is written down here, in
   `EXPECTED`, before anything is run. The experiment then either confirms it
   or does not.

2. **Two of the types are predicted to be MISSED by specific ablations.** An
   `off_grid` anomaly is invisible to a rotation-*invariant* model, and an
   `unseen_area` anomaly is invisible to a purely relative one. Showing your
   own ablations failing where you said they would is much stronger evidence
   than winning everywhere.

3. **Marginals are matched where possible.** A detour preserves the endpoints;
   a kinematic anomaly preserves the path; an off-grid rotation preserves every
   invariant (lengths, angles between steps, duration) exactly. That prevents
   the trivial detectors in `trivial_scores` from finding them for free, and
   those detectors are reported alongside the model so the reader can see the
   gap rather than take it on trust.

Every injector returns a new `Traj` and never mutates its input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Traj
from .geo import enu_to_lonlat, lonlat_to_enu, rot2

# Which channel of the tokenizer should register each anomaly, and which
# ablation should provably fail to. Written down before running anything.
EXPECTED = {
    "detour": {
        "perturbs": "route between fixed endpoints",
        "caught_by": "relative channel (turn/step sequence)",
        "missed_by": None},
    "kinematic": {
        "perturbs": "speed profile, path unchanged",
        # Corrected after the first run contradicted the original prediction.
        # phi(dx) encodes the displacement *magnitude*, which a re-timing does
        # not touch, and the time channel encodes absolute time-of-day, not
        # dt. So NO channel of the base construction sees this, and the run
        # confirmed it (AUROC ~0.5) while a one-line mean-speed detector got
        # 1.000. That is a genuine gap in the tokenizer, not a weak anomaly:
        # it needs an explicit dt / speed channel, which `SpeedGEO` supplies.
        "caught_by": "a dt / speed channel (SpeedGEO); NOT phi(dx) or phi_time",
        "missed_by": "loc + disp + time without a speed channel"},
    "wrong_time": {
        "perturbs": "timestamps only",
        "caught_by": "cyclic C24 x C7 time channel",
        "missed_by": "any spatial-only tokenizer"},
    "off_grid": {
        "perturbs": "orientation relative to the local street grid",
        "caught_by": "angular channel AND phi_loc, jointly",
        "missed_by": "a rotation-INVARIANT model (all invariants preserved)"},
    "teleport": {
        "perturbs": "one step becomes a jump",
        "caught_by": "radial channel (large r)",
        "missed_by": None},
    "off_grid_control": {
        "perturbs": "orientation by 90 deg -- STILL grid-aligned",
        # The control that makes the off_grid row interpretable. A 90 degree
        # rotation displaces absolute positions just as much as 45 degrees
        # does, but a square grid is invariant mod 90, so the trip remains
        # grid-aligned. Any detector keyed on position alone therefore scores
        # the same on both; only a detector genuinely sensitive to grid
        # orientation scores higher on off_grid than on this control. The
        # diagnostic is the DIFFERENCE, not either number.
        "caught_by": "nothing, if the detector is orientation-aware",
        "missed_by": "an orientation-aware model (that is the point)"},
    "unseen_area": {
        "perturbs": "whole trip moved to a rarely-visited region",
        "caught_by": "phi_loc (absolute position)",
        "missed_by": "a purely relative/displacement model"},
}
TYPES = tuple(EXPECTED)


@dataclass(eq=False)     # arrays inside: identity comparison, not elementwise
class Injected:
    traj: Traj
    kind: str            # "normal" or one of TYPES
    severity: float      # 0 for normal; the knob value otherwise


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _xy(traj, ref):
    return lonlat_to_enu(traj.lonlat, ref)[0]


def _back(xy, ref, t, traj, ):
    return Traj(enu_to_lonlat(xy, ref), t, traj.uid, traj.label)


def _speed_mps(xy, t):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.maximum(np.diff(t) * 3600.0, 1e-6)
    return d / dt


# --------------------------------------------------------------------------
# the six injectors
# --------------------------------------------------------------------------
def inject_detour(traj, rng, ref, severity: float = 1.0):
    """Bulge the middle of the path sideways, holding both endpoints fixed.

    Start and end are exactly preserved, so anything keyed on origin and
    destination is blind to it; path length does grow, which is precisely why
    `trivial_scores` reports path length as a competing detector.
    """
    xy = _xy(traj, ref).copy()
    n = len(xy)
    if n < 5:
        return Traj(traj.lonlat.copy(), traj.t.copy(), traj.uid, traj.label)
    span = np.linalg.norm(xy[-1] - xy[0]) + 1e-9
    # unit normal to the straight line between the endpoints
    d = (xy[-1] - xy[0]) / span
    nrm = np.array([-d[1], d[0]])
    amp = severity * 0.35 * span * rng.choice([-1.0, 1.0])
    bump = np.sin(np.pi * np.linspace(0, 1, n))[:, None] * nrm[None, :] * amp
    return _back(xy + bump, ref, traj.t.copy(), traj)


def inject_kinematic(traj, rng, ref, severity: float = 1.0):
    """Re-time the same path: identical geometry, altered speed profile.

    The visited points are unchanged, so every purely spatial statistic
    (length, shape, bounding box, endpoints) is preserved exactly; only the
    timestamps move.
    """
    t = traj.t.copy()
    n = len(t)
    if n < 4:
        return Traj(traj.lonlat.copy(), t, traj.uid, traj.label)
    dur = t[-1] - t[0]
    u = np.linspace(0, 1, n)
    # monotone warp: accelerate through one half, crawl through the other
    k = 1.0 + 2.0 * severity
    w = u ** k if rng.random() < 0.5 else 1.0 - (1.0 - u) ** k
    return Traj(traj.lonlat.copy(), t[0] + w * dur, traj.uid, traj.label)


def inject_wrong_time(traj, rng, ref, severity: float = 1.0):
    """Shift the trip to an unusual hour, geometry untouched.

    Severity 1 pushes it to the small hours; the shift is in whole hours so
    the day-of-week structure is disturbed too when it crosses midnight.
    """
    h0 = float(traj.t[0] % 24.0)
    target = 3.0 + rng.uniform(-1.5, 1.5)          # dead of night
    shift = (target - h0) * severity
    return Traj(traj.lonlat.copy(), traj.t + shift, traj.uid, traj.label)


def inject_off_grid(traj, rng, ref, severity: float = 1.0, angle_deg=None):
    """Rotate the trip about its own centroid, against the local street grid.

    Every rotation-invariant quantity is preserved **exactly**: step lengths,
    angles between consecutive steps, duration, speed profile, path length,
    radius of gyration. What changes is the orientation of the trip relative
    to the map. A rotation-invariant representation therefore cannot register
    this at all -- that is the prediction, and the experiment tests it.

    45 degrees is maximally off a square grid (which is invariant mod 90).
    """
    ang = np.deg2rad(45.0 * severity if angle_deg is None else angle_deg)
    xy = _xy(traj, ref)
    c = xy.mean(0)
    return _back((xy - c) @ rot2(ang).T + c, ref, traj.t.copy(), traj)


def inject_teleport(traj, rng, ref, severity: float = 1.0):
    """Displace a contiguous tail segment, creating one impossible step."""
    xy = _xy(traj, ref).copy()
    n = len(xy)
    if n < 6:
        return Traj(traj.lonlat.copy(), traj.t.copy(), traj.uid, traj.label)
    i = int(rng.integers(n // 4, 3 * n // 4))
    extent = np.ptp(xy, axis=0).mean() + 1.0
    jump = rng.normal(size=2)
    jump = jump / (np.linalg.norm(jump) + 1e-9) * severity * 0.5 * extent
    xy[i:] += jump
    return _back(xy, ref, traj.t.copy(), traj)


def inject_unseen_area(traj, rng, ref, severity: float = 1.0, grid=None):
    """Translate the whole trip into a rarely-visited part of the city.

    Shape, duration and every relative quantity are preserved exactly; only
    the absolute position changes. A displacement-only model is blind to this
    by construction -- the other half of the `off_grid` prediction.

    `grid` is the occupancy model from `fit_occupancy`; without it the trip is
    simply pushed far from the corpus centroid.
    """
    xy = _xy(traj, ref)
    c = xy.mean(0)
    if grid is None:
        extent = np.ptp(xy, axis=0).mean() + 1.0
        v = rng.normal(size=2)
        target = c + v / (np.linalg.norm(v) + 1e-9) * severity * 8.0 * extent
    else:
        target = sample_low_density(grid, rng, severity)
    return _back(xy - c + target, ref, traj.t.copy(), traj)


def inject_off_grid_control(traj, rng, ref, severity: float = 1.0):
    """90-degree rotation: same positional displacement, grid alignment kept."""
    return inject_off_grid(traj, rng, ref, severity=1.0, angle_deg=90.0)


INJECTORS = {"detour": inject_detour, "kinematic": inject_kinematic,
             "wrong_time": inject_wrong_time, "off_grid": inject_off_grid,
             "off_grid_control": inject_off_grid_control,
             "teleport": inject_teleport, "unseen_area": inject_unseen_area}


# --------------------------------------------------------------------------
# occupancy model, so "rarely visited" means something measured
# --------------------------------------------------------------------------
def fit_occupancy(trajs, ref, cell_m: float = 500.0):
    ll = np.concatenate([t.lonlat for t in trajs])
    xy = lonlat_to_enu(ll, ref)[0]
    keys = np.floor(xy / cell_m).astype(int)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return {"cells": uniq, "counts": counts, "cell_m": cell_m,
            "lo": keys.min(0), "hi": keys.max(0)}


def sample_low_density(grid, rng, severity: float = 1.0):
    """A point in a cell from the lower tail of the visit distribution.

    Severity interpolates from the median cell (0) to the least-visited
    occupied cell (1), so the anomaly gets harder to spot as severity falls.
    """
    order = np.argsort(grid["counts"])
    q = int(np.clip((1.0 - severity) * 0.5, 0.0, 0.99) * (len(order) - 1))
    cell = grid["cells"][order[q]]
    return (cell + rng.random(2)) * grid["cell_m"]


# --------------------------------------------------------------------------
# building an evaluation set
# --------------------------------------------------------------------------
def build_anomaly_set(trajs, types=TYPES, rate: float = 0.1,
                      severity: float = 1.0, ref=None, seed: int = 0,
                      occupancy_cell_m: float = 500.0, verbose: bool = True):
    """Return (items, ref). A fraction `rate` of trajectories is replaced by an
    injected copy of itself, cycling through `types` so the classes stay
    balanced. Normal trajectories are passed through untouched.
    """
    rng = np.random.default_rng(seed)
    if ref is None:
        ref = np.concatenate([t.lonlat for t in trajs]).mean(0)
    grid = fit_occupancy(trajs, ref, occupancy_cell_m)

    n_bad = int(rate * len(trajs))
    idx = rng.permutation(len(trajs))[:n_bad]
    assign = {int(i): types[k % len(types)] for k, i in enumerate(idx)}

    items = []
    for i, tr in enumerate(trajs):
        kind = assign.get(i)
        if kind is None:
            items.append(Injected(tr, "normal", 0.0))
            continue
        fn = INJECTORS[kind]
        kw = {"grid": grid} if kind == "unseen_area" else {}
        items.append(Injected(fn(tr, rng, ref, severity, **kw), kind, severity))

    if verbose:
        from collections import Counter
        c = Counter(it.kind for it in items)
        print(f"anomaly set: {len(items)} trajectories, "
              f"{sum(v for k, v in c.items() if k != 'normal')} injected  "
              + str({k: v for k, v in sorted(c.items()) if k != 'normal'}))
    return items, ref


# --------------------------------------------------------------------------
# the guard: could a one-line statistic have found these?
# --------------------------------------------------------------------------
def trivial_scores(items, ref):
    """Univariate detectors any reviewer would try first.

    Each returns a per-trajectory |z| against the pooled distribution. If one
    of these matches the model, the model has demonstrated nothing, so these
    are reported next to it rather than left out.
    """
    feats = {"path_length_m": [], "mean_speed_mps": [], "duration_h": [],
             "bbox_area_m2": [], "start_hour": []}
    for it in items:
        xy = lonlat_to_enu(it.traj.lonlat, ref)[0]
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        sp = _speed_mps(xy, it.traj.t)
        ext = np.ptp(xy, axis=0)
        feats["path_length_m"].append(float(d.sum()))
        feats["mean_speed_mps"].append(float(np.mean(sp)) if len(sp) else 0.0)
        feats["duration_h"].append(float(it.traj.t[-1] - it.traj.t[0]))
        feats["bbox_area_m2"].append(float(ext[0] * ext[1]))
        feats["start_hour"].append(float(it.traj.t[0] % 24.0))
    out = {}
    for k, v in feats.items():
        v = np.asarray(v, dtype=float)
        if k == "start_hour":      # circular: distance from the corpus mode
            ang = 2 * np.pi * v / 24.0
            mu = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
            out[k] = np.abs(np.angle(np.exp(1j * (ang - mu))))
        else:
            # Robust z on a log scale. A plain (v - mean)/std breaks on heavy
            # tails: a re-timed trip can reach absurd step speeds, those few
            # values drag the mean far above every normal trip, and the
            # moderately anomalous trips then sit *closer* to the mean than
            # the normal ones -- the detector inverts (AUROC ~0.02 measured on
            # GeoLife where a sane mean-speed check is ~1.0). Median/MAD on
            # log1p is immune to that.
            lv = np.log1p(np.maximum(v, 0.0))
            med = np.median(lv)
            mad = 1.4826 * np.median(np.abs(lv - med)) + 1e-9
            out[k] = np.abs(lv - med) / mad
    return out
