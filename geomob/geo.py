"""Geometry: local metric frames, and the group action of G = SE(2) x Theta.

Everything downstream depends on one distinction that the GPE paper never makes:

  * an encoder that is equivariant in *angular* (lon/lat) coordinates is only
    metrically consistent at a fixed latitude, because one degree of longitude
    subtends R*cos(lat) metres;
  * an encoder built on displacements in a *local metric frame* (ENU) is
    consistent everywhere.

Hypothesis H6 in the proposal is exactly the empirical version of this remark,
and `reproject_to_latitude` below is the instrument that tests it.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8  # mean Earth radius, metres


# --------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------
def lonlat_to_enu(lonlat_deg: np.ndarray, ref_deg: np.ndarray | None = None):
    """Local East-North-Up tangent plane around `ref_deg` (degrees).

    Returns (xy_metres, ref_deg). Accurate to well under a metre over the
    ~50 km extent of a city, which is the regime every dataset here lives in.
    """
    lonlat_deg = np.asarray(lonlat_deg, dtype=np.float64)
    if ref_deg is None:
        ref_deg = lonlat_deg.mean(axis=0)
    ref_deg = np.asarray(ref_deg, dtype=np.float64)
    lat0 = np.deg2rad(ref_deg[1])
    d = np.deg2rad(lonlat_deg - ref_deg)
    x = R_EARTH * d[..., 0] * np.cos(lat0)
    y = R_EARTH * d[..., 1]
    return np.stack([x, y], axis=-1), ref_deg


def enu_to_lonlat(xy: np.ndarray, ref_deg: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    ref_deg = np.asarray(ref_deg, dtype=np.float64)
    lat0 = np.deg2rad(ref_deg[1])
    lon = ref_deg[0] + np.rad2deg(xy[..., 0] / (R_EARTH * np.cos(lat0)))
    lat = ref_deg[1] + np.rad2deg(xy[..., 1] / R_EARTH)
    return np.stack([lon, lat], axis=-1)


def displacements(xy: np.ndarray) -> np.ndarray:
    """Delta x_i = l_i - l_{i-1}; translation-invariant by construction."""
    return np.diff(np.asarray(xy, dtype=np.float64), axis=0)


# --------------------------------------------------------------------------
# group actions
# --------------------------------------------------------------------------
def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def act_rotate(xy: np.ndarray, theta: float, about: np.ndarray | None = None) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    about = xy.mean(axis=0) if about is None else np.asarray(about, float)
    return (xy - about) @ rot2(theta).T + about


def act_translate(xy: np.ndarray, v) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64) + np.asarray(v, dtype=np.float64)


def act_timeshift(t_hours: np.ndarray, tau: float) -> np.ndarray:
    """Theta = C24 x C7 acts by shifting the clock; tau in hours."""
    return np.asarray(t_hours, dtype=np.float64) + tau


def act_se2(xy, t, theta=0.0, v=(0.0, 0.0), tau=0.0, about=None):
    """Full g = (R_theta, b, tau) acting on a trajectory in metric coordinates."""
    return act_translate(act_rotate(xy, theta, about), v), act_timeshift(t, tau)


def sample_group_element(rng, rot=True, trans_m=1000.0, tau_h=12.0):
    theta = rng.uniform(-np.pi, np.pi) if rot else 0.0
    v = rng.uniform(-trans_m, trans_m, size=2)
    tau = rng.uniform(-tau_h, tau_h)
    return theta, v, tau


# --------------------------------------------------------------------------
# the latitude stress test
# --------------------------------------------------------------------------
def reproject_to_latitude(lonlat_deg: np.ndarray, target_lat: float,
                          target_lon: float = 0.0) -> np.ndarray:
    """Move a trajectory to a new latitude *preserving its metric shape*.

    The trip is identical on the ground -- same metres, same angles, same
    duration -- only its (lon, lat) coordinates change. Any encoder that is
    metrically consistent must therefore be unaffected; any encoder that is
    equivariant only in angular coordinates (GPE, raw XY, angular grids) sees
    an east-west rescaling of cos(lat_src)/cos(lat_dst).

    Beijing/Porto/Rome all sit at 40-42 N, so GPE's own cross-city experiments
    never exercise this factor by more than ~3%. At 0 N vs 60 N it is ~2x.
    """
    xy, ref = lonlat_to_enu(lonlat_deg)
    return enu_to_lonlat(xy, np.array([target_lon, target_lat]))


def lon_scale_ratio(lat_src_deg: float, lat_dst_deg: float) -> float:
    """How much one metre of easting is stretched in longitude degrees."""
    return float(np.cos(np.deg2rad(lat_src_deg)) / np.cos(np.deg2rad(lat_dst_deg)))
