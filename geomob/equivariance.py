"""Equivariance auditing.

Three levels, and it matters which one a claim is made at:

  1. `encoder_equivariance_error` -- is the *encoder* a GEO?
  2. `model_equivariance_error`   -- is the *trained model* equivariant?
     GPE's embedding is equivariant but the LSTM stacked on top is not, so
     their cross-city transfer is approximate for a reason their paper does
     not identify. This function localises that.
  3. `explanation_equivariance_error` -- do *attributions* transform
     correctly? (Crabbe & van der Schaar, NeurIPS 2023.) No non-GEO
     competitor can match this by construction, and it is the experiment the
     draft cites but never runs.

Baselines are given every advantage: where no analytic rho exists we fit the
best orthogonal one by Procrustes on a held-out split and report the residual.
"""
from __future__ import annotations

import numpy as np


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """Globally normalised relative error.

    Normalising per-sample would divide by near-zero norms (a grid cell at the
    origin, a code that happens to sit near zero) and report meaningless
    magnitudes; the denominator is therefore the RMS norm over the batch.
    """
    num = np.linalg.norm(a - b, axis=-1)
    den = np.sqrt(np.mean(np.linalg.norm(b, axis=-1) ** 2)) + 1e-12
    return float(np.mean(num) / den)


def fit_orthogonal_rho(E_src: np.ndarray, E_dst: np.ndarray) -> np.ndarray:
    """Best orthogonal R with E_src @ R.T ~= E_dst (Procrustes)."""
    M = E_dst.T @ E_src
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def encoder_equivariance_error(encoder, lonlat, action, rho=None,
                               fit_split: float = 0.5, seed: int = 0):
    """Relative error of enc(g.l) vs rho(g) enc(l).

    `action(lonlat) -> lonlat'` applies the group element.
    `rho` is the analytic representation, or None to fit the best orthogonal
    one on `fit_split` of the points and evaluate on the rest.
    """
    E0 = encoder.encode(lonlat)
    E1 = encoder.encode(action(lonlat))
    if rho is not None:
        return {"error": _rel(E1, E0 @ np.asarray(rho).T), "rho": "analytic"}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(E0))
    k = max(int(fit_split * len(idx)), 1)
    R = fit_orthogonal_rho(E0[idx[:k]], E1[idx[:k]])
    te = idx[k:] if len(idx) > k else idx[:k]
    return {"error": _rel(E1[te], E0[te] @ R.T), "rho": "best-fit orthogonal"}


def rotation_equivariance_error(disp_encoder, dxy, thetas=None, seed: int = 0):
    """SO(2) audit of the displacement channel (exact by Sec. 2.3)."""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(-np.pi, np.pi, 16) if thetas is None else thetas
    errs = []
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        E0 = disp_encoder.encode_disp(dxy)
        E1 = disp_encoder.encode_disp(dxy @ R.T)
        errs.append(_rel(E1, E0 @ disp_encoder.rho_rotation(th).T))
    return float(np.mean(errs))


def time_equivariance_error(time_encoder, t_hours, taus=(1.0, 5.0, 13.0)):
    errs = []
    for tau in taus:
        E0 = time_encoder.encode(t=t_hours)
        E1 = time_encoder.encode(t=t_hours + tau)
        errs.append(_rel(E1, E0 @ time_encoder.shift_matrix(tau).T))
    return float(np.mean(errs))


def model_equivariance_error(predict_fn, trajs, theta_samples=8, seed: int = 0,
                             equivariant_output: bool = True):
    """E_theta || pred(R.T) - R pred(T) ||, normalised.

    `predict_fn(list_of_xy) -> (B, 2)` displacement predictions.
    Set equivariant_output=False for invariant outputs (e.g. embeddings used
    only for ranking), in which case the target is pred(T) itself.
    """
    rng = np.random.default_rng(seed)
    errs = []
    base = predict_fn(trajs)
    for _ in range(theta_samples):
        th = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        rot = [t @ R.T for t in trajs]
        got = predict_fn(rot)
        want = base @ R.T if equivariant_output else base
        errs.append(_rel(got, want))
    return float(np.mean(errs))


def explanation_equivariance_error(attrib_fn, trajs, theta_samples=8, seed=0):
    """Attributions live on input positions, so they must rotate with them."""
    rng = np.random.default_rng(seed)
    errs = []
    A0 = [attrib_fn(t) for t in trajs]
    for _ in range(theta_samples):
        th = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        A1 = [attrib_fn(t @ R.T) for t in trajs]
        errs += [_rel(a1, a0 @ R.T if a0.ndim == 2 and a0.shape[1] == 2 else a0)
                 for a0, a1 in zip(A0, A1)]
    return float(np.mean(errs))


# --------------------------------------------------------------------------
# stability / non-expansiveness
# --------------------------------------------------------------------------
def empirical_lipschitz(encoder, lonlat, deltas_m=(1.0, 5.0, 15.0), seed=0):
    """||enc(l+d) - enc(l)|| / ||d||, for d a random metric perturbation."""
    from .geo import enu_to_lonlat, lonlat_to_enu
    rng = np.random.default_rng(seed)
    xy, ref = lonlat_to_enu(lonlat)
    out = {}
    for d in deltas_m:
        u = rng.normal(size=(len(xy), 2))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        E0 = encoder.encode(lonlat)
        E1 = encoder.encode(enu_to_lonlat(xy + d * u, ref))
        out[d] = float(np.mean(np.linalg.norm(E1 - E0, axis=1)) / d)
    return out


def saturation_curve(encoder, lonlat, sigmas=(0, 1, 2, 5, 10, 15, 25, 50),
                     seed: int = 0):
    """Cosine similarity between clean and jittered codes vs GPS noise sigma.

    An encoder whose finest wavelength sits below the noise floor decorrelates
    -- the curve flattens at the level of two random codes. GPE's default
    epsilon puts its finest wavelength around 30-40 m, just above typical GPS
    error, and their own epsilon ablation (finer = worse) is the shadow of
    this effect.
    """
    from .geo import enu_to_lonlat, lonlat_to_enu
    rng = np.random.default_rng(seed)
    xy, ref = lonlat_to_enu(lonlat)
    E0 = encoder.encode(lonlat)
    E0n = E0 / (np.linalg.norm(E0, axis=1, keepdims=True) + 1e-12)
    out = {}
    for s in sigmas:
        E1 = encoder.encode(enu_to_lonlat(xy + rng.normal(0, s, xy.shape), ref))
        E1n = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + 1e-12)
        out[float(s)] = float(np.mean(np.sum(E0n * E1n, axis=1)))
    return out
