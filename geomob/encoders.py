"""Position encoders under one interface, so they can be swapped inside the
same backbone (the comparison GPE's Table 4 makes, plus the ones it omits).

Every encoder exposes:

    enc.dim                      -> int, output dimensionality
    enc.encode(lonlat, t=None)   -> (N, dim) float array
    enc.rho(v)                   -> (dim, dim) orthogonal matrix, or None

`rho` is the crux. Where an encoder is an exact GEO under translation by `v`,
it returns the analytic representation matrix rho(v) such that

    enc(l + v) = rho(v) @ enc(l).

Returning None means "no exact representation is claimed", and the audit in
`equivariance.py` then falls back to fitting the *best possible* orthogonal
rho on held-out data -- i.e. it gives unconstrained baselines the benefit of
the doubt and still measures a residual.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geo import R_EARTH, lonlat_to_enu


# ==========================================================================
# helpers
# ==========================================================================
def _sincos_block_rho(freqs: np.ndarray, shift: float) -> np.ndarray:
    """Block-diagonal rho for a code laid out as [sin(w x), cos(w x)] per w.

    A shift x -> x + s rotates each 2-d block by angle w*s. This is exactly
    GPE's Theorems 3 and 4 (equal distance / equal similarity) restated as a
    group representation: rho is orthogonal, so it preserves both Euclidean
    distance and cosine similarity. The theorems are corollaries of it.
    """
    blocks = []
    for w in freqs:
        a = w * shift
        c, s = np.cos(a), np.sin(a)
        # d/ds [sin(w(x+s)), cos(w(x+s))] rotation in the (sin, cos) basis
        blocks.append(np.array([[c, s], [-s, c]]))
    return _block_diag(blocks)


def _block_diag(blocks):
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    i = 0
    for b in blocks:
        k = b.shape[0]
        out[i:i + k, i:i + k] = b
        i += k
    return out


class Encoder:
    """Base class. Subclasses set self.dim and implement encode()."""

    name = "encoder"
    dim = 0

    def encode(self, lonlat: np.ndarray, t=None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def rho(self, v):  # noqa: D401
        """Analytic representation for a metric translation v (metres)."""
        return None

    def lipschitz(self) -> float:
        """Upper bound on ||d enc / d position|| in units of 1/metre.

        Relevant to the GENEO non-expansiveness condition and to the GPS-noise
        stress test: GPE's finest block at h=128 has a Lipschitz constant of
        order 1e6 per radian, i.e. ~0.16 rad of phase per metre, so a 15 m
        fix error fully decorrelates it.
        """
        return float("nan")


# ==========================================================================
# 1. GPE (Liu et al., KDD 2025) -- faithful reimplementation of Algorithm 1
# ==========================================================================
class GPE(Encoder):
    """Multi-base sinusoidal encoding of (lon, lat) in radians.

    Reimplements Algorithm 1: the tau-filter picks bases skipping exponents of
    earlier bases, the epsilon-cut bounds the index of each base.

    Read as a GEO: GPE(p + v) = rho(v) @ GPE(p) with rho block-diagonal
    orthogonal, exactly. It is therefore a *translation-equivariant GEO on the
    lon/lat torus* -- the paper's four properties (global, continuous, unique,
    dynamic) do not name this, but it is what drives their zero-shot results.
    Two things it is not: rotation-equivariant (lon and lat are encoded on
    separate axes) and metrically consistent across latitude.
    """

    name = "GPE"

    def __init__(self, h: int = 128, eps_over_2pi: float = 1e-6):
        assert h % 4 == 0, "GPE emits 4 numbers per frequency"
        self.h = h
        self.eps_over_2pi = eps_over_2pi
        self.freqs = self._build_frequencies(h, eps_over_2pi)
        self.dim = 4 * len(self.freqs)

    @staticmethod
    def _build_frequencies(h: int, eps_over_2pi: float) -> np.ndarray:
        target = h // 4
        w = [1.0]
        sieve = np.zeros(max(h, 4) + 1, dtype=bool)  # Lambda
        lam = 2
        max_i = int(np.ceil(-np.log(eps_over_2pi) / np.log(2)))  # loosest bound
        while len(w) < target and lam < len(sieve):
            while lam < len(sieve) and sieve[lam]:
                lam += 1
            if lam >= len(sieve):
                break
            # mark exponents of lam (tau-filter)
            p = lam
            while p < len(sieve):
                sieve[p] = True
                p *= lam
            # epsilon-cut: maximal index for this base
            i_max = int(np.ceil(-np.log(eps_over_2pi) / np.log(lam)))
            i_max = min(i_max, max_i)
            for i in range(1, i_max + 1):
                if len(w) >= target:
                    break
                w.append(float(lam) ** i)
            lam += 1
        while len(w) < target:  # degenerate small-h fallback
            w.append(w[-1] * 2.0)
        return np.array(w[:target], dtype=np.float64)

    def encode(self, lonlat: np.ndarray, t=None) -> np.ndarray:
        p = np.deg2rad(np.asarray(lonlat, dtype=np.float64))
        out = []
        for axis in (0, 1):
            a = np.outer(p[:, axis], self.freqs)  # (N, F)
            out.append(np.stack([np.sin(a), np.cos(a)], axis=-1).reshape(len(p), -1))
        return np.concatenate(out, axis=1)

    def rho(self, v, ref_lat_deg: float = 0.0):
        """v is a metric offset (metres, east/north) -> angular shift."""
        v = np.asarray(v, dtype=np.float64)
        dlon = v[0] / (R_EARTH * np.cos(np.deg2rad(ref_lat_deg)))
        dlat = v[1] / R_EARTH
        return _block_diag([_sincos_block_rho(self.freqs, dlon),
                            _sincos_block_rho(self.freqs, dlat)])

    def lipschitz(self) -> float:
        # phase rate per metre, summed in quadrature over blocks
        w_per_m = self.freqs / R_EARTH
        return float(np.sqrt(2.0 * np.sum(w_per_m ** 2)))

    def finest_wavelength_m(self) -> float:
        return float(2 * np.pi * R_EARTH / self.freqs.max())


# ==========================================================================
# 2. Space2Vec / grid-cell multi-scale encoder (Mai et al., ICLR 2020)
# ==========================================================================
class Space2Vec(Encoder):
    """Three unit directions at 120 degrees, geometric ladder of wavelengths.

    Exactly translation-equivariant in the metric plane; equivariant under
    rotations by multiples of 120 degrees only, not all of SO(2).
    """

    name = "Space2Vec"

    def __init__(self, n_scales: int = 8, lam_min: float = 50.0,
                 lam_max: float = 50_000.0, ref_deg=None):
        self.lams = lam_min * (lam_max / lam_min) ** (
            np.arange(n_scales) / max(n_scales - 1, 1))
        angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        self.dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (3,2)
        self.ref_deg = ref_deg
        self.dim = 6 * n_scales

    def encode(self, lonlat: np.ndarray, t=None) -> np.ndarray:
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        proj = xy @ self.dirs.T                       # (N, 3)
        a = 2 * np.pi * proj[:, :, None] / self.lams[None, None, :]  # (N,3,S)
        return np.stack([np.sin(a), np.cos(a)], -1).reshape(len(xy), -1)

    def rho(self, v):
        v = np.asarray(v, float)
        proj = self.dirs @ v
        # layout is (dir, scale, {sin, cos}) -> build rho in that order
        blocks = [_sincos_block_rho(np.array([2 * np.pi / lam]), proj[d])
                  for d in range(3) for lam in self.lams]
        return _block_diag(blocks)

    def lipschitz(self) -> float:
        return float(np.sqrt(2 * 3 * np.sum((2 * np.pi / self.lams) ** 2)))


# ==========================================================================
# 3. phi_loc, regime (i): uniform multi-scale lattice  -- exact, no adaptivity
# ==========================================================================
class TorusLattice(Encoder):
    """Section 2.5 regime (i), with one fix relative to the draft.

    A *single* lambda aliases every lambda, so the code is not unique and the
    draft's own uniqueness requirement fails. Using a geometric ladder whose
    longest period exceeds the domain diameter restores uniqueness while
    keeping exact equivariance: a direct sum of equivariant blocks is
    equivariant. (This is precisely the trick GPE uses, with a fundamental
    period of one globe.)
    """

    name = "TorusLattice"

    def __init__(self, lams=(200.0, 1_000.0, 5_000.0, 25_000.0, 125_000.0),
                 ref_deg=None):
        self.lams = np.asarray(lams, dtype=np.float64)
        self.ref_deg = ref_deg
        self.dim = 4 * len(self.lams)

    def encode(self, lonlat: np.ndarray, t=None) -> np.ndarray:
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        out = []
        for axis in (0, 1):
            a = 2 * np.pi * xy[:, axis][:, None] / self.lams[None, :]
            out.append(np.stack([np.sin(a), np.cos(a)], -1).reshape(len(xy), -1))
        return np.concatenate(out, axis=1)

    def rho(self, v):
        v = np.asarray(v, float)
        w = 2 * np.pi / self.lams
        return _block_diag([_sincos_block_rho(w, v[0]),
                            _sincos_block_rho(w, v[1])])

    def lipschitz(self) -> float:
        return float(np.sqrt(2 * np.sum((2 * np.pi / self.lams) ** 2)))

    def periods(self) -> np.ndarray:
        """Translations under which the discrete code is *identical*.

        Used by the fixed Experiment C: an equivariant code does not leave
        tokens fixed under an arbitrary v, only under v in the lattice.
        """
        return self.lams


# ==========================================================================
# 4. phi_loc, regime (ii): separable warp -- gauge-equivariant to first order
# ==========================================================================
class SeparableWarp(Encoder):
    """Axis-separable scale fields L_x(x), L_y(y), integrated along their own
    axis, then wrapped on the torus.

    Proposition 2's path-dependence disappears (a 1-d integral has no path to
    choose). Proposition 3 does not: eta is not affine, so only

        phi(l+v) = rho_l(v) phi(l) + O(||v||^2),  rho_l(v) = J_eta(l) v

    holds. `defect_curve` measures that O(||v||^2) term, which the proposal
    asks for rather than assumes.
    """

    name = "SeparableWarp"

    def __init__(self, n_grid: int = 512, lam_min: float = 100.0,
                 lam_max: float = 5_000.0, ref_deg=None, smooth: float = 12.0):
        self.n_grid, self.lam_min, self.lam_max = n_grid, lam_min, lam_max
        self.ref_deg, self.smooth = ref_deg, smooth
        self.dim = 4
        self._fitted = False

    def fit(self, lonlat_train: np.ndarray):
        xy, ref = lonlat_to_enu(lonlat_train, self.ref_deg)
        self.ref_deg = ref
        self.grids, self.etas, self.Ls, self.dEtas = [], [], [], []
        for axis in (0, 1):
            lo, hi = np.percentile(xy[:, axis], [0.1, 99.9])
            pad = 0.25 * (hi - lo) + 1.0
            g = np.linspace(lo - pad, hi + pad, self.n_grid)
            hist, _ = np.histogram(xy[:, axis], bins=np.r_[g, g[-1] + (g[1] - g[0])])
            hist = _gauss1d(hist.astype(float), self.smooth) + 1e-9
            dens = hist / hist.max()
            L = np.clip(self.lam_min / np.sqrt(dens), self.lam_min, self.lam_max)
            # trapezoid rule, so eta is a consistent antiderivative of 1/L
            inv = 1.0 / L
            eta = np.concatenate(
                [[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(g))])
            self.grids.append(g)
            self.etas.append(eta)
            self.Ls.append(L)
            # The gauge J must be the derivative of the *interpolant* actually
            # used by encode(), not of the underlying continuous field.
            # Mixing the two leaves an O(||v||) discretisation residual that
            # would masquerade as a failure of the first-order prediction (5).
            self.dEtas.append(np.gradient(eta, g))
        self._fitted = True
        return self

    def _eta(self, xy):
        return np.stack([np.interp(xy[:, a], self.grids[a], self.etas[a])
                         for a in (0, 1)], axis=1)

    def jacobian(self, lonlat: np.ndarray) -> np.ndarray:
        """J_eta(l) = diag(1/L_x, 1/L_y): the base-point-dependent gauge."""
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        return np.stack([np.interp(xy[:, a], self.grids[a], self.dEtas[a])
                         for a in (0, 1)], axis=1)

    def encode(self, lonlat: np.ndarray, t=None) -> np.ndarray:
        assert self._fitted, "call .fit(train_lonlat) first"
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        y = self._eta(xy) % 1.0
        a = 2 * np.pi * y
        return np.concatenate([np.sin(a[:, :1]), np.cos(a[:, :1]),
                               np.sin(a[:, 1:]), np.cos(a[:, 1:])], axis=1)

    def rho_gauge(self, lonlat_base: np.ndarray, v) -> np.ndarray:
        """Position-dependent rho_l(v) predicted by Eq. (5), per point."""
        J = self.jacobian(lonlat_base)          # (N, 2)
        v = np.asarray(v, float)
        return 2 * np.pi * J * v[None, :]       # phase advance per axis


def _gauss1d(x, sigma):
    if sigma <= 0:
        return x
    r = int(4 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    # np.convolve(mode='same') returns max(len(x), len(k)); pad instead so the
    # output always matches the input grid, whatever the smoothing width.
    xp = np.pad(x, r, mode="edge")
    return np.convolve(xp, k, mode="valid")[:len(x)]


# ==========================================================================
# 5. phi_loc, regime (iii): patchwork of local lattices -- exact, adaptive
# ==========================================================================
@dataclass
class PatchworkLattice(Encoder):
    """Tiles with a locally fitted, piecewise-constant lambda_k.

    Equivariance is exact *within* a tile. Crossing steps are handled either
    by 'hard' assignment (tile of the current point) or 'dual' (both tiles'
    codes concatenated, the proposal's soft-blend stand-in).
    """

    tile_m: float = 8_000.0
    lam_min: float = 100.0
    lam_max: float = 5_000.0
    n_harm: int = 2
    ref_deg: np.ndarray | None = None
    name: str = "PatchworkLattice"
    dim: int = field(init=False, default=0)

    def __post_init__(self):
        self.dim = 4 * self.n_harm
        self._fitted = False

    def fit(self, lonlat_train: np.ndarray):
        xy, ref = lonlat_to_enu(lonlat_train, self.ref_deg)
        self.ref_deg = ref
        keys = np.floor(xy / self.tile_m).astype(int)
        uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                                      return_counts=True)
        # lambda_k from local density, in the spirit of density-equalizing maps
        dens = counts / (self.tile_m ** 2)
        lam = np.clip(1.0 / np.sqrt(dens + 1e-12), self.lam_min, self.lam_max)
        self.lam_by_tile = {tuple(k): float(l) for k, l in zip(uniq, lam)}
        self.lam_default = float(np.median(lam))
        self._fitted = True
        return self

    def _lam(self, keys):
        return np.array([self.lam_by_tile.get(tuple(k), self.lam_default)
                         for k in keys])

    def encode(self, lonlat: np.ndarray, t=None) -> np.ndarray:
        assert self._fitted, "call .fit(train_lonlat) first"
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        keys = np.floor(xy / self.tile_m).astype(int)
        lam = self._lam(keys)
        out = []
        for m in range(1, self.n_harm + 1):
            for axis in (0, 1):
                a = 2 * np.pi * m * xy[:, axis] / lam
                out += [np.sin(a), np.cos(a)]
        return np.stack(out, axis=1)

    def tile_ids(self, lonlat: np.ndarray) -> np.ndarray:
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        return np.floor(xy / self.tile_m).astype(int)


# ==========================================================================
# 6. The displacement GEO: invariant radial channel + equivariant angular one
# ==========================================================================
class DisplacementGEO(Encoder):
    """phi(dx) = [ PE(r) | g(r) * u_m(psi) ].

    Two deviations from the draft, both deliberate:

    * `gate=True` multiplies the angular channel by a smooth invariant g(r)
      with g(0)=0. Raw u(psi) is discontinuous as r -> 0, and stay points and
      GPS dwell produce many near-zero displacements. Gating restores
      continuity (a property GPE lists and will be checked against) while
      keeping SO(2)-equivariance exact, since g(r) is invariant.
    * higher harmonics u_m, m = 1..M, transforming under rho_m, are available
      as the dictionary ablation.
    """

    name = "DisplacementGEO"

    def __init__(self, n_scales: int = 8, lam_min: float = 10.0,
                 lam_max: float = 20_000.0, harmonics=(1,), gate: bool = True,
                 gate_r0: float = 5.0):
        self.lams = lam_min * (lam_max / lam_min) ** (
            np.arange(n_scales) / max(n_scales - 1, 1))
        self.harmonics = tuple(harmonics)
        self.gate, self.gate_r0 = gate, gate_r0
        self.dim = 2 * n_scales + 2 * len(self.harmonics)
        self.n_inv = 2 * n_scales

    def encode_disp(self, dxy: np.ndarray) -> np.ndarray:
        dxy = np.asarray(dxy, dtype=np.float64)
        r = np.linalg.norm(dxy, axis=1)
        psi = np.arctan2(dxy[:, 1], dxy[:, 0])
        a = 2 * np.pi * r[:, None] / self.lams[None, :]
        # invariant; explicit width so n_scales = 0 (angular-only) works
        pe = np.stack([np.sin(a), np.cos(a)], -1).reshape(len(r), 2 * len(self.lams))
        g = (r / np.hypot(r, self.gate_r0)) if self.gate else np.ones_like(r)
        ang = [np.stack([np.cos(m * psi), np.sin(m * psi)], 1) * g[:, None]
               for m in self.harmonics]
        return np.concatenate([pe] + ang, axis=1)

    def encode(self, lonlat: np.ndarray, t=None) -> np.ndarray:
        xy, _ = lonlat_to_enu(lonlat)
        return self.encode_disp(np.diff(xy, axis=0))

    def rho_rotation(self, theta: float) -> np.ndarray:
        """Exact SO(2) representation: identity on PE(r), rho_m on each u_m."""
        blocks = [np.eye(self.n_inv)]
        for m in self.harmonics:
            c, s = np.cos(m * theta), np.sin(m * theta)
            blocks.append(np.array([[c, -s], [s, c]]))
        return _block_diag(blocks)

    def rho(self, v):
        return np.eye(self.dim)  # displacements are translation-invariant


# ==========================================================================
# 6b. Speed / dt channel  --  invariant under the whole of SE(2)
# ==========================================================================
class SpeedGEO(Encoder):
    """Multi-scale code for step duration and speed.

    Added because the anomaly experiment demonstrated a gap rather than
    because the theory predicted one: phi(dx) encodes the displacement
    *magnitude* and phi_time encodes absolute time-of-day, so a trajectory
    re-timed along an unchanged path is invisible to both. Speed is the
    missing quantity, and it is a scalar, hence invariant under rotation and
    translation -- so this channel is a GEO with rho = identity and costs the
    construction nothing in symmetry.

    Encodes log1p of dt (seconds) and of speed (m/s) on a geometric ladder, so
    that walking, cycling and driving separate at different scales.
    """

    name = "SpeedGEO"

    def __init__(self, n_scales: int = 6, lo: float = 0.05, hi: float = 60.0):
        self.lams = lo * (hi / lo) ** (np.arange(n_scales) /
                                       max(n_scales - 1, 1))
        self.dim = 4 * n_scales

    def encode_speed(self, dxy: np.ndarray, dt_h: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(np.asarray(dxy, dtype=np.float64), axis=1)
        dt = np.maximum(np.asarray(dt_h, dtype=np.float64) * 3600.0, 1e-6)
        out = []
        for x in (np.log1p(r / dt), np.log1p(dt)):
            a = 2 * np.pi * x[:, None] / self.lams[None, :]
            out.append(np.stack([np.sin(a), np.cos(a)], -1).reshape(len(x), -1))
        return np.concatenate(out, axis=1)

    def rho(self, v):
        return np.eye(self.dim)          # invariant



class CyclicTimeGENEO(Encoder):
    """Harmonic code on C24 x C7 plus an optional circulant filter.

    The filter is constrained to operator norm <= 1 in the Fourier domain,
    which is what upgrades the operator from a GEO to a GENEO. `shift_matrix`
    gives the exact representation of a tau-hour shift, so time-equivariance
    is auditable with the same machinery as space.
    """

    name = "CyclicTimeGENEO"

    def __init__(self, n_hour_harm: int = 4, n_dow_harm: int = 3,
                 non_expansive: bool = True, seed: int = 0):
        self.mh = np.arange(1, n_hour_harm + 1)
        self.md = np.arange(1, n_dow_harm + 1)
        self.dim = 2 * (n_hour_harm + n_dow_harm)
        self.non_expansive = non_expansive
        rng = np.random.default_rng(seed)
        self.kappa = self._make_circulant(rng) if non_expansive else None

    def _make_circulant(self, rng):
        k = rng.normal(size=24)
        f = np.fft.fft(k)
        f /= max(np.abs(f).max(), 1e-12)   # ||kappa||_op <= 1
        return np.real(np.fft.ifft(f))

    def encode(self, lonlat=None, t=None) -> np.ndarray:
        """t in hours since an epoch (float)."""
        t = np.asarray(t, dtype=np.float64)
        ah = 2 * np.pi * np.outer(t % 24.0, self.mh) / 24.0
        ad = 2 * np.pi * np.outer((t // 24.0) % 7.0, self.md) / 7.0
        return np.concatenate([np.sin(ah), np.cos(ah),
                               np.sin(ad), np.cos(ad)], axis=1)

    def shift_matrix(self, tau_hours: float) -> np.ndarray:
        """rho(tau): block rotations by 2*pi*m*tau/24 and .../7 (in days)."""
        blocks_h = [np.array([[np.cos(2 * np.pi * m * tau_hours / 24),
                               np.sin(2 * np.pi * m * tau_hours / 24)],
                              [-np.sin(2 * np.pi * m * tau_hours / 24),
                               np.cos(2 * np.pi * m * tau_hours / 24)]])
                    for m in self.mh]
        # layout is [sin_h..., cos_h..., sin_d..., cos_d...] -> permute
        return _interleaved_to_grouped_rho(blocks_h, self.mh, self.md,
                                           tau_hours)

    def op_norm(self) -> float:
        if self.kappa is None:
            return float("inf")
        return float(np.abs(np.fft.fft(self.kappa)).max())


def _interleaved_to_grouped_rho(_unused, mh, md, tau_hours):
    """rho for the layout [sin(m_h t)..., cos(m_h t)..., sin(m_d)..., cos(m_d)...]."""
    def two_block(ms, period, tau):
        k = len(ms)
        A = np.zeros((2 * k, 2 * k))
        for j, m in enumerate(ms):
            a = 2 * np.pi * m * tau / period
            A[j, j] = np.cos(a)
            A[j, k + j] = np.sin(a)
            A[k + j, j] = -np.sin(a)
            A[k + j, k + j] = np.cos(a)
        return A
    return _block_diag([two_block(mh, 24.0, tau_hours),
                        two_block(md, 7.0, tau_hours / 24.0)])


# ==========================================================================
# 8. Non-equivariant baselines (the UniTE / GPE comparison set)
# ==========================================================================
class RawXY(Encoder):
    """Raw coordinates, optionally min-max normalised (UniTE's 'FC' embedder,
    GPE's 'XY'). The space-shifting (SS) protocol of GPE's global experiments
    is `normalize=True` fitted on the training city."""

    name = "RawXY"

    def __init__(self, normalize: bool = True):
        self.normalize, self.dim, self._box = normalize, 2, None

    def fit(self, lonlat_train):
        self._box = (np.min(lonlat_train, 0), np.max(lonlat_train, 0))
        return self

    def encode(self, lonlat, t=None):
        p = np.asarray(lonlat, float)
        if self.normalize and self._box is not None:
            lo, hi = self._box
            p = (p - lo) / np.maximum(hi - lo, 1e-9)
        return p


class GridEmbed(Encoder):
    """Discretising grid baseline (GPE's 'Grid'): a one-hot-free stand-in that
    returns the grid cell centre. Not continuous, not unique -- which is the
    point of including it."""

    name = "Grid"

    def __init__(self, cell_m: float = 150.0, ref_deg=None):
        self.cell_m, self.ref_deg, self.dim = cell_m, ref_deg, 2

    def encode(self, lonlat, t=None):
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        return np.floor(xy / self.cell_m) * self.cell_m / 1000.0


class LearnedFourier(Encoder):
    """Random-Fourier-feature encoder with random frequencies and phases
    (GPE's 'TriW', UniTE's Fourier embedder).

    Note, because it is easy to get wrong: this IS exactly
    translation-equivariant, and the audit confirms it at ~1e-12. Each output
    pair (sin(a_j), cos(a_j)) shares a frequency and a phase, so a translation
    rotates the pair; the random phase is absorbed and does not break it. What
    it lacks is rotation equivariance and any control over its scale
    spectrum -- the frequencies are drawn, not chosen, so nothing ties its
    resolution to the geography. Do not claim it fails translation.""" 

    name = "LearnedFourier"

    def __init__(self, dim: int = 128, scale: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=1.0 / scale, size=(2, dim // 2))
        self.b = rng.uniform(0, 2 * np.pi, size=dim // 2)
        self.dim = dim

    def encode(self, lonlat, t=None):
        p = np.deg2rad(np.asarray(lonlat, float))
        a = p @ self.W + self.b
        return np.concatenate([np.sin(a), np.cos(a)], axis=1)


ENCODER_REGISTRY = {
    "speed": SpeedGEO,
    "gpe": GPE,
    "space2vec": Space2Vec,
    "torus": TorusLattice,
    "warp": SeparableWarp,
    "patchwork": PatchworkLattice,
    "disp_geo": DisplacementGEO,
    "xy": RawXY,
    "grid": GridEmbed,
    "fourier": LearnedFourier,
}
