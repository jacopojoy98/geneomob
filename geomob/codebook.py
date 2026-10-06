"""Discrete codebooks, and the token-identity test of Experiment C.

Correction to the draft's prediction. Under a translation v, an equivariant
code does not stay put -- it moves by rho(v). So a k-means index on phi_loc is
*not* preserved for an arbitrary v inside a tile, and the draft's "regime
(i)/(iii) should score at or very near 100%" is false as stated. Two repairs,
both implemented here, and both still falsifiable:

  A. `lattice_period_test`  -- restrict v to the lattice, v in lambda Z^2.
     Then rho(v) = identity and the index is preserved *exactly*, for every
     codebook, by construction. This is the sharp prediction.

  B. `equivariant_codebook_test` -- use a codebook whose cells rho(v)
     permutes: a uniform grid on the torus phase. Then the prediction is not
     "the index is unchanged" but "the index changes by the predicted
     permutation", which is a stronger and more informative claim, and one a
     naive k-means on raw (x, y) cannot satisfy at all.

Regime (ii) is expected to fail both in the amount predicted by Eq. (5), which
`gauge_defect_curve` measures against its own O(||v||^2) prediction.
"""
from __future__ import annotations

import numpy as np

from .geo import enu_to_lonlat, lonlat_to_enu


# --------------------------------------------------------------------------
# codebooks
# --------------------------------------------------------------------------
class KMeansCodebook:
    """Plain k-means; no adversarial or perceptual loss is needed here."""

    def __init__(self, n_codes: int = 512, iters: int = 40, seed: int = 0):
        self.n_codes, self.iters, self.seed = n_codes, iters, seed

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        C = X[rng.choice(len(X), size=min(self.n_codes, len(X)), replace=False)]
        for _ in range(self.iters):
            idx = self._assign(X, C)
            for j in range(len(C)):
                m = idx == j
                if m.any():
                    C[j] = X[m].mean(0)
        self.C = C
        return self

    @staticmethod
    def _assign(X, C):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        return np.argmin(d, axis=1)

    def encode(self, X):
        return self._assign(np.asarray(X, dtype=np.float64), self.C)

    def perplexity(self, idx):
        c = np.bincount(idx, minlength=len(self.C)).astype(float)
        p = c / c.sum()
        return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


class EquivariantTorusCodebook:
    """A codebook on which the group acts by a known permutation.

    Uniform bins on the torus phase of one scale lambda. Translation shifts
    the phase by a constant, so it permutes the codebook cells rather than
    scrambling them: rho(v) descends from the continuous code to the discrete
    tokens. This is what makes a *discrete* tokenizer equivariant, which a
    k-means codebook fitted on an equivariant code is not -- the underlying
    vectors move correctly, but the cell boundaries are arbitrary, so the
    induced index map is not a group action.

    Two regimes, and the distinction is the whole point:

      on-lattice   v a multiple of lambda/bins  -> the permutation is EXACT,
                   agreement must be 1.0 to floating point;
      off-lattice  otherwise                    -> the nearest-bin prediction
                   is wrong for the fraction of points sitting within the
                   rounding residual of a boundary, so agreement degrades
                   predictably to 1 - 2*frac (see `predicted_agreement`).

    The codebook is usable, not merely diagnostic: `encode` returns integer
    tokens suitable for a transformer over discrete inputs.
    """

    def __init__(self, lam: float, bins: int = 16, ref_deg=None):
        self.lam, self.bins, self.ref_deg = lam, bins, ref_deg
        self.n_codes = bins * bins
        self.cell_m = lam / bins

    def fit(self, *_):
        return self

    def _phase(self, lonlat):
        xy, _ = lonlat_to_enu(lonlat, self.ref_deg)
        return (xy / self.lam) % 1.0

    def _phase_bins(self, lonlat):
        return np.floor(self._phase(lonlat) * self.bins).astype(int) % self.bins

    def encode(self, lonlat):
        b = self._phase_bins(lonlat)
        return b[:, 0] * self.bins + b[:, 1]

    def decode(self, idx):
        """Cell-centre phase of each token, for inspecting what a token means."""
        bx, by = np.asarray(idx) // self.bins, np.asarray(idx) % self.bins
        return (np.stack([bx, by], 1) + 0.5) / self.bins

    def shift_in_cells(self, v) -> np.ndarray:
        return np.asarray(v, float) / self.cell_m

    def is_on_lattice(self, v, tol=1e-9) -> bool:
        s = self.shift_in_cells(v)
        return bool(np.all(np.abs(s - np.round(s)) < tol))

    def predicted_permutation(self, v) -> np.ndarray:
        """The index map rho(v) induces on the codebook.

        Exact when v is a multiple of the cell size; otherwise the nearest-cell
        approximation, whose expected error `predicted_agreement` quantifies.
        """
        shift = np.round(self.shift_in_cells(v)).astype(int)
        idx = np.arange(self.n_codes)
        bx, by = idx // self.bins, idx % self.bins
        return ((bx + shift[0]) % self.bins) * self.bins + (by + shift[1]) % self.bins

    def predicted_agreement(self, v) -> float:
        """Expected agreement for uniformly distributed phases.

        With rounding residual r_a in [-1/2, 1/2] cells on axis a, a point
        disagrees on that axis exactly when its within-cell offset falls in a
        band of width |r_a|. The axes are independent, so
        agreement = prod_a (1 - |r_a|). On-lattice gives 1.0 exactly, which is
        the falsifiable case; off-lattice gives a number to check against.
        """
        s = self.shift_in_cells(v)
        return float(np.prod(1.0 - np.abs(s - np.round(s))))


# Backwards-compatible alias
TorusPhaseCodebook = EquivariantTorusCodebook


# --------------------------------------------------------------------------
# the two repaired tests
# --------------------------------------------------------------------------
def lattice_period_test(encoder, codebook, lonlat, v, ref=None):
    """Fraction of points keeping their index under translation by v.

    With v a lattice period of the encoder this must be 1.0 exactly; with an
    arbitrary v it will not be, and should not be expected to.
    """
    xy, ref = lonlat_to_enu(lonlat, ref)
    moved = enu_to_lonlat(xy + np.asarray(v, float), ref)
    i0 = codebook.encode(_maybe_encode(encoder, lonlat))
    i1 = codebook.encode(_maybe_encode(encoder, moved))
    return float((i0 == i1).mean())


def equivariant_codebook_test(codebook: TorusPhaseCodebook, lonlat, v, ref=None):
    """Fraction of points whose index moves *as predicted* by rho(v)."""
    xy, ref = lonlat_to_enu(lonlat, ref)
    moved = enu_to_lonlat(xy + np.asarray(v, float), ref)
    i0, i1 = codebook.encode(lonlat), codebook.encode(moved)
    return float((codebook.predicted_permutation(v)[i0] == i1).mean())


def codebook_agreement_curve(cb: EquivariantTorusCodebook, lonlat, mags,
                            axis=(1.0, 0.0), ref=None):
    """Measured vs predicted permutation agreement, as a function of ||v||.

    Reports both the on-lattice magnitudes (multiples of the cell size, where
    the prediction is exact and must be met at 1.0) and arbitrary ones (where
    the analytic prediction is a number to be checked, not a bound).
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    xy, ref = lonlat_to_enu(lonlat, ref)
    out = {}
    for m in mags:
        v = m * axis
        moved = enu_to_lonlat(xy + v, ref)
        i0, i1 = cb.encode(lonlat), cb.encode(moved)
        out[float(m)] = {
            "measured": float((cb.predicted_permutation(v)[i0] == i1).mean()),
            "predicted": cb.predicted_agreement(v),
            "on_lattice": cb.is_on_lattice(v)}
    return out


def permutation_transfer_test(encoder, codebook, lonlat, v, ref=None,
                              seed: int = 0):
    """Does the induced index map generalise to points it was not fitted on?

    This separates a genuine group action from a coincidence. Fit the index
    map empirically on half the points -- for each source token, the most
    frequent destination token -- and apply it to the held-out half.

    An equivariant codebook has a single permutation that is correct
    everywhere, so held-out agreement equals fitted agreement at 1.0, and the
    empirically fitted map turns out to BE the analytic one.

    Read `heldout_agreement` as the headline, not `generalisation_gap`. For a
    coarse k-means codebook and a small v the best empirical map is close to
    the identity, which generalises fine while agreeing only ~0.83 of the
    time; a near-zero gap there means the map is trivial, not that the
    codebook is equivariant. `empirical_matches_analytic` exists only where an
    analytic permutation exists at all, which is the real distinction.
    """
    xy, ref = lonlat_to_enu(lonlat, ref)
    moved = enu_to_lonlat(xy + np.asarray(v, float), ref)
    src = _maybe_encode(encoder, lonlat)
    dst = _maybe_encode(encoder, moved)
    i0, i1 = codebook.encode(src), codebook.encode(dst)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(i0))
    fit, held = order[:len(order) // 2], order[len(order) // 2:]

    n_codes = getattr(codebook, "n_codes", None) or len(codebook.C)
    perm = np.arange(n_codes)
    for c in np.unique(i0[fit]):
        tgt = i1[fit][i0[fit] == c]
        if len(tgt):
            perm[c] = np.bincount(tgt, minlength=n_codes).argmax()

    out = {"fitted_agreement": float((perm[i0[fit]] == i1[fit]).mean()),
           "heldout_agreement": float((perm[i0[held]] == i1[held]).mean())}
    out["generalisation_gap"] = out["fitted_agreement"] - out["heldout_agreement"]
    if hasattr(codebook, "predicted_permutation"):
        an = codebook.predicted_permutation(v)
        out["analytic_agreement"] = float((an[i0[held]] == i1[held]).mean())
        out["empirical_matches_analytic"] = float((perm == an).mean())
    return out


def gauge_defect_curve(warp_encoder, lonlat, vs, ref=None):
    """Regime (ii): measured deviation vs the first-order prediction (5).

    Returns, per ||v||, the phase error between phi(l+v) and the gauge
    prediction rho_l(v) phi(l). The proposal asks for this O(||v||^2) defect
    to be measured rather than asserted; a slope of ~2 on a log-log fit is the
    confirmation.
    """
    xy, ref = lonlat_to_enu(lonlat, ref)
    out = {}
    for v in vs:
        v = np.asarray(v, float)
        moved = enu_to_lonlat(xy + v, ref)
        E0, E1 = warp_encoder.encode(lonlat), warp_encoder.encode(moved)
        dphase = warp_encoder.rho_gauge(lonlat, v)          # (N, 2) predicted
        pred = _rotate_pairs(E0, dphase)
        err = np.linalg.norm(E1 - pred, axis=1).mean()
        out[float(np.linalg.norm(v))] = float(err)
    return out


def _rotate_pairs(E, dphase):
    """E laid out as [sin x, cos x, sin y, cos y]; rotate each pair."""
    out = np.empty_like(E)
    for a in (0, 1):
        s, c = E[:, 2 * a], E[:, 2 * a + 1]
        ca, sa = np.cos(dphase[:, a]), np.sin(dphase[:, a])
        out[:, 2 * a] = s * ca + c * sa
        out[:, 2 * a + 1] = c * ca - s * sa
    return out


def fit_loglog_slope(curve: dict, min_v: float = 0.0) -> float:
    """Log-log slope of the defect curve, optionally on the tail only.

    At small ||v|| the measurement sits on a floor set by the resolution of
    the interpolant used for eta, not by the construction, so the O(||v||^2)
    prediction is only testable in the asymptotic regime. Report both.
    """
    ks = [k for k in curve if k > min_v]
    x = np.log(np.array(ks))
    y = np.log(np.array([curve[k] for k in ks]) + 1e-15)
    return float(np.polyfit(x, y, 1)[0])


def _maybe_encode(encoder, lonlat):
    return lonlat if encoder is None else encoder.encode(lonlat)
