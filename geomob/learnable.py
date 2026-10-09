"""Learnable wavelengths for GEO (and, for a fair comparison, for GPE).

With `[encoders] learn_lambdas = "loc" | "disp" | "both"` the wavelengths are
only INITIALISED on the geometric ladder lam_min .. lam_max and are then
trained with the rest of the model.

Why the symmetry guarantees survive
    Location block: phi(p) = [sin(2 pi p / lam), cos(2 pi p / lam)]. For ANY
    lam, phi(p + v) = R(2 pi v / lam) phi(p) with R a 2x2 rotation, so the
    block stays exactly translation-equivariant; learning lam only changes
    WHICH representation rho(v) acts, not WHETHER one exists.
    Radial block: a function of r = |dx| only, so it stays exactly invariant
    to rotations and translations whatever the wavelengths are.
    Angular block: has no wavelength and is not touched.
What is lost is that rho is known in advance: it depends on the learned
wavelengths, which are therefore saved with the results.

How it is wired
    The tokenizer emits RAW geometry for the learnable parts (east/north
    metres in the local frame, step length r) and ready-made codes for the
    rest; a small torch front end (`FrontEnd`) turns the raw columns into
    sin/cos codes with trainable wavelengths, in exactly the same layout as
    the fixed encoders. At initialisation the two are numerically identical
    (checked by `check_init_identical`).

Parametrisation
    log lam = log lam0 + kappa * u, with u a trainable vector starting at 0.
    Adam normalises step sizes, so kappa (`lambda_lr_scale`) sets how fast
    log-wavelengths move. Scales whose initial wavelength is below
    `learn_lambda_min_m` stay fixed: for a short wavelength and a far point
    the phase 2 pi x / lam is huge and its gradient with respect to lam
    oscillates wildly, so learning those scales is mostly noise.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import Traj
from .geo import R_EARTH, lonlat_to_enu


# ==========================================================================
# torch side
# ==========================================================================
class SinCosBank(nn.Module):
    """sin/cos of 2 pi v / lam for a vector of trainable wavelengths."""

    def __init__(self, lams0, learn_mask, kappa=1.0):
        super().__init__()
        lams0 = np.asarray(lams0, dtype=np.float64)
        # float64: the wavelength's own rounding error is multiplied by the
        # phase (up to ~1e6 rad for GPE), so it must be stored in double
        self.register_buffer("log0", torch.tensor(np.log(lams0), dtype=torch.float64))
        self.register_buffer("mask", torch.tensor(np.asarray(learn_mask, float),
                                                  dtype=torch.float64))
        self.u = nn.Parameter(torch.zeros(len(lams0)))
        self.kappa = float(kappa)

    def lams(self):
        return torch.exp(self.log0 + self.kappa * self.u.double() * self.mask)

    def forward(self, v, offset=0.0):            # v: (...,) -> (..., 2n)
        # phases in float64: 2 pi x / lam reaches 1e4 rad for GEO and 1e6 rad
        # for GPE's top frequency, where float32 would garble the fine scales
        lam = self.lams()
        a = 2 * np.pi * (v.double().unsqueeze(-1) + offset) / lam
        return torch.stack([torch.sin(a), torch.cos(a)], -1).flatten(-2).float()


class FrontEnd(nn.Module):
    """Raw columns -> codes. `plan` is a list of
        ("xy", start, bank)   two raw columns, each through the same bank
        ("r", col, bank)      one raw column
        ("pass", start, end)  columns already encoded
    in output order."""

    def __init__(self, plan, xy_offset=(0.0, 0.0)):
        super().__init__()
        self.plan = [(k, a, b if not isinstance(b, nn.Module) else i)
                     for i, (k, a, b) in enumerate(plan)]
        self.banks = nn.ModuleDict({str(i): b for i, (_, _, b) in enumerate(plan)
                                    if isinstance(b, nn.Module)})
        # constant added (in float64) to the two xy columns: lets GPE take
        # small offsets from a reference instead of absolute radians
        self.xy_offset = tuple(float(o) for o in xy_offset)

    def forward(self, x):
        out = []
        for kind, a, b in self.plan:
            if kind == "xy":
                bank = self.banks[str(b)]
                out += [bank(x[..., a], self.xy_offset[0]),
                        bank(x[..., a + 1], self.xy_offset[1])]
            elif kind == "r":
                # r < 0 marks "no previous point" (first point of a trip): the
                # fixed encoder gives it an all-zero code, so does this
                v = x[..., a]
                out.append(self.banks[str(b)](v.clamp(min=0.0))
                           * (v >= 0).unsqueeze(-1).float())
            else:
                out.append(x[..., a:b])
        return torch.cat(out, -1)

    def wavelengths(self, unit_scale=None):
        """Current wavelengths per bank (metres; radians x R for GPE)."""
        names = {"xy": "location", "r": "radial"}
        res = {}
        for kind, a, b in self.plan:
            if kind == "pass":
                continue
            lam = self.banks[str(b)].lams().detach().cpu().numpy().astype(float)
            if unit_scale:
                lam = lam * unit_scale
            res[names[kind]] = [float(v) for v in lam]
        return res


class WithFrontEnd(nn.Module):
    """front end, then any sequence model taking (x, mask)."""

    def __init__(self, front, body):
        super().__init__()
        self.front, self.body = front, body

    def forward(self, x, mask=None):
        return self.body(self.front(x), mask)


def attach(model_or_head, tok):
    """Wrap a freshly built model so it starts with tok's front end. Works on
    a TrajEncoder directly or on a module with an `.enc` TrajEncoder."""
    if not hasattr(tok, "frontend"):
        return model_or_head
    if hasattr(model_or_head, "enc"):
        model_or_head.enc = WithFrontEnd(tok.frontend(), model_or_head.enc)
        return model_or_head
    return WithFrontEnd(tok.frontend(), model_or_head)


def learned_wavelengths(model):
    """{bank name: [wavelengths]} for every front end inside `model`."""
    out = {}
    for m in model.modules():
        if isinstance(m, WithFrontEnd):
            out.update(m.front.wavelengths(getattr(m.front, "unit_scale", None)))
    return out


# ==========================================================================
# numpy side: tokenizers that emit raw geometry for the learnable parts
# ==========================================================================
class _Loc:
    """Stand-in for the location channel: carries the frame origin, so
    `tok.loc.ref_deg` works exactly as for the fixed tokenizer."""

    def __init__(self, ref_deg=None):
        self.ref_deg = ref_deg


class LearnableGEO:
    """GEO with trainable wavelengths in the location and/or radial blocks.
    Built from the same (fixed) parts as the normal GEO tokenizer, whose
    wavelengths become the initial values."""

    def __init__(self, loc, disp, time=None, speed=None, learn="both",
                 min_m=0.0, kappa=1.0, name="GEO"):
        self.fixed_loc, self.disp, self.time, self.speed = loc, disp, time, speed
        self.learn_loc = learn in ("loc", "both")
        self.learn_disp = learn in ("disp", "both") and disp is not None \
            and len(disp.lams) > 0
        self.loc = _Loc(getattr(loc, "ref_deg", None))
        self.min_m, self.kappa, self.name = float(min_m), float(kappa), name
        self.learn = learn

    # -- shape ------------------------------------------------------------
    @property
    def dim(self):
        d = self.fixed_loc.dim
        d += 0 if self.disp is None else self.disp.dim
        d += 0 if self.time is None else self.time.dim
        d += 0 if self.speed is None else self.speed.dim
        return d

    def _raw_layout(self):
        """(raw columns per block, in order) -> used by both sides."""
        lay = [("loc", 2 if self.learn_loc else self.fixed_loc.dim)]
        if self.disp is not None:
            if self.learn_disp:
                lay += [("r", 1), ("ang", self.disp.dim - self.disp.n_inv)]
            else:
                lay += [("disp", self.disp.dim)]
        if self.speed is not None:
            lay += [("speed", self.speed.dim)]
        if self.time is not None:
            lay += [("time", self.time.dim)]
        return lay

    @property
    def raw_cols(self):
        """Raw (un-encoded) columns: these must not be standardised."""
        cols, c = [], 0
        for name, w in self._raw_layout():
            if (name == "loc" and self.learn_loc) or name == "r":
                cols += list(range(c, c + w))
            c += w
        return cols

    # -- fitting / encoding ---------------------------------------------------
    def fit(self, trajs, force=False):
        if self.loc.ref_deg is None or force:
            self.loc.ref_deg = np.concatenate([t.lonlat for t in trajs]).mean(0)
        self.fixed_loc.ref_deg = self.loc.ref_deg
        return self

    def __call__(self, traj: Traj) -> np.ndarray:
        self.fixed_loc.ref_deg = self.loc.ref_deg       # follow a re-fit frame
        xy = lonlat_to_enu(traj.lonlat, self.loc.ref_deg)[0]
        parts = [xy if self.learn_loc else self.fixed_loc.encode(traj.lonlat)]
        if self.disp is not None:
            d = np.diff(xy, axis=0)
            if self.learn_disp:
                code = self.disp.encode_disp(d)
                r = np.linalg.norm(d, axis=1)[:, None]
                block = np.concatenate([r, code[:, self.disp.n_inv:]], 1)
                first = np.zeros((1, block.shape[1]))
                first[0, 0] = -1.0              # no step before the first point
            else:
                block = self.disp.encode_disp(d)
                first = np.zeros((1, block.shape[1]))
            parts.append(np.vstack([first, block]))
        if self.speed is not None:
            sp = self.speed.encode_speed(np.diff(xy, axis=0), np.diff(traj.t))
            parts.append(np.vstack([np.zeros((1, sp.shape[1])), sp]))
        if self.time is not None:
            parts.append(self.time.encode(t=traj.t))
        return np.concatenate(parts, 1).astype(np.float32)

    def frontend(self):
        plan, c = [], 0
        for name, w in self._raw_layout():
            if name == "loc" and self.learn_loc:
                lams = self.fixed_loc.lams
                plan.append(("xy", c, SinCosBank(lams, lams >= self.min_m, self.kappa)))
            elif name == "r":
                lams = self.disp.lams
                plan.append(("r", c, SinCosBank(lams, lams >= self.min_m, self.kappa)))
            else:
                plan.append(("pass", c, c + w))
            c += w
        return FrontEnd(plan)


class LearnableGPE:
    """GPE with trainable frequencies: the fairness control for learnable GEO.
    Raw columns are (lon, lat) in radians; wavelengths are reported in metres
    (radians x Earth radius) so they can be read next to GEO's."""

    def __init__(self, gpe, min_m=0.0, kappa=1.0, name="GPE (learned freqs)",
                 time=None, speed=None):
        self.gpe, self.min_m, self.kappa, self.name = gpe, float(min_m), float(kappa), name
        self.time, self.speed = time, speed
        self.loc = _Loc(None)          # no metric frame: nothing to re-anchor
        self.dim = gpe.dim + (time.dim if time else 0) + (speed.dim if speed else 0)
        self.raw_cols = [0, 1]
        self.ref_rad = None

    def fit(self, trajs, force=False):
        # a reference point only for numerical precision: the code is still
        # GPE's absolute global code (the offset is added back in float64)
        if self.ref_rad is None or force:
            self.ref_rad = np.deg2rad(np.concatenate([t.lonlat for t in trajs]).mean(0))
        return self

    def __call__(self, traj):
        rad = np.deg2rad(np.asarray(traj.lonlat, dtype=np.float64))
        parts = [rad - self.ref_rad]
        if self.speed is not None:          # same order as the fixed Tokenizer
            xy = lonlat_to_enu(traj.lonlat)[0]
            sp = self.speed.encode_speed(np.diff(xy, axis=0), np.diff(traj.t))
            parts.append(np.vstack([np.zeros((1, sp.shape[1])), sp]))
        if self.time is not None:
            parts.append(self.time.encode(t=traj.t))
        return np.concatenate(parts, 1).astype(np.float32)

    def frontend(self):
        lams_rad = 2 * np.pi / self.gpe.freqs
        bank = SinCosBank(lams_rad, lams_rad * R_EARTH >= self.min_m, self.kappa)
        plan = [("xy", 0, bank)]
        extra = (self.speed.dim if self.speed else 0) + (self.time.dim if self.time else 0)
        if extra:
            plan.append(("pass", 2, 2 + extra))
        fe = FrontEnd(plan, xy_offset=tuple(self.ref_rad))
        fe.unit_scale = R_EARTH
        return fe


# ==========================================================================
# checks
# ==========================================================================
@torch.no_grad()
def check_init_identical(tok_learn, tok_fixed, traj) -> float:
    """Max |difference| between the learnable pipeline at initialisation and
    the fixed tokenizer on one trajectory (should be ~1e-5, float32)."""
    raw = torch.from_numpy(tok_learn(traj)).unsqueeze(0)
    out = tok_learn.frontend()(raw)[0].numpy()
    return float(np.abs(out - tok_fixed(traj)).max())


@torch.no_grad()
def check_translation_equivariance(front, n=200, shift_m=(1234.0, -567.0), seed=0):
    """Perturb the wavelengths away from their initial values, then check that
    translating the location by v rotates every (sin, cos) pair by exactly
    2 pi v / lam, with lam the CURRENT (learned) wavelength. Returns the max
    error (float32 rounding, ~1e-3 at most for 5 km coordinates)."""
    torch.manual_seed(seed)
    for b in front.banks.values():
        b.u.normal_(0, 0.3)
    xy = [(a, bi) for k, a, bi in front.plan if k == "xy"]
    if not xy:
        return 0.0
    bank = front.banks[str(xy[0][1])]
    lam = bank.lams().double().numpy()
    k = len(lam)
    raw = torch.randn(n, 2, dtype=torch.float64) * 5000.0
    shifted = raw + torch.tensor(shift_m, dtype=torch.float64)

    def code(v):            # in float64 to measure the maths, not rounding
        a = 2 * np.pi * v.numpy()[:, None] / lam[None, :]
        return np.sin(a), np.cos(a)

    err = 0.0
    for axis, v in enumerate(shift_m):
        s0, c0 = code(raw[:, axis])
        s1, c1 = code(shifted[:, axis])
        th = 2 * np.pi * v / lam
        err = max(err, float(np.abs(s0 * np.cos(th) + c0 * np.sin(th) - s1).max()),
                  float(np.abs(c0 * np.cos(th) - s0 * np.sin(th) - c1).max()))
    # and the torch bank really computes that code
    out = front.banks[str(xy[0][1])](raw[:, 0].float()).double().numpy().reshape(n, k, 2)
    s0, c0 = code(raw[:, 0])
    err_impl = float(np.abs(out[..., 0] - s0).max() + np.abs(out[..., 1] - c0).max())
    return max(err, 0.0), err_impl
