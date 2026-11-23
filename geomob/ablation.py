"""Channel ablations for GEO tokenizers.

A GEO token is the concatenation, in this order, of

    loc      location lattice
    radial   step length (displacement, invariant part)
    angular  heading harmonics (displacement, equivariant part)
    speed    speed and step duration
    time     time of day / day of week

An ablated tokenizer computes the FULL token exactly as usual (same frame,
same displacement and speed computation, same wavelengths) and then drops the
columns of the removed channels, so every remaining channel is bit-for-bit
the one the full model sees. With learned wavelengths the codes are computed
inside the model, so the columns are dropped there, after the front end.

Variants (`mode`):
    drop   leave one channel out ("GEO - location", ...)
    only   keep one channel ("GEO: location only", ...)
    both   all of the above
    none   no ablation
Only channels the task's GEO actually has are ablated (e.g. no speed for
travel-time estimation).
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("loc", "radial", "angular", "speed", "time")
LABEL = {"loc": "location", "radial": "step length", "angular": "heading",
         "speed": "speed", "time": "time"}
MODES = ("none", "drop", "only", "both")


def _core(tok):
    """The plain (Tokenizer) or learnable GEO tokenizer inside wrappers."""
    seen = 0
    while seen < 10:
        seen += 1
        d = vars(tok)
        if "tok" in d and isinstance(tok, Ablated):
            tok = d["tok"]
        elif "inner" in d and not ("disp" in d and "time" in d):
            tok = d["inner"]                    # FramedTokenizer
        else:
            return tok
    return tok


def channel_slices(tok):
    """{channel: (start, end)} in the model-input (encoded) token."""
    t = _core(tok)
    loc = getattr(t, "fixed_loc", None)
    if loc is None:
        loc = getattr(t, "loc", None)
    out, c = {}, 0
    if loc is not None and getattr(loc, "dim", 0):
        out["loc"] = (c, c + loc.dim)
        c += loc.dim
    disp = getattr(t, "disp", None)
    if disp is not None and disp.dim:
        n_inv = int(getattr(disp, "n_inv", disp.dim))
        if n_inv:
            out["radial"] = (c, c + n_inv)
        if disp.dim - n_inv:
            out["angular"] = (c + n_inv, c + disp.dim)
        c += disp.dim
    for name in ("speed", "time"):
        ch = getattr(t, name, None)
        if ch is not None and ch.dim:
            out[name] = (c, c + ch.dim)
            c += ch.dim
    return out


def present(tok):
    return list(channel_slices(tok))


def variants(tok, mode="drop"):
    """[(key, channels kept)] for the requested ablation mode."""
    if mode not in MODES:
        raise SystemExit(f"ablations must be one of {MODES}, got {mode!r}")
    chans = present(tok)
    if mode == "none" or len(chans) < 2:
        return []
    out = []
    if mode in ("drop", "both"):
        out += [(f"no_{c}", tuple(x for x in chans if x != c)) for c in chans]
    if mode in ("only", "both"):
        out += [(f"only_{c}", (c,)) for c in chans]
    return out


def variant_label(vkey, prefix="GEO"):
    kind, ch = vkey.split("_", 1)
    return f"{prefix} − {LABEL[ch]}" if kind == "no" else f"{prefix}: {LABEL[ch]} only"


def parse_variant(vkey):
    """'no_loc' -> ('no', 'loc')"""
    kind, ch = vkey.split("_", 1)
    if kind not in ("no", "only") or ch not in CHANNELS:
        raise KeyError(vkey)
    return kind, ch


def keep_for(tok, vkey):
    kind, ch = parse_variant(vkey)
    chans = present(tok)
    if ch not in chans:
        raise KeyError(f"{vkey}: this tokenizer has no {LABEL[ch]} channel")
    return tuple(x for x in chans if x != ch) if kind == "no" else (ch,)


class Ablated:
    """A tokenizer that keeps only some channels of `tok` (see module doc).
    Everything else (fit, refit, frame, ...) is delegated to `tok`, including
    attribute writes, so code that saves and restores a frame still works."""

    def __init__(self, tok, keep, name=None):
        sl = channel_slices(tok)
        missing = [c for c in keep if c not in sl]
        if missing:
            raise KeyError(f"no channel(s) {missing} in this tokenizer")
        idx = [i for c, (a, b) in sl.items() if c in keep for i in range(a, b)]
        object.__setattr__(self, "tok", tok)
        object.__setattr__(self, "keep", np.asarray(idx, dtype=np.int64))
        object.__setattr__(self, "channels", [c for c in sl if c in keep])
        object.__setattr__(self, "name", name or (getattr(tok, "name", "GEO")
                                                  + " [" + "+".join(keep) + "]"))

    @property
    def dim(self):
        return int(len(self.keep))

    def _learnable(self):
        try:
            getattr(object.__getattribute__(self, "tok"), "frontend")
            return True
        except AttributeError:
            return False

    def __call__(self, traj):
        x = self.tok(traj)
        if self._learnable():
            return x                 # raw columns; the front end drops
        return x[:, self.keep]

    def __getattr__(self, name):
        if name.startswith("__") or name in ("tok", "keep", "channels"):
            raise AttributeError(name)
        if name == "frontend":
            make = getattr(object.__getattribute__(self, "tok"), "frontend")
            keep = object.__getattribute__(self, "keep")

            def frontend():
                fe = make()
                fe.set_keep(keep)
                return fe
            return frontend
        return getattr(object.__getattribute__(self, "tok"), name)

    def __setattr__(self, name, value):
        if name in ("name",):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "tok"), name, value)


def ablate(tok, vkey, name=None):
    return Ablated(tok, keep_for(tok, vkey), name=name)
