"""Run configuration: one TOML file for every hyperparameter, saved with results.

    python -m geomob.cli config --write geomob.toml     # commented template
    python -m geomob.cli gpe_suite --config geomob.toml
    python -m geomob.cli gpe_suite --config geomob.toml --epochs 2   # flag wins

Precedence, lowest to highest:
    built-in defaults  <  [common]  <  [<command>]  <  command-line flags

Sections
    [paths]        dataset locations (name = "path"); beat DATA_<NAME> exports
    [common]       keys applied to every command that has them (device, seeds..)
    [model]        sequence-model architecture shared by every experiment
    [encoders]     GPE / GEO / time / speed encoder hyperparameters used by the
                   comparison runners (gpe_suite, downstream, run_all)
    [search_space] candidate values per [encoders]/[model] key, for `search`
    [<command>]    that command's own options, same names as its flags with
                   '-' written as '_' (e.g. [gpe_suite] n_test = 5000)

Unknown sections or keys are an error (with a suggestion), so a typo can never
silently fall back to a default.

Every results JSON written through `save_results` carries a `_run` block --
command, argv, the config file verbatim, every resolved option including the
defaults nobody touched, the architecture, data paths and library versions --
and a `<results>.config.toml` is written beside it that re-runs the same
experiment.
"""
from __future__ import annotations

import copy
import datetime as _dt
import difflib
import json
import os
import platform
import socket
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# architecture hyperparameters (not command-line flags; set via [model] /
# [encoders]). Read at construction time by models.TrajEncoder and by the
# encoder factories of the comparison runners.
# --------------------------------------------------------------------------
_HP_DEFAULTS = {
    "model": {
        "hidden": 128,        # width of the LSTM / Transformer / GNN
        "layers": 2,          # LSTM and Transformer depth
        "heads": 4,           # Transformer / GAT attention heads
        "dropout": 0.0,
    },
    "encoders": {
        "h": 128,                       # embedding size: GPE h; GEO total dim
        "gpe_eps_over_2pi": 1e-6,       # GPE Algorithm 1 epsilon-cut
        "geo_loc_dims": 0,              # of the h dims, how many go to location
                                        # (0 = half); the rest is displacement
        "geo_loc_scales": 0,            # 0 = derive from h (h // 8)
        "geo_loc_lam_min_m": 25.0,      # finest location wavelength
        "geo_loc_lam_max_m": 200_000.0, # coarsest location wavelength
        "geo_disp_scales": 0,           # 0 = derive from h ((h/2 - 4) / 2)
        "geo_disp_lam_min_m": 5.0,
        "geo_disp_lam_max_m": 20_000.0,
        "geo_harmonics": [1, 4],        # angular harmonics; m=4 = grid order
        "time_hour_harmonics": 4,       # C24 harmonics
        "time_dow_harmonics": 3,        # C7 harmonics
        "speed_scales": 6,              # SpeedGEO ladder
        "speed_lo": 0.05,
        "speed_hi": 60.0,
    },
}
HP = copy.deepcopy(_HP_DEFAULTS)

SPECIAL_SECTIONS = ("paths", "common", "model", "encoders", "search_space")
SKIP_DESTS = {"help", "config"}

# the record of the current invocation, attached to every saved result
RUN: dict = {}
# command -> option names the CLI accepts (filled by the CLI); a re-run
# config must only contain these
VALID_KEYS: dict = {}


def hp(section: str, key: str):
    return HP[section][key]


def reset():
    HP.clear()
    HP.update(copy.deepcopy(_HP_DEFAULTS))
    RUN.clear()


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def load(path: str) -> tuple[dict, str]:
    """Parse a .toml (or .json) config. Returns (dict, raw text)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"config file not found: {p}")
    text = p.read_text()
    if p.suffix.lower() == ".json":
        return json.loads(text), text
    try:
        import tomllib                      # Python >= 3.11
    except ImportError:                     # pragma: no cover
        try:
            import tomli as tomllib         # pip install tomli
        except ImportError:
            raise SystemExit("reading TOML needs Python >= 3.11 or "
                             "`pip install tomli` (or use a .json config)")
    try:
        return tomllib.loads(text), text
    except Exception as e:                  # noqa: BLE001
        raise SystemExit(f"{p}: invalid TOML: {e}")


def _unknown(kind, key, valid, where):
    near = difflib.get_close_matches(key, list(valid), n=3)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    raise SystemExit(f"config: unknown {kind} '{key}' in {where}.{hint}\n"
                     f"  valid: {', '.join(sorted(valid))}")


def _norm(d: dict) -> dict:
    return {k.replace("-", "_"): v for k, v in d.items()}


def apply_architecture(cfg: dict):
    for sec in ("model", "encoders"):
        for k, v in _norm(cfg.get(sec, {})).items():
            if k not in HP[sec]:
                _unknown("key", k, HP[sec], f"[{sec}]")
            ref = _HP_DEFAULTS[sec][k]
            if isinstance(ref, float) and isinstance(v, int):
                v = float(v)
            HP[sec][k] = v


def apply_paths(cfg: dict):
    """[paths] name = "path" -> DATA_<NAME>, overriding any export."""
    from .data import DATASETS
    for k, v in cfg.get("paths", {}).items():
        if k not in DATASETS:
            _unknown("dataset", k, DATASETS, "[paths]")
        os.environ[f"DATA_{k.upper()}"] = str(Path(str(v)).expanduser())


def _actions(subparser):
    return {a.dest: a for a in subparser._actions
            if a.dest not in SKIP_DESTS and a.option_strings}


def _coerce(action, key, v, where):
    many = action.nargs in ("+", "*") or (isinstance(action.nargs, int)
                                          and action.nargs > 0)
    if many and not isinstance(v, list):
        v = [v]
    if not many and isinstance(v, list):
        raise SystemExit(f"config: {where} {key} takes one value, got a list")
    t = action.type
    vals = v if many else [v]
    if t in (int, float):
        try:
            vals = [t(x) for x in vals]
        except (TypeError, ValueError):
            raise SystemExit(f"config: {where} {key} must be {t.__name__}, got {v!r}")
    if action.choices is not None:
        bad = [x for x in vals if x not in action.choices]
        if bad:
            raise SystemExit(f"config: {where} {key}={bad} not in "
                             f"{list(action.choices)}")
    return vals if many else vals[0]


def command_defaults(cfg: dict, subparsers: dict, cmd: str) -> dict:
    """Validated defaults for `cmd`: [common] keys it has, then [cmd]."""
    for sec in cfg:
        if sec in SPECIAL_SECTIONS or sec in subparsers:
            continue
        if not isinstance(cfg[sec], dict):
            continue                        # tolerate top-level scalars
        _unknown("section", sec, list(SPECIAL_SECTIONS) + list(subparsers),
                 "the config file")
    # validate EVERY command section now, so a typo in a section you are not
    # running today is still caught
    for sec, sp in subparsers.items():
        acts = _actions(sp)
        for k in _norm(cfg.get(sec, {})):
            if k not in acts:
                _unknown("key", k, acts, f"[{sec}]")
    common = _norm(cfg.get("common", {}))
    known_anywhere = set().union(*(_actions(sp) for sp in subparsers.values()))
    for k in common:
        if k not in known_anywhere:
            _unknown("key", k, known_anywhere, "[common]")

    acts = _actions(subparsers[cmd])
    out = {}
    for k, v in common.items():
        if k in acts:
            out[k] = _coerce(acts[k], k, v, "[common]")
    for k, v in _norm(cfg.get(cmd, {})).items():
        out[k] = _coerce(acts[k], k, v, f"[{cmd}]")
    return out


def section_overrides(cfg: dict, subparsers: dict, sec: str) -> dict:
    """[common] + [sec] for a step that run_all calls directly."""
    if sec not in subparsers:
        return {}
    return command_defaults(cfg, subparsers, sec)


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------
def _versions():
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "host": socket.gethostname()}
    for mod in ("numpy", "torch", "pandas", "scipy", "matplotlib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                   # noqa: BLE001
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:                       # noqa: BLE001
        pass
    return out


def start_run(cmd: str, args, config_path=None, config_text=None, argv=None):
    from . import __version__
    from .data import LOADERS
    RUN.clear()
    resolved = {k: v for k, v in vars(args).items() if k not in SKIP_DESTS | {"cmd"}}
    RUN.update({
        "geomob_version": __version__,
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "argv": list(sys.argv if argv is None else argv),
        "config_file": str(Path(config_path).resolve()) if config_path else None,
        "config_file_text": config_text,
        "options": resolved,
        "architecture": copy.deepcopy(HP),
        "data_paths": {n: os.environ[f"DATA_{n.upper()}"] for n in LOADERS
                       if os.environ.get(f"DATA_{n.upper()}")},
        "environment": _versions(),
    })
    return RUN


def save_results(res: dict, out, extra: dict | None = None):
    """json.dump with the run record attached, plus a re-runnable TOML."""
    if not out:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if RUN:
        res["_run"] = {**copy.deepcopy(RUN), **(extra or {}),
                       "saved": _dt.datetime.now().isoformat(timespec="seconds")}
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=_json_default)
    if RUN:
        cmd = (extra or {}).get("command", RUN.get("command"))
        opts = (extra or {}).get("options", RUN.get("options", {}))
        side = out.with_suffix(".config.toml")
        side.write_text(to_toml(cmd, opts, RUN.get("data_paths", {}), HP,
                                header=f"written with {out.name}"))


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


# --------------------------------------------------------------------------
# writing TOML (tiny subset: scalars, strings, flat lists)
# --------------------------------------------------------------------------
def _tv(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_tv(x) for x in v) + "]"
    return json.dumps(str(v))


def _section(name, d, helps=None, commented=False):
    lines = [f"[{name}]"]
    for k, v in d.items():
        h = (helps or {}).get(k)
        if h:
            lines.append(f"# {h}")
        if v is None:
            lines.append(f"# {k} =")
            continue
        lines.append(("# " if commented else "") + f"{k} = {_tv(v)}")
    return "\n".join(lines) + "\n"


def to_toml(cmd, options, paths, hp_, header=""):
    head = [f"# geomob run configuration ({header})",
            f"# re-run with:  python -m geomob.cli {cmd} --config <this file>", ""]
    parts = ["\n".join(head)]
    if paths:
        parts.append(_section("paths", paths))
    valid = VALID_KEYS.get(cmd)
    opts = {k: v for k, v in options.items() if k != "paths"}
    good = {k: v for k, v in opts.items() if valid is None or k in valid}
    other = {k: v for k, v in opts.items() if k not in good}
    sec = _section(cmd, good)
    if other:   # recorded, but not settable from a config / the CLI
        sec += "# fixed by the code for this run (informational):\n" + "".join(
            f"#   {k} = {_tv(v) if v is not None else 'unset'}\n" for k, v in other.items())
    parts.append(sec)
    parts.append(_section("model", hp_["model"]))
    parts.append(_section("encoders", hp_["encoders"]))
    return "\n".join(parts)


ENCODER_HELP = {
    "h": "embedding size: GPE h, and GEO's total dimension (half location, half displacement)",
    "gpe_eps_over_2pi": "GPE Algorithm 1 epsilon-cut",
    "geo_loc_dims": "how many of the h numbers go to LOCATION (multiple of 4); the rest goes to displacement: angular first (2 per harmonic), what remains to the radial part. 0 = half. Total stays exactly h. With h=128 and 2 harmonics: 64 = default, 96 = 96 loc + 28 radial + 4 angular, 124 = no radial part (angular only), 128 with geo_harmonics=[] = location only. When set, the two *_scales keys must stay 0",
    "geo_loc_scales": "number of torus-lattice scales; 0 = h // 8",
    "geo_loc_lam_min_m": "finest location wavelength (m)",
    "geo_loc_lam_max_m": "coarsest location wavelength (m)",
    "geo_disp_scales": "radial displacement scales; 0 = (h/2 - 4) / 2",
    "geo_disp_lam_min_m": "finest displacement wavelength (m)",
    "geo_disp_lam_max_m": "coarsest displacement wavelength (m)",
    "geo_harmonics": "angular harmonics of the displacement channel (m=4: street-grid order)",
    "time_hour_harmonics": "cyclic C24 harmonics",
    "time_dow_harmonics": "cyclic C7 harmonics",
    "speed_scales": "SpeedGEO scales (dt and speed)",
    "speed_lo": "SpeedGEO shortest wavelength (log1p units)",
    "speed_hi": "SpeedGEO longest wavelength (log1p units)",
}
MODEL_HELP = {
    "hidden": "width of every sequence model (LSTM / Transformer / GNN)",
    "layers": "LSTM and Transformer depth",
    "heads": "attention heads (Transformer, GAT)",
    "dropout": "dropout inside the sequence model",
}


def template(subparsers: dict, commands=None) -> str:
    """Commented template generated from the CLI itself, so it is always
    complete and in sync with the code."""
    from .data import LOADERS
    out = ["# geomob configuration file (TOML).",
           "# Use with:  python -m geomob.cli <command> --config this_file.toml",
           "# Precedence: built-in defaults < [common] < [<command>] < command-line flags.",
           "# Unknown sections / keys are an error, so typos are caught.",
           "# Command options are listed commented-out at their defaults: uncomment",
           "# and edit only what you change (the run record saves ALL of them anyway).",
           "# A line '# key =' means the default is 'unset'.",
           "",
           "[paths]",
           "# dataset locations; these beat exported DATA_<NAME> variables"]
    out += [f'# {n} = "/path/to/{n}"' for n in LOADERS]
    out += ["", "[common]",
            "# applied to every command that has the option",
            '# device = "cuda"', "# seeds = [0, 1, 2]", "",
            _section("model", HP["model"], MODEL_HELP),
            _section("encoders", HP["encoders"], ENCODER_HELP)]
    out += ["[search_space]",
            "# used by `python -m geomob.cli search`: for any [encoders] or [model]",
            "# key, the list of values to try. Every combination is a trial (grid),",
            "# or a random subset of them (method = \"random\" in [search]).",
            "# geo_loc_lam_min_m = [10.0, 25.0, 50.0]",
            "# geo_loc_lam_max_m = [50000.0, 200000.0]",
            "# geo_disp_lam_min_m = [2.0, 5.0, 10.0]",
            "# geo_disp_lam_max_m = [5000.0, 20000.0]",
            "# geo_harmonics = [[1], [1, 4], [1, 2, 4]]",
            "# geo_loc_dims = [64, 80, 96, 112, 124]   # location share of h; 124 = no radial part",
            ""]
    for name, sp in subparsers.items():
        if commands and name not in commands:
            continue
        if name in ("config", "selfcheck", "datasets", "figures"):
            continue
        acts = _actions(sp)
        d, helps = {}, {}
        for k, a in acts.items():
            if k == "paths":            # use the [paths] section instead
                continue
            v = a.default
            d[k] = list(v) if isinstance(v, tuple) else v
            if a.help:
                helps[k] = " ".join(str(a.help).split())
        desc = (sp.description or "").strip()
        out.append((f"# {desc}\n" if desc else "") + _section(name, d, helps,
                                                             commented=True))
    return "\n".join(out)
