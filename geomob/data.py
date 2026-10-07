"""Datasets.

Synthetic generator (Experiment A) plus loaders for the corpora GPE uses, so
that head-to-head numbers are directly comparable to their Tables 4 and 10-18:
Porto, T-drive, GeoLife, Roma, AIS.

A Trajectory is (lonlat: (N,2) float64 degrees, t: (N,) float hours).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geo import enu_to_lonlat, lonlat_to_enu

# --------------------------------------------------------------------------
# Time convention: every loader returns hours since 1970-01-01 measured on the
# corpus's own **local wall clock**, never converted to UTC.
#
# This matters for the time channel and is easy to get wrong. Calling
# datetime.timestamp() on a naive timestamp silently applies whatever timezone
# the machine happens to have, so the same file yields different hour-of-day
# features on different machines -- and T-drive 08:00 (Beijing rush hour) would
# land at 00:00 on a UTC host. Mobility periodicity is a local-clock
# phenomenon, so we keep the wall clock and do the epoch arithmetic ourselves.
# --------------------------------------------------------------------------
_EPOCH = None


def _naive_hours(d) -> float:
    """Hours since 1970-01-01, treating a naive datetime as local wall clock."""
    import datetime as _dt
    global _EPOCH
    if _EPOCH is None:
        _EPOCH = _dt.datetime(1970, 1, 1)
    if d.tzinfo is not None:          # keep the local reading, drop the offset
        d = d.replace(tzinfo=None)
    return (d - _EPOCH).total_seconds() / 3600.0


# GeoLife's raw mode strings, collapsed onto the classes the literature
# actually reports. Rail modes are merged because train/subway/railway are
# only separable with timetable data, and "run" is folded into walk.
MODE_CANON = {
    "walk": "walk", "run": "walk", "bike": "bike",
    "bus": "bus", "taxi": "car", "car": "car",
    "subway": "rail", "train": "rail", "railway": "rail",
    "airplane": "other", "boat": "other", "motorcycle": "other",
}


@dataclass
class Traj:
    lonlat: np.ndarray
    t: np.ndarray               # hours since the Unix epoch, every corpus
    uid: str | None = None      # person, when the corpus identifies one
    label: str | None = None    # class label, e.g. GeoLife travel mode

    def __len__(self):
        return len(self.lonlat)

    def xy(self, ref=None):
        return lonlat_to_enu(self.lonlat, ref)[0]

    @property
    def hour(self):
        return self.t % 24.0

    @property
    def dow(self):
        """Day of week as a float; consistent across corpora because every
        loader returns hours since the Unix epoch (which began on a Thursday)."""
        return (self.t // 24.0) % 7.0


# --------------------------------------------------------------------------
# synthetic: biased random walks toward latent hubs
# --------------------------------------------------------------------------
def make_synthetic(n: int = 2000, n_steps: int = 40, n_hubs: int = 6,
                   city_km: float = 12.0, centre=(0.0, 41.15),
                   step_m: float = 180.0, bias: float = 0.35,
                   noise_m: float = 8.0, seed: int = 0, hard: bool = False,
                   grid_deg: float | None = 20.0, commute_hours=(8.0, 17.5)):
    """Trajectories with a known generative process and a known destination.

    Centre defaults near Porto's latitude so the synthetic and real regimes
    are comparable; `centre` is what the latitude stress test varies.
    """
    if hard:
        # The default preset saturates MRR at ~1.0: distinct hubs in a large
        # area make trajectories trivially separable, so Experiments B and G
        # cannot discriminate between tokenizers on it. The hard preset packs
        # few hubs into a small area with short, noisy traces, which is what
        # makes the ranking task informative without real data.
        n_hubs, city_km, n_steps = 2, 3.0, 20
        noise_m, bias, step_m = 40.0, 0.6, 120.0
    # Two structures the earlier generator lacked, both required for the
    # anomaly experiment to mean anything:
    #   `grid_deg`    a preferred street-grid orientation, so that rotating a
    #                 trip is actually anomalous rather than just another
    #                 sample from an isotropic distribution;
    #   `commute_hours`  bimodal departure times on weekdays, so that a trip
    #                 at 03:00 is actually unusual.
    # Set grid_deg=None / commute_hours=None to recover the isotropic,
    # uniform-in-time corpus.
    rng = np.random.default_rng(seed)
    half = city_km * 500.0
    hubs = rng.uniform(-half, half, size=(n_hubs, 2))
    grid = None if grid_deg is None else np.deg2rad(grid_deg)
    trajs, targets = [], []
    for _ in range(n):
        hub = hubs[rng.integers(n_hubs)]
        p = rng.uniform(-half, half, size=2)
        start = p.copy()
        heading = rng.uniform(-np.pi, np.pi)
        path = [p.copy()]
        for _ in range(n_steps):
            to_hub = hub - p
            to_hub = to_hub / (np.linalg.norm(to_hub) + 1e-9)
            d = np.array([np.cos(heading), np.sin(heading)])
            d = (1 - bias) * d + bias * to_hub
            d = d / (np.linalg.norm(d) + 1e-9)
            heading = np.arctan2(d[1], d[0]) + rng.normal(0, 0.25)
            if grid is not None:      # snap toward the nearest grid axis
                rel = heading - grid
                heading = grid + np.round(rel / (np.pi / 2)) * (np.pi / 2) \
                    + 0.12 * (rel - np.round(rel / (np.pi / 2)) * (np.pi / 2))
            p = p + step_m * np.array([np.cos(heading), np.sin(heading)])
            path.append(p.copy())
        path = np.asarray(path)
        obs = path[:-1] + rng.normal(0, noise_m, size=path[:-1].shape)
        if commute_hours is None:
            t0 = rng.uniform(0, 24 * 7)
        else:                         # weekday commute peaks, quiet at night
            day = rng.integers(0, 7)
            peak = commute_hours[rng.integers(len(commute_hours))]
            t0 = 24 * day + np.clip(peak + rng.normal(0, 0.9), 0, 23.9)
        trajs.append(Traj(enu_to_lonlat(obs, np.asarray(centre, float)),
                          t0 + np.arange(len(obs)) / 60.0))
        targets.append(path[-1] - path[-2])  # next displacement, equivariant
    return trajs, np.asarray(targets), hubs


# --------------------------------------------------------------------------
# real loaders
# --------------------------------------------------------------------------
def load_porto(path: str, max_traj: int | None = 20000, min_len: int = 20,
               max_len: int = 200) -> list[Traj]:
    """Kaggle 'Taxi Service Trajectory' train.csv (POLYLINE column, 15 s fixes)."""
    import csv
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                pts = json.loads(row["POLYLINE"])
            except Exception:
                continue
            if len(pts) < min_len:
                continue
            pts = np.asarray(pts[:max_len], dtype=np.float64)
            t0 = float(row.get("TIMESTAMP", 0)) / 3600.0
            out.append(Traj(pts, t0 + np.arange(len(pts)) * 15.0 / 3600.0,
                            uid=row.get("TAXI_ID")))
            if max_traj and len(out) >= max_traj:
                break
    return out


def load_tdrive(root: str, max_traj: int | None = 20000, gap_min: float = 30.0,
                min_len: int = 20, max_len: int = 200) -> list[Traj]:
    """T-drive release: one txt per taxi, lines 'id,datetime,lon,lat'."""
    import datetime as _dt
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"T-drive path does not exist: {root}")
    # recursive, for the same reason as GeoLife: the release unpacks to
    # 'taxi_log_2008_by_id/*.txt' and pointing one level above it should work
    files = sorted(root.rglob("*.txt")) if root.is_dir() else [root]
    if not files:
        raise FileNotFoundError(
            f"T-drive: no .txt files found anywhere under {root}. The release "
            f"unpacks to 'taxi_log_2008_by_id/' holding one .txt per taxi; "
            f"point --path at that directory or any level above it.")
    out = []
    for fp in files:
        rows = []
        for line in open(fp, errors="ignore"):
            f = line.strip().split(",")
            if len(f) < 4:
                continue
            try:
                ts = _dt.datetime.strptime(f[1], "%Y-%m-%d %H:%M:%S")
                rows.append((_naive_hours(ts), float(f[2]), float(f[3])))
            except ValueError:
                continue
        # one file per taxi: its name is the driver id (needed for
        # trajectory-user linking and for per-user splits)
        out += _split_on_gaps(rows, gap_min / 60.0, min_len, max_len,
                              uid=fp.stem)
        if max_traj and len(out) >= max_traj:
            return out[:max_traj]
    return out


# days between 1899-12-30 (the .plt epoch) and 1970-01-01
_PLT_EPOCH_OFFSET_DAYS = 25569.0


def load_geolife(root: str, max_traj: int | None = 20000, min_len: int = 20,
                 max_len: int = 200, gap_min: float = 20.0, overlap: int = 100,
                 labelled_only: bool = False, verbose: bool = True,
                 utc_offset_h: float = 8.0) -> list[Traj]:
    """GeoLife .plt files, with user ids and transportation-mode labels.

    **Time zone.** GeoLife stores every timestamp (.plt and labels.txt) in
    GMT, while the users lived in Beijing (UTC+8). Hour-of-day and day-of-week
    must be on the local clock -- otherwise a 08:00 commute reads as 00:00 and
    a trip moved to "3 a.m." lands at 11:00 local, an ordinary hour -- so both
    are shifted by `utc_offset_h` (default +8; pass 0 for the raw GMT clock).

    The canonical layout is ``<user id>/Trajectory/*.plt`` with an optional
    ``<user id>/labels.txt`` beside it, for 73 of the 182 users. But the
    download unpacks to ``Geolife Trajectories 1.3/Data/<user id>/...``, and
    people reasonably point this at any level of that, so the loader searches
    for ``*.plt`` **recursively** rather than assuming a fixed depth. The user
    id is taken from the directory holding ``Trajectory/``, whatever its
    depth; if a .plt sits somewhere without a ``Trajectory`` parent, its own
    parent directory names the user.

    Times become hours since the Unix epoch on the local (Beijing) clock (the
    .plt field is days since 1899-12-30, GMT), so hour-of-day and day-of-week
    line up with every other loader here.

    Raises rather than returning an empty list when nothing is found -- a
    silent [] here surfaces much later as an unhelpful "need at least one
    array to concatenate" from whichever experiment consumed it.
    """
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"GeoLife path does not exist: {root}")

    plts = sorted(root.rglob("*.plt"))
    if not plts:
        raise FileNotFoundError(_no_plt_message(root))

    def user_of(fp: Path) -> tuple[str, Path]:
        """(user id, user directory) for a trajectory file at any depth."""
        d = fp.parent
        if d.name.lower() == "trajectory":
            return d.parent.name, d.parent
        return d.name, d

    # Group by user BEFORE applying any cap. Taking a prefix of the file list
    # instead would stop partway through the first few users -- with the
    # default cap that yields ~11 of GeoLife's 182 users and, because the 73
    # labelled users are mostly numbered well above those, zero mode labels.
    # It would also quietly wreck `--split user`, which needs many users.
    by_user: dict[str, list[Path]] = {}
    udirs: dict[str, Path] = {}
    for fp in plts:
        uid, udir = user_of(fp)
        by_user.setdefault(uid, []).append(fp)
        udirs[uid] = udir

    labels = {uid: _geolife_labels(udirs[uid] / "labels.txt", utc_offset_h)
              for uid in by_user}
    n_users_total = len(by_user)
    n_labelled_users = sum(1 for v in labels.values() if v)
    users = sorted(by_user)
    if labelled_only:
        users = [u for u in users if labels[u]]
        if not users:
            raise ValueError(
                f"GeoLife: labelled_only=True but no labels.txt was found "
                f"under {root}. GeoLife ships mode labels for 73 of its 182 "
                f"users; check the archive is complete.")

    # Spread the budget evenly over users rather than exhausting them in turn.
    quota = None
    if max_traj:
        quota = max(max_traj // max(len(users), 1), 1)

    out, n_lab, truncated = [], 0, False
    for uid in users:
        udir, spans, got = udirs[uid], labels[uid], 0
        for fp in sorted(by_user[uid]):
            if quota is not None and got >= quota:
                truncated = True
                break
            rows = []
            for line in list(open(fp, errors="ignore"))[6:]:
                f = line.strip().split(",")
                if len(f) < 5:
                    continue
                try:
                    rows.append(((float(f[4]) - _PLT_EPOCH_OFFSET_DAYS) * 24.0
                                 + utc_offset_h,
                                 float(f[1]), float(f[0])))      # (t, lon, lat)
                except ValueError:
                    continue
            segs = _split_on_gaps(rows, gap_min / 60.0, min_len, max_len,
                                  uid=uid, spans=spans, overlap=overlap)
            n_lab += sum(1 for s_ in segs if s_.label)
            out += segs
            got += len(segs)

    if labelled_only:
        out = [t for t in out if t.label]
    if not out:
        raise ValueError(
            f"GeoLife: found {len(plts)} .plt files under {root} but no usable "
            f"trajectory survived filtering (min_len={min_len}; try lowering "
            f"it, or raising gap_min={gap_min:.0f} min which currently splits "
            f"the traces).")

    if verbose:
        modes = {}
        for t in out:
            if t.label:
                modes[t.label] = modes.get(t.label, 0) + 1
        seen = len({t.uid for t in out})
        print(f"GeoLife: {len(out)} trajectories, {seen}/{n_users_total} users, "
              f"{n_lab} mode-labelled"
              + (f"  {modes}" if modes else ""))
        if truncated:
            print(f"  (max_traj={max_traj} reached; sampled up to {quota} "
                  f"trajectories per user so all {len(users)} users are "
                  f"represented -- raise max_traj or set it to None for the "
                  f"whole corpus)")
        if n_lab == 0 and n_labelled_users and not labelled_only:
            print(f"  note: {n_labelled_users} users have labels.txt but no "
                  f"window matched a labelled span; labels cover only part of "
                  f"each user's data")
    return out


def _no_plt_message(root: Path) -> str:
    """A diagnostic that says what was actually there, not just that it failed."""
    try:
        entries = sorted(p.name for p in root.iterdir())[:12]
    except OSError:
        entries = []
    hint = ""
    # the usual cause: pointed one level above the extracted archive
    for p in root.iterdir() if root.is_dir() else []:
        if p.is_dir() and any(p.rglob("*.plt")):
            hint = (f"\n  A subdirectory does contain .plt files: try "
                    f"--path '{p}'")
            break
    return (f"GeoLife: no .plt files found anywhere under {root}\n"
            f"  contents: {entries}{hint}\n"
            f"  Expected the extracted archive, whose layout is\n"
            f"    <somewhere>/Data/<user id>/Trajectory/*.plt\n"
            f"  Point --path (or DATA_GEOLIFE) at any level at or above "
            f"'Data' and it will be found.")


def _geolife_labels(path: Path, utc_offset_h: float = 0.0):
    """Parse labels.txt into (t_start_hours, t_end_hours, canonical mode).

    labels.txt is in GMT like the .plt files; the same offset is applied so
    the spans stay aligned with the shifted trajectory times."""
    import datetime as _dt
    if not path.exists():
        return None
    spans = []
    for line in list(open(path, errors="ignore"))[1:]:      # skip header
        f = line.strip().split("\t")
        if len(f) < 3:
            continue
        try:
            a = _dt.datetime.strptime(f[0].strip(), "%Y/%m/%d %H:%M:%S")
            b = _dt.datetime.strptime(f[1].strip(), "%Y/%m/%d %H:%M:%S")
        except ValueError:
            continue
        mode = MODE_CANON.get(f[2].strip().lower())
        if mode:
            spans.append((_naive_hours(a) + utc_offset_h,
                          _naive_hours(b) + utc_offset_h, mode))
    return spans or None


def load_roma(path: str, max_traj: int | None = 20000, gap_min: float = 1.0,
              min_len: int = 20, max_len: int = 200, overlap: int = 100,
              latlon_order: bool = True, verbose: bool = True) -> list[Traj]:
    """Roma taxi (CRAWDAD roma/taxi), semicolon-separated:

        id;date_time;POINT(lat lon)

    The WKT in this release lists **latitude first**, which is the opposite of
    the usual convention, so `latlon_order=True` is the default. The loader
    prints the resulting bounding box: Rome is near lon 12.5, lat 41.9, so a
    swapped file is obvious at a glance.
    """
    import datetime as _dt
    import re as _re

    by_taxi: dict[str, list] = {}
    pat = _re.compile(r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)")
    with open(path, errors="ignore") as fh:
        for line in fh:
            f = line.strip().split(";")
            if len(f) < 3:
                continue
            m = pat.search(f[2])
            if not m:
                continue
            a, b = float(m.group(1)), float(m.group(2))
            lat, lon = (a, b) if latlon_order else (b, a)
            ts = f[1].strip()
            try:                        # '2014-02-01 00:00:00.739166+01'
                t = _naive_hours(_dt.datetime.fromisoformat(ts))
            except ValueError:
                try:
                    t = _naive_hours(_dt.datetime.strptime(ts[:19],
                                                           "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
            by_taxi.setdefault(f[0].strip(), []).append((t, lon, lat))

    out = []
    for tid, rows in by_taxi.items():
        out += _split_on_gaps(rows, gap_min / 60.0, min_len, max_len,
                              uid=tid, overlap=overlap)
        if max_traj and len(out) >= max_traj:
            out = out[:max_traj]
            break
    if verbose and out:
        ll = np.concatenate([t.lonlat for t in out])
        print(f"Roma: {len(out)} trajectories, {len({t.uid for t in out})} taxis, "
              f"bbox lon [{ll[:,0].min():.3f}, {ll[:,0].max():.3f}] "
              f"lat [{ll[:,1].min():.3f}, {ll[:,1].max():.3f}]  "
              f"(Rome is near lon 12.5, lat 41.9)")
    return out


def load_ais(path: str, max_traj: int | None = 20000, gap_h: float = 1.0,
             min_len: int = 20, max_len: int = 200) -> list[Traj]:
    """MarineCadastre AIS csv: MMSI, BaseDateTime, LAT, LON, ...

    The road-free, orientation-free dataset -- the natural home for the
    rotation-equivariance argument, and for cross-mode transfer.
    """
    import csv
    import datetime as _dt
    by_ship: dict[str, list] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                # AIS spans many longitudes, so there is no single local
                # clock; these stay UTC. Time-of-day features are therefore
                # not comparable with the city corpora -- fine for the spatial
                # experiments, worth stating in any temporal one.
                ts = _dt.datetime.fromisoformat(row["BaseDateTime"])
                by_ship.setdefault(row["MMSI"], []).append(
                    (_naive_hours(ts), float(row["LON"]), float(row["LAT"])))
            except Exception:
                continue
    out = []
    for mmsi, rows in by_ship.items():
        out += _split_on_gaps(rows, gap_h, min_len, max_len, uid=mmsi)
        if max_traj and len(out) >= max_traj:
            return out[:max_traj]
    return out


def _split_on_gaps(rows, gap_h, min_len, max_len, uid=None, spans=None,
                   overlap: int = 0):
    """rows: iterable of (t_hours, lon, lat). Splits on temporal gaps, then
    windows to `max_len` with `overlap` points of overlap (GPE uses 100).

    `spans` is an optional list of (t_start, t_end, label) used to attach a
    class label to each window; a window takes the label covering the majority
    of its duration, and is left unlabelled if none covers it.
    """
    if not rows:
        return []
    rows = sorted(rows)
    t = np.array([r[0] for r in rows])
    ll = np.array([[r[1], r[2]] for r in rows])
    cuts = np.flatnonzero(np.diff(t) > gap_h) + 1
    stride = max(max_len - overlap, 1)
    out = []
    for seg in np.split(np.arange(len(t)), cuts):
        for s in range(0, max(len(seg) - min_len + 1, 1), stride):
            w = seg[s:s + max_len]
            if len(w) >= min_len:
                out.append(Traj(ll[w], t[w], uid,
                                _label_for(t[w], spans) if spans else None))
    return out


def _label_for(tw, spans):
    """Label covering the largest share of this window's time span."""
    best, best_cov = None, 0.0
    lo, hi = tw[0], tw[-1]
    for s0, s1, lab in spans:
        cov = max(0.0, min(hi, s1) - max(lo, s0))
        if cov > best_cov:
            best, best_cov = lab, cov
    return best if best_cov > 0.5 * max(hi - lo, 1e-9) else None


def _load_csv(path, **kw):
    from .csvdata import load_csv
    return load_csv(path, **kw)


LOADERS = {"porto": load_porto, "tdrive": load_tdrive,
           "geolife": load_geolife, "roma": load_roma, "ais": load_ais,
           "csv": _load_csv}
DATASETS = ("synthetic", "synthetic_hard") + tuple(LOADERS)


def load_dataset(name: str, path: str | None = None, **kw) -> list[Traj]:
    """`name='synthetic'` needs no path; the rest need DATA_<NAME> or `path`."""
    if name in ("synthetic", "synthetic_hard"):
        return make_synthetic(hard=(name == "synthetic_hard"), **kw)[0]
    if name not in LOADERS:
        raise KeyError(f"unknown dataset '{name}'; available: {DATASETS}")
    path = path or os.environ.get(f"DATA_{name.upper()}")
    if not path:
        raise FileNotFoundError(
            f"no path given (set DATA_{name.upper()} or pass a path)")
    if not os.path.exists(os.path.expanduser(path)):
        raise FileNotFoundError(f"path does not exist: {path}")
    trajs = LOADERS[name](os.path.expanduser(path), **kw)
    # Never hand an empty corpus downstream. Without this the failure surfaces
    # much later as "need at least one array to concatenate" from whichever
    # experiment consumed it, which says nothing about the real cause.
    if not trajs:
        raise ValueError(
            f"dataset '{name}' loaded 0 trajectories from {path}. Check the "
            f"path points at the data, and that the length filters are not "
            f"excluding everything (min_len / gap thresholds).")
    return trajs


# --------------------------------------------------------------------------
# running with whatever data is on disk
# --------------------------------------------------------------------------
def dataset_status(name: str, path: str | None = None) -> tuple[bool, str]:
    """Cheap availability check (no loading): (available, where-or-why)."""
    if name in ("synthetic", "synthetic_hard"):
        return True, "generated"
    if name not in LOADERS:
        return False, f"unknown dataset name (known: {', '.join(DATASETS)})"
    p = path or os.environ.get(f"DATA_{name.upper()}")
    if not p:
        return False, f"no path given (set DATA_{name.upper()} or pass a path)"
    if not os.path.exists(os.path.expanduser(p)):
        return False, f"path does not exist: {p}"
    return True, p


def load_available(names, paths=None, kw_for=None, required=1,
                   what="this experiment"):
    """Load every dataset that can be loaded; skip the rest *loudly*.

    Returns (loaded, skipped): `loaded` maps name -> list[Traj] in the order
    requested, `skipped` maps name -> reason. A dataset is skipped when its
    path is missing, the path does not exist, the loader finds nothing, or the
    loader raises. Only if fewer than `required` datasets survive is this an
    error, and the error lists every reason so the fix is obvious.
    """
    paths = paths or {}
    loaded, skipped = {}, {}
    for name in names:
        kw = kw_for(name) if kw_for else {}
        try:
            loaded[name] = load_dataset(name, paths.get(name), **kw)
        except (FileNotFoundError, NotADirectoryError, KeyError, ValueError,
                OSError, RuntimeError) as e:
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            skipped[name] = msg[:300]
            print(f"  !! SKIPPED dataset '{name}': {skipped[name]}")
    if skipped:
        print(f"  -> continuing {what} with {len(loaded)}/{len(names)} "
              f"dataset(s): {', '.join(loaded) or 'none'}; skipped: "
              f"{', '.join(skipped)}")
    if len(loaded) < required:
        detail = "\n".join(f"    {n}: {r}" for n, r in skipped.items())
        raise FileNotFoundError(
            f"{what} needs at least {required} dataset(s) but only "
            f"{len(loaded)} could be loaded.\n{detail}\n"
            f"Point at the data with --paths name=/path (or DATA_<NAME>), or "
            f"run `python -m geomob.cli datasets` to see what is found.")
    return loaded, skipped


# --------------------------------------------------------------------------
# augmentations (GPE's positive-pair recipe, kept identical for comparability)
# --------------------------------------------------------------------------
def augment(traj: Traj, rng, drop: float = 0.1, jitter_m: float = 20.0) -> Traj:
    keep = rng.random(len(traj)) > drop
    keep[0] = keep[-1] = True
    xy, ref = lonlat_to_enu(traj.lonlat)
    xy = xy[keep] + rng.normal(0, jitter_m / 2.0, size=(keep.sum(), 2))
    return Traj(enu_to_lonlat(xy, ref), traj.t[keep], traj.uid)


def augment_geo(traj: Traj, rng, drop: float = 0.1, jitter_m: float = 20.0,
                rot: bool = True, trans_m: float = 0.0,
                tau_h: float = 0.0) -> Traj:
    """Positives that are valid *by the group structure*, not by assumption.

    Any g in G maps a trajectory to one the tokenizer is guaranteed to encode
    equivariantly, so g.T is a provably valid contrastive positive. This is
    the augmentation whose payoff Experiment D isolates: reduced sensitivity
    to augmentation hyperparameters, because the invariance is constructed
    rather than learned.
    """
    from .geo import act_rotate, act_translate
    base = augment(traj, rng, drop, jitter_m)
    xy, ref = lonlat_to_enu(base.lonlat)
    if rot:
        xy = act_rotate(xy, rng.uniform(-np.pi, np.pi))
    if trans_m:
        xy = act_translate(xy, rng.uniform(-trans_m, trans_m, 2))
    t = base.t + (rng.uniform(-tau_h, tau_h) if tau_h else 0.0)
    return Traj(enu_to_lonlat(xy, ref), t, traj.uid)


def odd_even_split(traj: Traj) -> tuple[Traj, Traj]:
    """t2vec / GPE evaluation protocol: sub-sample to two half-rate copies."""
    return (Traj(traj.lonlat[0::2], traj.t[0::2], traj.uid),
            Traj(traj.lonlat[1::2], traj.t[1::2], traj.uid))


def space_shift(trajs: list[Traj], src_box, dst_box) -> list[Traj]:
    """GPE's SS protocol: linear bounding-box remap, so that non-transferable
    baselines can be evaluated cross-city at all."""
    (slo, shi), (dlo, dhi) = src_box, dst_box
    scale = (dhi - dlo) / np.maximum(shi - slo, 1e-9)
    return [Traj((tr.lonlat - slo) * scale + dlo, tr.t, tr.uid) for tr in trajs]


def relocate_city(trajs: list[Traj], lat: float | None = None,
                  lon: float | None = None, rotate_deg: float = 0.0,
                  shift_km: tuple = (0.0, 0.0), ref=None) -> list[Traj]:
    """Move a whole corpus as one rigid body, preserving every metric relation.

    All trajectories are projected to metres about the corpus centroid (`ref`),
    optionally rotated about it by `rotate_deg`, and laid back down around a
    new centre: the same city, on the ground, somewhere else on the globe
    and/or with its street grid turned. Distances, angles between trips,
    durations and the relative position of every trip are unchanged; only the
    (lon, lat) coordinates differ.

    `lat` / `lon` set the new centre (default: unchanged); `shift_km` moves it
    east / north by that many kilometres instead.
    """
    from .geo import R_EARTH, enu_to_lonlat, lonlat_to_enu, rot2
    if ref is None:
        ref = np.concatenate([t.lonlat for t in trajs]).mean(0)
    ref = np.asarray(ref, dtype=np.float64)
    new = ref.copy()
    if lon is not None:
        new[0] = lon
    if lat is not None:
        new[1] = lat
    new[1] += np.rad2deg(shift_km[1] * 1000.0 / R_EARTH)
    new[0] += np.rad2deg(shift_km[0] * 1000.0
                         / (R_EARTH * np.cos(np.deg2rad(new[1]))))
    R = rot2(np.deg2rad(rotate_deg)).T
    out = []
    for t in trajs:
        xy = lonlat_to_enu(t.lonlat, ref)[0]
        if rotate_deg:
            xy = xy @ R
        out.append(Traj(enu_to_lonlat(xy, new), t.t, t.uid, t.label))
    return out


def bbox(trajs: list[Traj]):
    ll = np.concatenate([t.lonlat for t in trajs])
    return np.min(ll, 0), np.max(ll, 0)
