"""Loading your own GPS commute CSV.

Written against the schema

    uid, DAY, HH24, lat, lng, SPEED, HEADING, QUALITY,
    ID_PANELSESSION, DELTAPOS, datetime, tid, trip_id, trip_id_segment, user

but every column name is configurable and every one of them is optional except
latitude and longitude. Nothing here assumes that exact header: pass a
`CsvSchema` (or let `infer_schema` guess) and the loader adapts.

Three things about this kind of data that change how the experiments should be
run, and that the synthetic corpus does not exercise:

1. **Trips repeat.** A commute corpus contains the same journey many times per
   user. Splitting trips at random puts near-duplicates of the same commute in
   both train and test, so scores measure memorisation. `user_disjoint_split`
   is the default remedy; `trip_disjoint_split` is the weaker fallback.
2. **Time finally matters.** DAY and HH24 are exactly the C24 x C7 coordinates
   the cyclic-time GENEO is defined on, so the time channel is testable here
   in a way it is not on the vehicle corpora GPE uses. Mind the leak warning
   in the similarity benchmark, though: use time on a downstream task.
3. **HEADING is a free ground truth** for the equivariant angular channel.
   `validate_heading` checks the recovered psi against the recorded bearing,
   which catches frame and sign errors before they become confusing results.

Human commute traces are personal data. Beyond the methodological point, note
that user-disjoint splitting is also the setting in which any claim about
generalisation to *new people* is meaningful at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Traj
from .geo import lonlat_to_enu


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
@dataclass
class CsvSchema:
    """Column names. Candidate lists are tried in order; the first present wins."""

    lat: str = "lat"
    lon: str = "lng"
    time: tuple[str, ...] = ("datetime",)
    day: str | None = "DAY"
    hour: str | None = "HH24"
    group: tuple[str, ...] = ("trip_id_segment", "trip_id", "tid")
    user: tuple[str, ...] = ("user", "uid")
    quality: str | None = "QUALITY"
    speed: str | None = "SPEED"
    heading: str | None = "HEADING"

    def pick(self, cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None


def infer_schema(path_or_df) -> CsvSchema:
    """Best-effort guess for headers that differ from the default."""
    cols = (path_or_df.columns if hasattr(path_or_df, "columns")
            else pd.read_csv(path_or_df, nrows=1).columns)
    low = {c.lower(): c for c in cols}

    def find(*names):
        for n in names:
            if n in low:
                return low[n]
        return None

    s = CsvSchema()
    s.lat = find("lat", "latitude", "y") or "lat"
    s.lon = find("lng", "lon", "long", "longitude", "x") or "lng"
    s.time = tuple(c for c in (find("datetime", "timestamp", "time", "ts"),) if c)
    s.day = find("day", "date", "dow")
    s.hour = find("hh24", "hour", "hh")
    s.group = tuple(c for c in (find("trip_id_segment"), find("trip_id"),
                                find("tid"), find("segment_id")) if c)
    s.user = tuple(c for c in (find("user"), find("uid"), find("user_id")) if c)
    s.quality = find("quality", "accuracy", "hdop")
    s.speed = find("speed")
    s.heading = find("heading", "bearing", "course")
    return s


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _to_hours(ts: "pd.Series") -> np.ndarray:
    """datetime64 of any resolution -> float hours since the Unix epoch."""
    return ((ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
            / 3600.0)



def load_csv(path: str, schema: CsvSchema | None = None, min_len: int = 20,
             max_len: int = 200, overlap: int = 100, gap_min: float = 10.0,
             max_traj: int | None = None, min_quality: float | None = None,
             group_col: str | None = None, max_speed_kmh: float = 250.0,
             verbose: bool = True) -> list[Traj]:
    """Read a trajectory CSV into a list of `Traj`.

    Defaults mirror GPE's preprocessing (cut at `max_len` with `overlap`,
    discard below `min_len`) so results stay comparable, but `gap_min` is 10
    minutes rather than their 30, because personal commute fixes are usually
    denser than taxi logs. Check the recommendation from `inspect_csv` before
    accepting it.

    `max_speed_kmh` drops physically impossible jumps, which in practice are
    coordinate glitches rather than real movement; leaving them in produces
    enormous spurious displacements that dominate the radial channel.
    """
    df = pd.read_csv(path)
    schema = schema or infer_schema(df)
    cols = set(df.columns)

    lat, lon = schema.lat, schema.lon
    if lat not in cols or lon not in cols:
        raise KeyError(f"latitude/longitude columns not found; got {sorted(cols)}. "
                       f"Pass CsvSchema(lat=..., lon=...).")

    gcol = group_col or schema.pick(cols, schema.group)
    if gcol is None:
        raise KeyError("no trip-grouping column found; pass group_col=...")
    ucol = schema.pick(cols, schema.user)
    tcol = schema.pick(cols, schema.time)

    # ---- time in hours since epoch, however it is available
    if tcol is not None:
        ts = pd.to_datetime(df[tcol], errors="coerce", format="mixed")
        # Resolution-independent: pandas >= 3.0 parses to datetime64[us], not
        # [ns], so dividing a raw int64 view by a fixed constant is wrong by
        # 1000x and silently turns every step into a speed outlier.
        t_hours = _to_hours(ts)
        bad = ts.isna().to_numpy()
    else:
        bad = np.zeros(len(df), dtype=bool)
        t_hours = None
    if t_hours is None or bad.all():
        if schema.day in cols and schema.hour in cols:
            day = pd.to_datetime(df[schema.day], errors="coerce")
            t_hours = (_to_hours(day)
                       + pd.to_numeric(df[schema.hour], errors="coerce").to_numpy())
            bad = ~np.isfinite(t_hours)
        else:
            raise KeyError("no usable time column (datetime, or DAY + HH24)")

    df = df.assign(_t=t_hours)
    keep = (~bad) & np.isfinite(df[lat].to_numpy()) & np.isfinite(df[lon].to_numpy())
    if min_quality is not None and schema.quality in cols:
        keep &= pd.to_numeric(df[schema.quality],
                              errors="coerce").to_numpy() >= min_quality
    dropped_rows = int((~keep).sum())
    df = df[keep]

    out, n_dup, n_speed = [], 0, 0
    for _, g in df.groupby(gcol, sort=False):
        g = g.sort_values("_t")
        ll = g[[lon, lat]].to_numpy(dtype=np.float64)
        t = g["_t"].to_numpy(dtype=np.float64)
        uid = str(g[ucol].iloc[0]) if ucol else None

        # drop repeated fixes at the same instant
        first = np.r_[True, np.diff(t) > 0]
        n_dup += int((~first).sum())
        ll, t = ll[first], t[first]
        if len(t) < min_len:
            continue

        # split on temporal gaps and on impossible speeds
        xy, _ = lonlat_to_enu(ll)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dt = np.diff(t)
        kmh = step / 1000.0 / np.maximum(dt, 1e-6)
        n_speed += int((kmh > max_speed_kmh).sum())
        cut = np.flatnonzero((dt > gap_min / 60.0) | (kmh > max_speed_kmh)) + 1

        for seg in np.split(np.arange(len(t)), cut):
            stride = max(max_len - overlap, 1)
            for s in range(0, max(len(seg) - min_len + 1, 1), stride):
                w = seg[s:s + max_len]
                if len(w) >= min_len:
                    out.append(Traj(ll[w], t[w], uid=uid))
            if max_traj and len(out) >= max_traj:
                break
        if max_traj and len(out) >= max_traj:
            break

    if verbose:
        print(f"loaded {len(out)} trajectories from {path}\n"
              f"  grouping column : {gcol}"
              f"{'  user column : ' + ucol if ucol else ''}\n"
              f"  dropped rows    : {dropped_rows} invalid, {n_dup} duplicate "
              f"timestamps, {n_speed} speed outliers")
    return out[:max_traj] if max_traj else out


# --------------------------------------------------------------------------
# inspection: read this before choosing hyperparameters
# --------------------------------------------------------------------------
def inspect_csv(path: str, schema: CsvSchema | None = None, sample: int = 400,
                **load_kw) -> dict:
    """Corpus statistics, plus concrete hyperparameter recommendations.

    The scale parameters of every encoder here (lambda ladders, epsilon, tile
    size) are physical lengths, so they should be set from the data rather
    than copied from a paper written about taxis.
    """
    trajs = load_csv(path, schema, **load_kw)
    ll = np.concatenate([t.lonlat for t in trajs])
    ref = ll.mean(0)
    lens = np.array([len(t) for t in trajs])
    steps, dts = [], []
    for t in trajs[:sample]:
        xy = t.xy(ref)
        steps.append(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        dts.append(np.diff(t.t) * 3600.0)
    steps = np.concatenate(steps)
    dts = np.concatenate(dts)
    xy_all, _ = lonlat_to_enu(ll, ref)
    extent = xy_all.max(0) - xy_all.min(0)
    users = {t.uid for t in trajs if t.uid is not None}
    hours = np.concatenate([t.t % 24 for t in trajs[:sample]])

    med_step = float(np.median(steps))
    diag = float(np.hypot(*extent))
    rec = {
        "DisplacementGEO.lam_min": round(max(med_step / 4, 5.0), 1),
        "DisplacementGEO.lam_max": round(diag, -1),
        "TorusLattice.lams": [float(round(x, -1)) for x in
                              _ladder(max(med_step, 50.0), diag * 1.5)],
        "PatchworkLattice.tile_m": round(diag / 8, -2),
        "load_csv.gap_min": round(float(np.percentile(dts, 99)) / 60 * 3, 1),
        "GPE.eps_over_2pi": _eps_for_wavelength(max(med_step, 15.0)),
    }
    res = {
        "n_trajectories": len(trajs),
        "n_users": len(users) or None,
        "length_points": {"median": float(np.median(lens)),
                          "p5": float(np.percentile(lens, 5)),
                          "p95": float(np.percentile(lens, 95))},
        "step_m": {"median": med_step,
                   "p5": float(np.percentile(steps, 5)),
                   "p95": float(np.percentile(steps, 95))},
        "sampling_interval_s": {"median": float(np.median(dts)),
                                "p95": float(np.percentile(dts, 95))},
        "extent_km": [round(extent[0] / 1000, 2), round(extent[1] / 1000, 2)],
        "lat_range": [float(ll[:, 1].min()), float(ll[:, 1].max())],
        "lon_range": [float(ll[:, 0].min()), float(ll[:, 0].max())],
        "hour_of_day_hist": np.histogram(hours, bins=24, range=(0, 24))[0].tolist(),
        "recommended": rec,
    }
    _print_inspection(res)
    return res


def _ladder(lo, hi, n=6):
    return list(lo * (hi / lo) ** (np.arange(n) / (n - 1)))


def _eps_for_wavelength(target_m: float) -> float:
    """epsilon/2pi whose finest GPE wavelength is about `target_m` on the ground."""
    from .geo import R_EARTH
    return float(target_m / (2 * np.pi * R_EARTH))


def _print_inspection(r):
    print("\n--- corpus ---")
    print(f"  {r['n_trajectories']} trajectories"
          + (f", {r['n_users']} users" if r["n_users"] else ""))
    print(f"  length   {r['length_points']['median']:.0f} pts "
          f"(p5 {r['length_points']['p5']:.0f}, p95 {r['length_points']['p95']:.0f})")
    print(f"  step     {r['step_m']['median']:.1f} m median, "
          f"p95 {r['step_m']['p95']:.1f} m")
    print(f"  interval {r['sampling_interval_s']['median']:.1f} s median, "
          f"p95 {r['sampling_interval_s']['p95']:.1f} s")
    print(f"  extent   {r['extent_km'][0]} x {r['extent_km'][1]} km, "
          f"lat {r['lat_range'][0]:.3f} to {r['lat_range'][1]:.3f}")
    print("--- recommended hyperparameters ---")
    for k, v in r["recommended"].items():
        print(f"  {k:28s} {v}")
    lat_span = r["lat_range"][1] - r["lat_range"][0]
    if lat_span < 2.0:
        print("\n  NOTE: this corpus spans <2 degrees of latitude, so the "
              "cos(lat) effect is negligible *within* it. The latitude "
              "hypothesis (H6) is untestable on this data alone -- use "
              "`transfer --dst-lat` for the controlled version, or a second "
              "corpus at a different latitude.")


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def user_disjoint_split(trajs: list[Traj], frac_train: float = 0.8, seed: int = 0):
    """Split so that no user appears in both halves.

    The default for commute data. A random trip-level split leaks: the same
    person's Tuesday and Wednesday commutes are near-duplicates, so a model
    can score well by recognising the individual rather than by representing
    movement. It also makes "generalises to new users" a claim you can make.
    """
    users = sorted({t.uid for t in trajs if t.uid is not None})
    if not users:
        raise ValueError("no user column was loaded; use trip_disjoint_split")
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    k = max(int(frac_train * len(users)), 1)
    tr_u = set(users[:k])
    return ([t for t in trajs if t.uid in tr_u],
            [t for t in trajs if t.uid not in tr_u])


def trip_disjoint_split(trajs: list[Traj], frac_train: float = 0.8, seed: int = 0):
    """Weaker fallback when no user column exists. Overlapping windows from the
    same trip are kept together, but repeated commutes still leak."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajs))
    k = int(frac_train * len(trajs))
    return [trajs[i] for i in idx[:k]], [trajs[i] for i in idx[k:]]


# --------------------------------------------------------------------------
# free ground truth: the HEADING column
# --------------------------------------------------------------------------
def validate_heading(path: str, schema: CsvSchema | None = None,
                     n_rows: int = 200_000,
                     bins_m=(5, 15, 30, 60, 120, 1e9)) -> dict:
    """Compare psi recovered from displacements against the recorded bearing.

    HEADING is normally a compass bearing (degrees clockwise from north),
    whereas psi = atan2(dy, dx) is measured anticlockwise from east, so
    psi = 90 - HEADING. A large residual means the local frame, the column
    convention, or the lon/lat order is wrong -- worth catching before it
    surfaces as a mysterious result in the rotation experiments.
    """
    df = pd.read_csv(path, nrows=n_rows)
    schema = schema or infer_schema(df)
    if schema.heading is None or schema.heading not in df.columns:
        return {"available": False}
    gcol = schema.pick(set(df.columns), schema.group)
    res, rs = [], []
    for _, g in df.groupby(gcol, sort=False):
        if len(g) < 5:
            continue
        ll = g[[schema.lon, schema.lat]].to_numpy(float)
        xy, _ = lonlat_to_enu(ll)
        d = np.diff(xy, axis=0)
        r = np.linalg.norm(d, axis=1)
        m = r > 5.0                      # heading is meaningless at a standstill
        if not m.any():
            continue
        psi = np.arctan2(d[m, 1], d[m, 0])
        rec = np.deg2rad(90.0 - g[schema.heading].to_numpy(float)[1:][m])
        res.append(np.abs(np.angle(np.exp(1j * (psi - rec)))))
        rs.append(r[m])
    if not res:
        return {"available": True, "n": 0}
    e = np.rad2deg(np.concatenate(res))
    r = np.concatenate(rs)

    # Report against step length. Positional noise of sigma metres randomises
    # the direction of a step of comparable length, so a large error at small
    # r is the noise floor, not a convention bug; only a large error at LARGE
    # r indicates a genuine frame, sign or lon/lat-order problem.
    by_r, lo = {}, 0.0
    for hi in bins_m:
        m = (r >= lo) & (r < hi)
        if m.sum() > 50:
            by_r[f"{lo:.0f}-{hi:.0f}m" if hi < 1e8 else f">{lo:.0f}m"] = {
                "n": int(m.sum()), "median_abs_error_deg": float(np.median(e[m]))}
        lo = hi
    tail = list(by_r.values())[-1]["median_abs_error_deg"] if by_r else float(np.median(e))
    out = {"available": True, "n": int(len(e)),
           "median_abs_error_deg": float(np.median(e)),
           "p90_abs_error_deg": float(np.percentile(e, 90)),
           "by_step_length": by_r, "large_step_median_deg": tail}
    print(f"heading check: median |error| {out['median_abs_error_deg']:.1f} deg "
          f"overall, {tail:.1f} deg on the longest steps ({out['n']} steps)")
    for k, v in by_r.items():
        print(f"    {k:>10s}  n={v['n']:7d}  median {v['median_abs_error_deg']:5.1f} deg")
    print("  -> " + ("consistent: psi = 90 - HEADING checks out."
                     if tail < 20 else
                     "LARGE even at long steps: check lon/lat order, the sign "
                     "convention, or whether HEADING is device orientation "
                     "rather than course over ground."))
    return out


# --------------------------------------------------------------------------
# a tiny corpus in the same schema, for checking the pipeline end to end
# --------------------------------------------------------------------------
def write_example_csv(path: str, n_users: int = 12, n_days: int = 14,
                      centre=(10.204, 56.162), seed: int = 0) -> str:
    """Write a fake commute CSV with the same header, so the whole pipeline
    can be exercised before pointing it at real data."""
    import datetime as _dt

    rng = np.random.default_rng(seed)
    lon0, lat0 = centre
    rows = []
    for u in range(n_users):
        home = np.array([lon0 + rng.normal(0, .02), lat0 + rng.normal(0, .015)])
        work = np.array([lon0 + rng.normal(0, .03), lat0 + rng.normal(0, .02)])
        tid = 0
        for day in range(n_days):
            for a, b, hh in ((home, work, 8), (work, home, 17)):
                tid += 1
                n = int(rng.integers(40, 90))
                t0 = (_dt.datetime(2025, 3, 3)
                      + _dt.timedelta(days=day, hours=hh,
                                      minutes=int(rng.integers(0, 40))))
                f = np.linspace(0, 1, n)[:, None]
                p = a + (b - a) * f + rng.normal(0, 3e-4, size=(n, 2))
                for i in range(n):
                    ts = t0 + _dt.timedelta(seconds=15 * i)
                    d = p[min(i + 1, n - 1)] - p[i]
                    rows.append(dict(
                        uid=f"U{u:03d}", DAY=ts.strftime("%Y-%m-%d"),
                        HH24=ts.hour, lat=p[i, 1], lng=p[i, 0],
                        SPEED=float(rng.uniform(0, 60)),
                        HEADING=float((90 - np.degrees(np.arctan2(d[1], d[0]))) % 360),
                        QUALITY=int(rng.integers(1, 6)),
                        ID_PANELSESSION=int(rng.integers(0, 3)),
                        DELTAPOS=float(rng.uniform(0, 30)),
                        datetime=ts.isoformat(), tid=f"U{u:03d}_{tid}",
                        trip_id=f"U{u:03d}_{tid}",
                        trip_id_segment=f"U{u:03d}_{tid}_0", user=f"user_{u}"))
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {len(rows)} rows to {path}")
    return path
