"""Run every comparison on whatever data is available, then build the figures.

    python -m geomob.cli run_all --paths geolife=/data/Geolife porto=/data/porto.csv
    python -m geomob.cli run_all --quick          # minutes-scale smoke run

Steps (each one is independent: a missing dataset or a failure in one step is
reported and the next step still runs):

  1. dataset check     which of tdrive / porto / roma / ais / geolife are found
  2. gpe_suite         GPE Tables 4, 6, 8, 9 on every *available* real corpus
  3. downstream        TUL / ETA / anomaly on every available corpus, one
                       results file each (mode only where labels exist)
  4. anomaly_synth     channel-aligned anomaly test on the synthetic corpus
                       (needs no data, so it always runs)
  5. figures           figures + tables.tex from every results JSON produced

A `run_all_status.json` next to the results records what ran, what was
skipped and why, so a half-complete batch is never mistaken for a full one.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from ..data import dataset_status

REAL = ("tdrive", "porto", "roma", "ais", "geolife")
STEPS = ("gpe_suite", "downstream", "relocate", "anomaly_synth", "figures")


# options a step fixes itself, so a config section cannot redirect them
_FIXED = {"gpe_suite": {"paths", "out"},
          "downstream": {"dataset", "path", "out"},
          "anomaly": {"dataset", "path", "out"},
          "relocate": {"paths", "out"}}


def _step_kwargs(fn, step, defaults, overrides):
    """quick/full defaults, then the config file's [<step>] section; only
    options `fn` actually accepts. Lists become tuples."""
    import inspect
    ok = set(inspect.signature(fn).parameters)
    kw = dict(defaults)
    for k, v in (overrides or {}).get(step, {}).items():
        if k in _FIXED.get(step, ()) or k not in ok:
            continue
        kw[k] = tuple(v) if isinstance(v, list) else v
    return kw


def run(paths=None, datasets=REAL, steps=STEPS, out_dir="results",
        fig_dir="figures", quick=False, device="cpu", seeds=(0,),
        overrides=None):
    """`overrides` maps a step's command name (gpe_suite / downstream /
    anomaly) to options from the config file; they beat the quick/full
    defaults below."""
    from .. import config as _cfg
    paths = dict(paths or {})
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": quick,
              "datasets": {}, "steps": {}}
    status_file = out_dir.parent / "run_all_status.json"

    parent = dict(_cfg.RUN)          # the run_all invocation itself

    def dump():
        if parent:
            status["_run"] = parent
        json.dump(status, open(status_file, "w"), indent=2, default=str)

    def record(step_cmd, fn, kw, fixed=None):
        """Make the step's results JSON record EVERY option it ran with:
        the function's own defaults, overlaid with what run_all passed."""
        import inspect
        if _cfg.RUN:
            full = {n: p.default for n, p in inspect.signature(fn).parameters.items()
                    if p.default is not inspect.Parameter.empty
                    and p.kind is not inspect.Parameter.VAR_KEYWORD}
            full.update(fixed or {})
            full.update(kw)
            _cfg.RUN.update(command=step_cmd, parent_command="run_all",
                            options={k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in full.items()
                                     if k not in ("out", "paths")},
                            run_all_options=parent.get("options"))
            status.setdefault("step_options", {})[step_cmd] = _cfg.RUN["options"]

    # 1 ------------------------------------------------------------------
    print("=" * 70 + "\n[1] datasets\n" + "=" * 70)
    avail = []
    for name in datasets:
        ok, why = dataset_status(name, paths.get(name))
        status["datasets"][name] = {"available": ok, "detail": why}
        print(f"  {'OK  ' if ok else 'MISS'} {name:8s} {why}")
        if ok:
            avail.append(name)
    if not avail:
        print("  no real corpus found: only the synthetic steps can run")
    dump()

    def step(name, fn):
        if name not in steps:
            return
        print("\n" + "=" * 70 + f"\n[{name}]\n" + "=" * 70)
        t0 = time.perf_counter()
        try:
            info = fn()
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            traceback.print_exc()
            status["steps"][name] = {"state": "FAILED",
                                     "error": f"{type(e).__name__}: {e}"}
        else:
            if isinstance(info, str):       # a reason for skipping
                print(f"  !! SKIPPED: {info}")
                status["steps"][name] = {"state": "SKIPPED", "reason": info}
            else:
                status["steps"][name] = {"state": "DONE", **(info or {})}
        status["steps"][name]["seconds"] = round(time.perf_counter() - t0, 1)
        dump()

    q = quick
    # 2 ------------------------------------------------------------------
    def gpe_suite():
        if not avail:
            return "no real corpus available (GPE's protocol is defined on real data)"
        from .exp_gpe_suite import run as gs
        f = out_dir / "gpe_suite.json"
        # pass the full request: gpe_suite re-checks, skips and records the
        # missing ones in its JSON so figure/table captions can name them
        kw = _step_kwargs(gs, "gpe_suite", dict(
            datasets=tuple(datasets),
            kinds=("lstm",) if q else ("lstm", "transformer", "gnn"),
            n_test=300 if q else 5000, max_traj=2000 if q else 20000,
            epochs=1 if q else 5, dims=(32, 64) if q else (32, 64, 128, 256),
            road_n_train=80 if q else 300, road_n_test=40 if q else 150,
            road_epochs=1 if q else 6, seeds=seeds, device=device), overrides)
        record("gpe_suite", gs, kw)
        res = gs(paths=paths, out=str(f), **kw)
        return {"out": str(f), "used": res["datasets"],
                "skipped": res.get("skipped_datasets", {}),
                "notes": res.get("notes", [])}

    # 3 ------------------------------------------------------------------
    def downstream():
        if not avail:
            return "no real corpus available"
        from .exp_downstream import run as ds
        info = {"out": [], "per_dataset": {}}
        for name in avail:
            print(f"\n---- downstream on {name} ----")
            f = out_dir / f"downstream_{name}.json"
            kw = _step_kwargs(ds, "downstream", dict(
                backbones=("lstm",) if q else ("lstm", "transformer"),
                seeds=seeds, epochs=1 if q else 10,
                max_traj=1500 if q else 20000, device=device), overrides)
            record("downstream", ds, kw, {"dataset": name})
            try:
                res = ds(name, paths.get(name), out=str(f), **kw)
            except Exception as e:  # noqa: BLE001 - other datasets still run
                traceback.print_exc()
                info["per_dataset"][name] = {"state": "FAILED",
                                             "error": f"{type(e).__name__}: {e}"}
                continue
            info["out"].append(str(f))
            info["per_dataset"][name] = {
                "state": "DONE", "tasks_done": list(res["tasks"]),
                "tasks_skipped": res.get("skipped_tasks", {})}
        if not info["out"]:
            raise RuntimeError("downstream failed on every dataset")
        return info

    # 3b -----------------------------------------------------------------
    def relocate():
        if not avail:
            return "no real corpus available"
        from .exp_relocate import run as rl
        f = out_dir / "relocate.json"
        kw = _step_kwargs(rl, "relocate", dict(
            datasets=tuple(datasets), n_test=200 if q else 2000,
            max_traj=1500 if q else 20000, epochs=1 if q else 5,
            latitudes=(0.0, 40.0, 70.0) if q else (0.0, 20.0, 40.0, 60.0, 70.0),
            angles=(0.0, 45.0, 90.0) if q else (0.0, 15.0, 30.0, 45.0, 60.0, 90.0),
            shifts_km=(0.0, 100.0) if q else (0.0, 1.0, 10.0, 100.0, 1000.0),
            seeds=seeds, device=device), overrides)
        record("relocate", rl, kw)
        res = rl(paths=paths, out=str(f), **kw)
        return {"out": str(f), "used": list(res["datasets"]),
                "skipped": res.get("skipped_datasets", {})}

    # 4 ------------------------------------------------------------------
    def anomaly_synth():
        from .exp_anomaly import run as an
        f = out_dir / "anomaly_synthetic.json"
        kw = _step_kwargs(an, "anomaly", dict(
            n_train=400 if q else 2000, n_test=200 if q else 1000,
            severities=(1.0,) if q else (1.0, 0.5, 0.25)), overrides)
        record("anomaly", an, kw, {"dataset": "synthetic_hard"})
        an("synthetic_hard", out=str(f), **kw)
        return {"out": str(f)}

    # 5 ------------------------------------------------------------------
    def figures():
        files = [str(p) for p in sorted(out_dir.glob("*.json"))]
        if not files:
            return "no results JSON produced"
        from ..figures.__main__ import main as figs
        figs(files + ["-o", str(fig_dir)])
        return {"out": str(fig_dir), "inputs": files}

    step("gpe_suite", gpe_suite)
    step("downstream", downstream)
    step("relocate", relocate)
    step("anomaly_synth", anomaly_synth)
    step("figures", figures)

    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump()
    print("\n" + "=" * 70 + "\nrun_all summary\n" + "=" * 70)
    for n, d in status["datasets"].items():
        print(f"  dataset {n:8s} {'used' if d['available'] else 'MISSING: ' + d['detail']}")
    for n, s in status["steps"].items():
        extra = s.get("reason") or s.get("error") or ""
        print(f"  step    {n:14s} {s['state']:8s} {extra}")
        for d, why in (s.get("skipped") or {}).items():
            print(f"            skipped dataset {d}: {why}")
        for t, why in (s.get("tasks_skipped") or {}).items():
            print(f"            skipped task {t}: {why}")
        for d, r in (s.get("per_dataset") or {}).items():
            print(f"            {d}: {r['state']}"
                  + (f" tasks {', '.join(r['tasks_done'])}" if r.get("tasks_done") else "")
                  + (f"  ({r['error']})" if r.get("error") else ""))
            for t, why in (r.get("tasks_skipped") or {}).items():
                print(f"              skipped task {t}: {why}")
    print(f"  status written to {status_file}")
    return status
