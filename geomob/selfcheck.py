"""Installation self-check: `python -m geomob.cli selfcheck`.

Catches the two usual problems in one go:
  * a mixed install (some files replaced by a newer release, others left old),
    which shows up as `ImportError: cannot import name ... from geomob.X`;
  * a missing sub-package (e.g. geomob/figures/ not copied), which shows up
    as `ModuleNotFoundError: No module named 'geomob.figures'`.
It also reports a second copy of geomob shadowing this one on sys.path.
"""
from __future__ import annotations

import importlib
import os
import sys

# every module of the release and the names other modules import from it
REQUIRED = {
    "geomob.geo": ["lonlat_to_enu", "enu_to_lonlat", "reproject_to_latitude"],
    "geomob.encoders": ["GPE", "TorusLattice", "DisplacementGEO", "SpeedGEO",
                        "CyclicTimeGENEO", "RawXY", "GridEmbed",
                        "ENCODER_REGISTRY"],
    "geomob.data": ["Traj", "load_dataset", "load_available", "dataset_status",
                    "DATASETS", "LOADERS"],
    "geomob.csvdata": ["load_csv", "inspect_csv"],
    "geomob.config": ["HP", "load", "save_results", "template"],
    "geomob.equivariance": [],
    "geomob.metrics": ["similarity_metrics"],
    "geomob.codebook": ["EquivariantTorusCodebook"],
    "geomob.anomaly": ["EXPECTED", "build_anomaly_set"],
    "geomob.graph": ["hit_ratio_at_k", "tie_fraction", "build_node_graph"],
    "geomob.models": ["TrajEncoder", "nce_loss"],
    "geomob.experiments.exp_similarity": ["Tokenizer", "train_contrastive"],
    "geomob.experiments.exp_a_task": ["run"],
    "geomob.experiments.exp_anomaly": ["run"],
    "geomob.experiments.exp_lambda": ["run", "STModel"],
    "geomob.experiments.exp_section55": ["run_architectures", "run_roadnetwork"],
    "geomob.experiments.exp_tul": ["run_tul"],
    "geomob.experiments.exp_downstream": ["run", "TASKS"],
    "geomob.experiments.exp_gpe_suite": ["run", "load_corpora"],
    "geomob.experiments.run_all": ["run"],
    "geomob.experiments.exp_search": ["run", "check_space"],
    "geomob.experiments.exp_relocate": ["run", "CanonicalFrame"],
    "geomob.figures": ["save", "use_paper_style"],
    "geomob.figures.builders": [],
    "geomob.figures.builders_suite": ["fig_gpe_suite", "fig_downstream"],
    "geomob.figures.builders_compare": ["build", "fig_cd", "cd_stats"],
    "geomob.figures.__main__": ["main"],
}
TORCH_MODULES = {"geomob.models"} | {m for m in REQUIRED
                                     if m.startswith("geomob.experiments")}


def run(verbose=True):
    import geomob
    root = os.path.dirname(os.path.abspath(geomob.__file__))
    print(f"geomob {getattr(geomob, '__version__', '?')} imported from {root}")

    shadows = []
    for p in sys.path:
        cand = os.path.join(os.path.abspath(p or "."), "geomob", "__init__.py")
        if os.path.exists(cand) and os.path.dirname(cand) != root:
            shadows.append(os.path.dirname(cand))
    if shadows:
        print("  !! other copies of geomob on sys.path (the first one wins):")
        for s in shadows:
            print(f"     {s}")

    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
        print("  !! torch is not installed: only audit/tokens/probe will run "
              "(pip install torch)")

    problems = []
    for mod, names in REQUIRED.items():
        if mod in TORCH_MODULES and not has_torch:
            continue
        rel = mod.replace("geomob.", "", 1).replace(".", os.sep)
        f_mod, f_pkg = os.path.join(root, rel + ".py"), os.path.join(root, rel, "__init__.py")
        if not (os.path.exists(f_mod) or os.path.exists(f_pkg)):
            problems.append(f"MISSING FILE  {mod}  (expected {f_mod} or {f_pkg})")
            continue
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 - report everything
            problems.append(f"IMPORT ERROR  {mod}: {type(e).__name__}: {e}")
            continue
        miss = [n for n in names if not hasattr(m, n)]
        if miss:
            problems.append(f"OUTDATED FILE {mod} ({m.__file__}) lacks: "
                            f"{', '.join(miss)}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
        print("\nFix: this is an incomplete or mixed install. Delete the whole\n"
              f"  {root}\nfolder and replace it with the geomob/ folder of the "
              "latest release\n(do not copy individual files over an older "
              "version).")
        return False
    print(f"OK: all {len(REQUIRED)} modules import"
          + ("" if has_torch else " (torch-dependent ones skipped)"))
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
