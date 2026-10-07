"""Command-line entry point.

    python -m geomob.cli selfcheck  # verify the install is complete and current
    python -m geomob.cli config --write geomob.toml   # hyperparameter template
    python -m geomob.cli gpe_suite --config geomob.toml
    python -m geomob.cli datasets   --paths geolife=/data/Geolife
    python -m geomob.cli run_all    --paths geolife=/data/Geolife porto=/data/porto.csv
    python -m geomob.cli audit      --dataset synthetic
    python -m geomob.cli tokens     --dataset porto --path /data/porto.csv
    python -m geomob.cli probe      --dataset synthetic
    python -m geomob.cli exp_a      --device cpu
    python -m geomob.cli benchmark  --dataset porto --epochs 5
    python -m geomob.cli transfer   --src tdrive --dst porto
    python -m geomob.cli transfer   --src porto --dst-lat 0     # latitude test

`audit`, `tokens` and `probe` need only numpy; the rest need torch.
"""
from __future__ import annotations

import argparse
import sys


def _csv_kw(dataset, args):
    """Loader options that only apply to a raw CSV corpus."""
    if dataset != "csv":
        return {}
    return {"group_col": getattr(args, "group_col", None),
            "gap_min": getattr(args, "gap_min", 10.0)}


def main(argv=None):
    p = argparse.ArgumentParser(prog="geomob")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", default="synthetic",
                        help="synthetic, synthetic_hard, csv, porto, tdrive, "
                             "geolife, roma, ais")
        sp.add_argument("--path", default=None,
                        help="file or directory for the dataset; required for "
                             "--dataset csv")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out", default=None)

    sub.add_parser("selfcheck", help="check every module imports (catches "
                                     "old / partially copied installs)")

    dst = sub.add_parser("datasets", help="which datasets are found, and where")
    dst.add_argument("--paths", nargs="*", default=[],
                     help="name=path pairs; otherwise DATA_<NAME> env vars")
    dst.add_argument("--load", action="store_true",
                     help="actually load each found dataset (slower, but "
                          "catches empty / malformed data)")
    dst.add_argument("--max-traj", type=int, default=2000)

    ra = sub.add_parser("run_all", help="gpe_suite + downstream + synthetic "
                                        "anomaly + figures on the available data")
    ra.add_argument("--paths", nargs="*", default=[],
                    help="name=path pairs; otherwise DATA_<NAME> env vars")
    ra.add_argument("--datasets", nargs="+",
                    default=["tdrive", "porto", "roma", "ais", "geolife"])
    ra.add_argument("--steps", nargs="+",
                    default=["gpe_suite", "downstream", "relocate",
                             "anomaly_synth", "figures"],
                    choices=["gpe_suite", "downstream", "relocate",
                             "anomaly_synth", "figures"])
    ra.add_argument("--out-dir", default="results")
    ra.add_argument("--fig-dir", default="figures")
    ra.add_argument("--quick", action="store_true",
                    help="tiny sizes / 1 epoch: checks the pipeline end to end")
    ra.add_argument("--seeds", type=int, nargs="+", default=[0])
    ra.add_argument("--device", default="cpu")

    insp = sub.add_parser("inspect", help="corpus stats + recommended "
                                          "hyperparameters for your CSV")
    insp.add_argument("--path", required=True)
    insp.add_argument("--group-col", default=None)
    insp.add_argument("--min-len", type=int, default=20)
    insp.add_argument("--gap-min", type=float, default=10.0)
    insp.add_argument("--out", default=None)

    figs = sub.add_parser("figures", help="build manuscript figures from "
                                          "results JSON")
    figs.add_argument("inputs", nargs="+")
    figs.add_argument("-o", "--out-dir", default="figures")
    figs.add_argument("--format", nargs="+", default=["pdf", "png"])
    figs.add_argument("--font", default="serif",
                      choices=["serif", "sans-serif"])

    arch = sub.add_parser("architectures",
                          help="GPE Sec 5.5: tokenizer x backbone ablation")
    arch.add_argument("--dataset", default="synthetic_hard")
    arch.add_argument("--path", default=None)
    arch.add_argument("--which", nargs="+",
                      default=["gpe", "geo_spatial", "grid", "xy"])
    arch.add_argument("--kinds", nargs="+",
                      default=["lstm", "transformer", "gnn"],
                      choices=["lstm", "transformer", "gnn"])
    arch.add_argument("--n-train", type=int, default=2000)
    arch.add_argument("--n-test", type=int, default=600)
    arch.add_argument("--epochs", type=int, default=3)
    arch.add_argument("--transfer-lat", type=float, default=None)
    arch.add_argument("--split", default="user", choices=["user", "trip"])
    arch.add_argument("--group-col", default=None)
    arch.add_argument("--gap-min", type=float, default=10.0)
    arch.add_argument("--device", default="cpu")
    arch.add_argument("--seed", type=int, default=0)
    arch.add_argument("--out", default=None)

    road = sub.add_parser("roadnetwork",
                          help="GPE Sec 5.5: vertex features on a road network")
    road.add_argument("--dataset", default="synthetic_hard")
    road.add_argument("--path", default=None)
    road.add_argument("--graph-path", default=None,
                      help="OSM .graphml or place name (needs osmnx); "
                           "otherwise the network is induced from the trips")
    road.add_argument("--cell-m", type=float, default=200.0)
    road.add_argument("--dim", type=int, default=128)
    road.add_argument("--n-train", type=int, default=1500)
    road.add_argument("--n-test", type=int, default=200)
    road.add_argument("--epochs", type=int, default=4)
    road.add_argument("--k", type=int, default=10)
    road.add_argument("--measures", nargs="+",
                      default=["TP", "DITA", "LCRS", "NetERP"],
                      choices=["TP", "DITA", "LCRS", "NetERP"])
    road.add_argument("--split", default="user", choices=["user", "trip"])
    road.add_argument("--group-col", default=None)
    road.add_argument("--gap-min", type=float, default=10.0)
    road.add_argument("--device", default="cpu")
    road.add_argument("--seed", type=int, default=0)
    road.add_argument("--out", default=None)

    ds = sub.add_parser(
        "downstream",
        help="encoders x backbones on TUL / ETA / mode / anomaly (GeoLife)")
    ds.add_argument("--dataset", default="geolife")
    ds.add_argument("--path", default=None)
    ds.add_argument("--tasks", nargs="+", default=["tul", "eta", "mode", "anomaly"],
                    choices=["tul", "eta", "mode", "anomaly"])
    ds.add_argument("--encoders", nargs="+",
                    default=["gpe", "gpe_ts", "geo", "geo_ts"],
                    choices=["gpe", "gpe_ts", "geo", "geo_ts"])
    ds.add_argument("--backbones", nargs="+", default=["lstm", "transformer"],
                    choices=["lstm", "transformer", "gnn"])
    ds.add_argument("--seeds", type=int, nargs="+", default=[0])
    ds.add_argument("--epochs", type=int, default=10)
    ds.add_argument("--bs", type=int, default=64)
    ds.add_argument("--lr", type=float, default=1e-3)
    ds.add_argument("--max-traj", type=int, default=20000)
    ds.add_argument("--min-trips", type=int, default=10,
                    help="TUL: users with fewer trajectories are dropped")
    ds.add_argument("--eta-step-m", type=float, default=30.0)
    ds.add_argument("--mode-split", default="user", choices=["user", "trip"])
    ds.add_argument("--anomaly-rate", type=float, default=0.28)
    ds.add_argument("--severity", type=float, default=1.0)
    ds.add_argument("--eta-max-min", type=float, default=180.0,
                    help="ETA: trips longer than this (minutes) are dropped")
    ds.add_argument("--z-dim", type=int, default=16,
                    help="anomaly: autoencoder bottleneck size")
    ds.add_argument("--k", type=int, default=10,
                    help="anomaly: neighbours for the latent kNN score")
    ds.add_argument("--device", default="cpu")
    ds.add_argument("--out", default=None)

    gs = sub.add_parser("gpe_suite",
                        help="every facet of GPE's evaluation (Tables 4, 6, 8, 9)")
    gs.add_argument("--datasets", nargs="+",
                    default=["tdrive", "porto", "roma", "ais", "geolife"])
    gs.add_argument("--paths", nargs="*", default=[],
                    help="name=path pairs; otherwise DATA_<NAME> env vars are used")
    gs.add_argument("--parts", nargs="+",
                    default=["local_global", "dims", "arch", "road"],
                    choices=["local_global", "dims", "arch", "road"])
    gs.add_argument("--encoders", nargs="+", default=["gpe", "geo"],
                    choices=["gpe", "geo", "xy", "grid", "fourier", "space2vec"])
    gs.add_argument("--kinds", nargs="+", default=["lstm", "transformer", "gnn"],
                    choices=["lstm", "transformer", "gnn"])
    gs.add_argument("--n-train", type=int, default=None)
    gs.add_argument("--n-test", type=int, default=5000)
    gs.add_argument("--split", default="gpe", choices=["gpe", "user", "trip"])
    gs.add_argument("--max-traj", type=int, default=20000)
    gs.add_argument("--epochs", type=int, default=5)
    gs.add_argument("--bs", type=int, default=128)
    gs.add_argument("--lr", type=float, default=1e-3)
    gs.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256])
    gs.add_argument("--road-dataset", default=None)
    gs.add_argument("--road-n-train", type=int, default=300)
    gs.add_argument("--road-n-test", type=int, default=150)
    gs.add_argument("--road-epochs", type=int, default=6)
    gs.add_argument("--road-lambda", type=float, default=0.5)
    gs.add_argument("--cell-m", type=float, default=200.0)
    gs.add_argument("--graph-path", default=None)
    gs.add_argument("--seeds", type=int, nargs="+", default=[0])
    gs.add_argument("--device", default="cpu")
    gs.add_argument("--out", default=None)

    anom = sub.add_parser("anomaly",
                          help="channel-aligned anomaly detection")
    anom.add_argument("--dataset", default="synthetic_hard")
    anom.add_argument("--path", default=None)
    anom.add_argument("--which", nargs="+",
                      default=["geo_c4", "geo_speed", "geo_full", "geo_spatial",
                               "disp_only", "radial_only", "time_only", "gpe",
                               "grid"])
    anom.add_argument("--rate", type=float, default=0.28)
    anom.add_argument("--severities", type=float, nargs="+",
                      default=[1.0, 0.5, 0.25])
    anom.add_argument("-k", type=int, default=10)
    anom.add_argument("--n-train", type=int, default=2000)
    anom.add_argument("--n-test", type=int, default=1000)
    anom.add_argument("--split", default="user", choices=["user", "trip"])
    anom.add_argument("--group-col", default=None)
    anom.add_argument("--gap-min", type=float, default=10.0)
    anom.add_argument("--seed", type=int, default=0)
    anom.add_argument("--out", default=None)

    lam = sub.add_parser("lambda",
                         help="ST2Vec-style lambda sweep over the "
                              "spatial/temporal ground truth")
    lam.add_argument("--dataset", default="synthetic_hard")
    lam.add_argument("--path", default=None)
    lam.add_argument("--measure", default="TP",
                     choices=["TP", "DITA", "LCRS", "NetERP"])
    lam.add_argument("--lambdas", type=float, nargs="+",
                     default=[1.0, 0.75, 0.5, 0.25, 0.0])
    lam.add_argument("--time-kinds", nargs="+",
                     default=["none", "raw", "time2vec", "cyclic"],
                     choices=["none", "raw", "time2vec", "cyclic"])
    lam.add_argument("--temporal-mode", default="cyclic",
                     choices=["cyclic", "absolute"])
    lam.add_argument("--shift-test-h", type=float, default=168.0)
    lam.add_argument("--n-train", type=int, default=400)
    lam.add_argument("--n-test", type=int, default=200)
    lam.add_argument("--cell-m", type=float, default=200.0)
    lam.add_argument("--epochs", type=int, default=8)
    lam.add_argument("-k", type=int, default=10)
    lam.add_argument("--split", default="user", choices=["user", "trip"])
    lam.add_argument("--group-col", default=None)
    lam.add_argument("--gap-min", type=float, default=10.0)
    lam.add_argument("--device", default="cpu")
    lam.add_argument("--seed", type=int, default=0)
    lam.add_argument("--out", default=None)

    tul = sub.add_parser("tul", help="trajectory-user linking (any corpus with "
                                     "user ids); see also `downstream`")
    tul.add_argument("--dataset", default="csv")
    tul.add_argument("--path", default=None)
    tul.add_argument("--which", nargs="+",
                     default=["gpe", "geo_spatial", "geo_full", "grid"])
    tul.add_argument("--split", default="day", choices=["day", "trip"])
    tul.add_argument("--epochs", type=int, default=15)
    tul.add_argument("--group-col", default=None)
    tul.add_argument("--gap-min", type=float, default=10.0)
    tul.add_argument("--device", default="cpu")
    tul.add_argument("--seed", type=int, default=0)
    tul.add_argument("--out", default=None)

    common(sub.add_parser("audit"))
    common(sub.add_parser("tokens"))
    common(sub.add_parser("probe"))

    a = sub.add_parser("exp_a", help="equivariance + sample efficiency on a "
                                     "real downstream task")
    a.add_argument("--dataset", default="synthetic")
    a.add_argument("--path", default=None)
    a.add_argument("--task", default="destination",
                   choices=["destination", "eta"])
    a.add_argument("--which", nargs="+",
                   default=["gpe", "geo_spatial", "grid", "xy", "fourier",
                            "space2vec"])
    a.add_argument("--prefix", type=float, default=0.7,
                   help="fraction of each trip that is observed")
    a.add_argument("--n-test", type=int, default=600)
    a.add_argument("--kind", default="lstm", choices=["lstm", "transformer"])
    a.add_argument("--split", default="user", choices=["user", "trip"])
    a.add_argument("--group-col", default=None)
    a.add_argument("--gap-min", type=float, default=10.0)
    a.add_argument("--device", default="cpu")
    a.add_argument("--epochs", type=int, default=10)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default=None)
    a.add_argument("--sizes", type=int, nargs="+",
                   default=[200, 500, 1000, 2000])

    b = sub.add_parser("benchmark")
    common(b)
    b.add_argument("--which", nargs="+",
                   default=["gpe", "geo_spatial", "xy", "grid"])
    b.add_argument("--kind", default="lstm", choices=["lstm", "transformer"])
    b.add_argument("--epochs", type=int, default=5)
    b.add_argument("--n-train", type=int, default=4000)
    b.add_argument("--n-test", type=int, default=1000)
    b.add_argument("--device", default="cpu")

    g = sub.add_parser("transfer")
    g.add_argument("--src", default="synthetic")
    g.add_argument("--dst", default="synthetic")
    g.add_argument("--src-path", default=None)
    g.add_argument("--dst-path", default=None)
    g.add_argument("--dst-lat", type=float, default=None)
    g.add_argument("--dst-scale", type=float, default=None)
    g.add_argument("--which", nargs="+",
                   default=["gpe", "geo_spatial", "xy", "grid"])
    g.add_argument("--epochs", type=int, default=5)
    g.add_argument("--ft-epochs", type=int, default=2)
    g.add_argument("--n-train", type=int, default=4000)
    g.add_argument("--n-test", type=int, default=1000)
    g.add_argument("--n-finetune", type=int, default=500)
    g.add_argument("--device", default="cpu")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=None)

    for sp in (b, g):
        sp.add_argument("--group-col", default=None)
        sp.add_argument("--gap-min", type=float, default=10.0)
        sp.add_argument("--split", default="user",
                        choices=["user", "trip"],
                        help="how to hold out test data on a CSV corpus; "
                             "'user' (default) prevents a person's repeated "
                             "commutes leaking across the split")

    se = sub.add_parser(
        "search", help="hyperparameter search for the GEO encoder, scored on "
                       "a validation split (values to try: [search_space] in "
                       "the config file)")
    se.add_argument("--objective", default="similarity",
                    choices=["similarity", "tul", "eta", "mode"],
                    help="what a trial is scored on: trajectory similarity "
                         "(any dataset) or a GeoLife downstream task")
    se.add_argument("--metric", default=None,
                    help="similarity: KP@10 (default), MRR, MP, MR, global_MRR; "
                         "tul: acc@1 (default), acc@5, macro_F1; eta: MAE "
                         "(default), RMSE, MAPE; mode: macro_F1 (default), acc@1")
    se.add_argument("--method", default="grid", choices=["grid", "random"],
                    help="grid = every combination; random = n_trials of them")
    se.add_argument("--n-trials", type=int, default=20,
                    help="number of combinations when method = random")
    se.add_argument("--datasets", nargs="+",
                    default=["tdrive", "porto", "roma", "ais", "geolife"],
                    help="similarity objective: corpora to average over "
                         "(missing ones are skipped)")
    se.add_argument("--paths", nargs="*", default=[],
                    help="name=path pairs; otherwise [paths] / DATA_<NAME>")
    se.add_argument("--encoder", default=None,
                    choices=["geo", "geo_ts"],
                    help="default: geo for similarity, geo_ts (with time and "
                         "speed) for the downstream objectives")
    se.add_argument("--kind", default="lstm",
                    choices=["lstm", "transformer", "gnn"])
    se.add_argument("--epochs", type=int, default=3,
                    help="training epochs per trial (keep small: this is "
                         "multiplied by the number of trials)")
    se.add_argument("--bs", type=int, default=128)
    se.add_argument("--lr", type=float, default=1e-3)
    se.add_argument("--n-train", type=int, default=4000,
                    help="similarity: training trajectories per dataset per trial")
    se.add_argument("--n-val", type=int, default=1000,
                    help="similarity: validation trajectories trials are scored on")
    se.add_argument("--n-test", type=int, default=5000,
                    help="similarity: size of the gpe_suite test set to hold "
                         "out untouched (use the same value as [gpe_suite] n_test)")
    se.add_argument("--max-traj", type=int, default=20000)
    se.add_argument("--seeds", type=int, nargs="+", default=[0])
    se.add_argument("--search-seed", type=int, default=0,
                    help="which random subset is drawn when method = random")
    se.add_argument("--device", default="cpu")
    se.add_argument("--out-dir", default="results/search")
    se.add_argument("--no-resume", action="store_true",
                    help="ignore trials already recorded in out_dir/trials.jsonl")

    rl = sub.add_parser(
        "relocate", help="relocated-city test: train once, then move the test "
                         "city to other latitudes, turn it, shift it")
    rl.add_argument("--datasets", nargs="+",
                    default=["tdrive", "porto", "roma", "ais", "geolife"])
    rl.add_argument("--paths", nargs="*", default=[],
                    help="name=path pairs; otherwise [paths] / DATA_<NAME>")
    rl.add_argument("--tasks", nargs="+", default=["similarity", "eta", "mode"],
                    choices=["similarity", "eta", "mode"],
                    help="similarity is the control; eta and mode are the "
                         "supervised tasks whose answer does not depend on "
                         "where the city is (mode needs GeoLife labels)")
    rl.add_argument("--channels", default="spatial", choices=["spatial", "ts"],
                    help="spatial = GPE vs GEO as in gpe_suite; ts = both with "
                         "time and speed added (eta and mode only)")
    rl.add_argument("--kinds", nargs="+", default=["lstm"],
                    choices=["lstm", "transformer", "gnn"])
    rl.add_argument("--latitudes", type=float, nargs="*",
                    default=[0.0, 20.0, 40.0, 60.0, 70.0],
                    help="latitudes (degrees north) the test city is moved to")
    rl.add_argument("--angles", type=float, nargs="*",
                    default=[0.0, 15.0, 30.0, 45.0, 60.0, 90.0],
                    help="rotations (degrees) of the test city about its centre")
    rl.add_argument("--shifts-km", type=float, nargs="*",
                    default=[0.0, 1.0, 10.0, 100.0, 1000.0],
                    help="eastward shifts (km) of the test city")
    rl.add_argument("--n-train", type=int, default=None)
    rl.add_argument("--n-test", type=int, default=2000)
    rl.add_argument("--max-traj", type=int, default=20000)
    rl.add_argument("--epochs", type=int, default=5)
    rl.add_argument("--bs", type=int, default=128)
    rl.add_argument("--lr", type=float, default=1e-3)
    rl.add_argument("--seeds", type=int, nargs="+", default=[0])
    rl.add_argument("--device", default="cpu")
    rl.add_argument("--out", default=None)

    cfgp = sub.add_parser("config", help="write a commented config template "
                                         "listing every hyperparameter")
    cfgp.add_argument("--write", default="geomob.toml",
                      help="output file (default geomob.toml)")
    cfgp.add_argument("--commands", nargs="*", default=None,
                      help="only these command sections (default: all)")
    cfgp.add_argument("--force", action="store_true",
                      help="overwrite an existing file")

    for name, sp in sub.choices.items():
        sp.add_argument("--config", default=None, metavar="FILE.toml",
                        help="TOML config with hyperparameters; flags given on "
                             "the command line override it")

    # ---- config file: architecture, paths, then this command's defaults ----
    from . import config as _cfg
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    cfg, cfg_text = {}, None
    cmd = next((a for a in argv if a in sub.choices), None)
    if known.config and cmd != "config":
        cfg, cfg_text = _cfg.load(known.config)
        _cfg.apply_architecture(cfg)
        _cfg.apply_paths(cfg)
        if cmd:
            sub.choices[cmd].set_defaults(
                **_cfg.command_defaults(cfg, sub.choices, cmd))

    _cfg.VALID_KEYS.update({n: set(_cfg._actions(sp)) for n, sp in sub.choices.items()})
    args = p.parse_args(argv)
    args._cfg_overrides = {
        sec: _cfg.section_overrides(cfg, sub.choices, sec)
        for sec in ("gpe_suite", "downstream", "anomaly", "relocate")} if cfg else {}
    args._subparsers = sub.choices
    args._cfg = cfg
    _cfg.start_run(args.cmd, argparse.Namespace(**{
        k: v for k, v in vars(args).items() if not k.startswith("_")}),
        known.config, cfg_text, ["geomob.cli"] + argv)
    if known.config:
        print(f"config: {known.config} (command-line flags override it)")

    # Create the parent directory of --out once, here, rather than in each
    # experiment: a fresh clone has no results/ and every runner would
    # otherwise die with FileNotFoundError after doing all the work.
    out_path = getattr(args, "out", None)
    if out_path:
        import pathlib
        pathlib.Path(out_path).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True)

    try:
        return _dispatch(args)
    except FileNotFoundError as e:
        # a dataset that is not there: say so plainly, no traceback
        print(f"\n[{args.cmd}] cannot run: {e}", file=sys.stderr)
        print("  `python -m geomob.cli datasets` lists what is found; "
              "`run_all` / `gpe_suite` skip missing datasets automatically.",
              file=sys.stderr)
        return None


def _paths(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--paths expects name=path, got '{p}'")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _dispatch(args):
    if args.cmd == "config":
        import os as _os
        from .config import template
        if _os.path.exists(args.write) and not args.force:
            print(f"{args.write} exists; pass --force to overwrite")
            return None
        open(args.write, "w").write(template(args._subparsers, args.commands))
        print(f"wrote {args.write}: every option of every command, commented "
              f"out at its default, plus [model] and [encoders]. Uncomment and "
              f"edit what you change, then pass --config {args.write}")
        return True

    if args.cmd == "search":
        from .experiments.exp_search import run as _se
        return _se(space=args._cfg.get("search_space"), base_config=args._cfg,
                   objective=args.objective, metric=args.metric,
                   method=args.method, n_trials=args.n_trials,
                   datasets=tuple(args.datasets), paths=_paths(args.paths),
                   encoder=args.encoder, kind=args.kind, epochs=args.epochs,
                   bs=args.bs, lr=args.lr, n_train=args.n_train,
                   n_val=args.n_val, n_test=args.n_test, max_traj=args.max_traj,
                   seeds=tuple(args.seeds), search_seed=args.search_seed,
                   device=args.device, out_dir=args.out_dir,
                   resume=not args.no_resume)

    if args.cmd == "relocate":
        from .experiments.exp_relocate import run as _rl
        return _rl(datasets=tuple(args.datasets), paths=_paths(args.paths),
                   tasks=tuple(args.tasks), channels=args.channels,
                   kinds=tuple(args.kinds), latitudes=tuple(args.latitudes),
                   angles=tuple(args.angles), shifts_km=tuple(args.shifts_km),
                   n_train=args.n_train, n_test=args.n_test,
                   max_traj=args.max_traj, epochs=args.epochs, bs=args.bs,
                   lr=args.lr, seeds=tuple(args.seeds), device=args.device,
                   out=args.out)

    if args.cmd == "selfcheck":
        from .selfcheck import run as _sc
        return True if _sc() else None

    if args.cmd == "datasets":
        from .data import DATASETS, dataset_status, load_dataset
        paths = _paths(args.paths)
        n_ok = 0
        for name in DATASETS:
            if name == "csv" and "csv" not in paths and not __import__("os").environ.get("DATA_CSV"):
                continue
            ok, why = dataset_status(name, paths.get(name))
            if ok and args.load and not name.startswith("synthetic"):
                try:
                    n = len(load_dataset(name, paths.get(name), max_traj=args.max_traj))
                    why += f"  ({n} trajectories loaded, max_traj={args.max_traj})"
                except Exception as e:  # noqa: BLE001
                    ok, why = False, f"found {why} but loading failed: {e}"
            n_ok += ok
            print(f"  {'OK  ' if ok else 'MISS'} {name:15s} {why}")
        return n_ok

    if args.cmd == "run_all":
        from .experiments.run_all import run as _ra
        return _ra(paths=_paths(args.paths), datasets=tuple(args.datasets),
                   steps=tuple(args.steps), out_dir=args.out_dir,
                   fig_dir=args.fig_dir, quick=args.quick,
                   device=args.device, seeds=tuple(args.seeds),
                   overrides=args._cfg_overrides)

    if args.cmd == "figures":
        from .figures.__main__ import main as _figs
        return _figs(args.inputs + ["-o", args.out_dir, "--font", args.font,
                                    "--format"] + args.format)

    if args.cmd == "downstream":
        from .experiments.exp_downstream import run as _ds
        return _ds(args.dataset, args.path, tasks=tuple(args.tasks),
                   encoders=tuple(args.encoders), backbones=tuple(args.backbones),
                   seeds=tuple(args.seeds), epochs=args.epochs, bs=args.bs,
                   lr=args.lr, max_traj=args.max_traj, min_trips=args.min_trips,
                   eta_step_m=args.eta_step_m, mode_split=args.mode_split,
                   anomaly_rate=args.anomaly_rate, severity=args.severity,
                   eta_max_min=args.eta_max_min, z_dim=args.z_dim, k=args.k,
                   device=args.device, out=args.out)

    if args.cmd == "gpe_suite":
        from .experiments.exp_gpe_suite import run as _gs
        paths = _paths(args.paths)
        return _gs(datasets=tuple(args.datasets), paths=paths,
                   parts=tuple(args.parts), encoders=tuple(args.encoders),
                   kinds=tuple(args.kinds), n_train=args.n_train,
                   n_test=args.n_test, split=args.split, max_traj=args.max_traj,
                   epochs=args.epochs, bs=args.bs, lr=args.lr,
                   dims=tuple(args.dims),
                   road_dataset=args.road_dataset, road_n_train=args.road_n_train,
                   road_n_test=args.road_n_test, road_epochs=args.road_epochs,
                   road_lambda=args.road_lambda, cell_m=args.cell_m,
                   graph_path=args.graph_path, seeds=tuple(args.seeds),
                   device=args.device, out=args.out)

    if args.cmd == "anomaly":
        from .experiments.exp_anomaly import run as _anom
        return _anom(args.dataset, args.path, which=tuple(args.which),
                     rate=args.rate, severities=tuple(args.severities),
                     k=args.k, n_train=args.n_train, n_test=args.n_test,
                     split=args.split, seed=args.seed, out=args.out,
                     **_csv_kw(args.dataset, args))

    if args.cmd == "lambda":
        from .experiments.exp_lambda import run as _lam
        return _lam(args.dataset, args.path, measure=args.measure,
                    lambdas=tuple(args.lambdas),
                    time_kinds=tuple(args.time_kinds),
                    temporal_mode=args.temporal_mode,
                    shift_test_h=args.shift_test_h, n_train=args.n_train,
                    n_test=args.n_test, cell_m=args.cell_m,
                    epochs=args.epochs, k=args.k, split=args.split,
                    device=args.device, seed=args.seed, out=args.out,
                    **_csv_kw(args.dataset, args))

    if args.cmd == "architectures":
        from .experiments.exp_section55 import run_architectures
        return run_architectures(
            args.dataset, args.path, which=tuple(args.which),
            kinds=tuple(args.kinds), n_train=args.n_train, n_test=args.n_test,
            epochs=args.epochs, transfer_lat=args.transfer_lat,
            split=args.split, device=args.device, seed=args.seed,
            out=args.out, **_csv_kw(args.dataset, args))

    if args.cmd == "roadnetwork":
        from .experiments.exp_section55 import run_roadnetwork
        return run_roadnetwork(
            args.dataset, args.path, n_train=args.n_train,
            n_test=args.n_test, cell_m=args.cell_m, dim=args.dim,
            epochs=args.epochs, measures=tuple(args.measures), k=args.k,
            split=args.split, device=args.device, seed=args.seed,
            graph_path=args.graph_path, out=args.out,
            **_csv_kw(args.dataset, args))

    if args.cmd == "inspect":
        from .csvdata import inspect_csv, validate_heading
        r = inspect_csv(args.path, group_col=args.group_col,
                        min_len=args.min_len, gap_min=args.gap_min,
                        verbose=False)
        r["heading_check"] = validate_heading(args.path)
        if args.out:
            import json as _j
            _j.dump(r, open(args.out, "w"), indent=2, default=float)
        return r

    if args.cmd == "tul":
        from .data import load_dataset
        from .experiments.exp_similarity import build_tokenizers
        from .experiments.exp_tul import run_tul
        ds_name = getattr(args, "dataset", "csv")
        trajs = load_dataset(ds_name, args.path, **_csv_kw(ds_name, args))
        return run_tul(trajs, build_tokenizers(tuple(args.which)),
                       split=args.split, epochs=args.epochs,
                       device=args.device, seed=args.seed, out=args.out)

    if args.cmd == "audit":
        from .experiments.exp_audit import run
        return run(args.dataset, args.path, seed=args.seed, out=args.out)
    if args.cmd == "tokens":
        from .experiments.exp_tokens_probe import run_tokens
        return run_tokens(args.dataset, args.path, seed=args.seed, out=args.out)
    if args.cmd == "probe":
        from .experiments.exp_tokens_probe import run_probe
        return run_probe(args.dataset, args.path, seed=args.seed, out=args.out)
    if args.cmd == "exp_a":
        from .experiments.exp_a_task import run
        return run(dataset=args.dataset, path=args.path, task=args.task,
                   which=tuple(args.which), sizes=tuple(args.sizes),
                   n_test=args.n_test, epochs=args.epochs,
                   prefix=args.prefix, kind=args.kind, split=args.split,
                   device=args.device, seed=args.seed, out=args.out,
                   **_csv_kw(args.dataset, args))
    if args.cmd == "benchmark":
        from .experiments.exp_similarity import run_benchmark
        return run_benchmark(args.dataset, args.path, n_train=args.n_train,
                             n_test=args.n_test, which=tuple(args.which),
                             kind=args.kind, epochs=args.epochs,
                             device=args.device, seed=args.seed, out=args.out,
                             split=args.split,
                             **_csv_kw(args.dataset, args))
    if args.cmd == "transfer":
        from .experiments.exp_similarity import run_transfer
        return run_transfer(args.src, args.dst, args.src_path, args.dst_path,
                            which=tuple(args.which), epochs=args.epochs,
                            ft_epochs=args.ft_epochs, n_train=args.n_train,
                            n_test=args.n_test, n_finetune=args.n_finetune,
                            device=args.device, seed=args.seed,
                            dst_lat=args.dst_lat, dst_scale=args.dst_scale,
                            out=args.out, split=args.split,
                            **_csv_kw(args.src, args))
    return None


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
