"""Reproduce every facet of GPE's evaluation, with your encoder in the table.

GPE (Liu et al., KDD '25) reports, for each position embedding:

  Table 4 left     local MR, MRR, MP, KP@10, per dataset and averaged
  Table 4 right    global zero-shot MRR: train on one dataset, test on each
  Tables 10-18     the per-dataset detail behind both
  Table 6 left     local MRR as the embedding size h varies
  Table 6 right    training time, split into position pre-training + model
  Table 8          the above with the LSTM swapped for a GNN or a Transformer
  Table 9          road-network HR@10 against TP, DITA, LCRS and NetERP

`parts` selects which of these to run. One contrastive training per
(dataset, encoder, backbone) produces a whole row of Tables 4 and 8: the model
is evaluated on its own test set (local) and zero-shot on every other
dataset's test set (global), so the matrix costs no extra training.

Protocol details kept identical to GPE Sec. 5.1, because every one of them
changes the numbers:

* split="gpe": the **final** n_test trajectories are the test set, the rest
  train. MR and MRR depend on the candidate pool, so n_test defaults to
  GPE's 5,000; smaller pools give better-looking numbers that are not
  comparable to theirs.
* odd/even self-similarity for MR/MRR/MP/KP (see metrics.similarity_metrics).
* contrastive NCE training with drop-0.1 / ~20 m positives.
* non-transferable baselines (raw XY, grid) are evaluated cross-dataset through
  GPE's space-shifting (SS) bounding-box remap; GPE and GEO run natively.

One addition. GEO works in a local metric frame, whose origin is the mean of
the coordinates it is given. Zero-shot on another city that origin can either
stay where training put it (thousands of km away, where the local-frame
approximation distorts displacements) or be re-anchored on the target city's
own coordinates -- which uses no labels and no training. Both are reported, as
`zero-shot` and `zero-shot, frame re-anchored`, so neither choice is hidden.

Table 9 is run the way GPE ran it -- inside ST2Vec, supervised: trajectories
are snapped to road-network vertices, each vertex gets a feature vector
(node2vec, GPE, GEO, or a half-and-half concatenation), ST2Vec's Time2Vec
temporal module is kept for every row so only the spatial features differ, and
the model regresses the exact network distance with ST2Vec's Eq. (12). HR@10
then measures fidelity to that distance.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch

from ..data import Traj, bbox, load_dataset, space_shift
from ..encoders import (GPE, DisplacementGEO, GridEmbed, LearnedFourier, RawXY,
                        Space2Vec, TorusLattice)
from ..geo import enu_to_lonlat
from ..config import save_results
from ..models import set_seed

PARTS = ("local_global", "dims", "arch", "road")


# ==========================================================================
# encoders
# ==========================================================================
def _enc_hp():
    from ..config import HP
    return HP["encoders"]


def geo_layout(h: int | None = None, e: dict | None = None) -> dict:
    """How GEO's dimensions are divided, from the [encoders] configuration.

    Location uses 4 numbers per wavelength, the radial displacement part 2 per
    wavelength, the angular part 2 per harmonic.

    * `geo_loc_dims = 0` (default): location gets `geo_loc_scales` wavelengths
      (0 = h/8, i.e. half of h) and the radial part `geo_disp_scales`
      (0 = whatever fills the other half).
    * `geo_loc_dims = D`: location gets D of the h numbers; displacement gets
      the remaining h - D, angular harmonics first, the rest radial -- possibly
      none. The total is exactly h.

    Raises ValueError (with the nearest valid values) for an impossible split.
    """
    e = e or _enc_hp()
    base_h = e["h"]
    h = h or base_h
    n_harm = len(e["geo_harmonics"])
    D = e.get("geo_loc_dims", 0)
    if D:
        if e["geo_loc_scales"] or e["geo_disp_scales"]:
            raise ValueError("set either geo_loc_dims or geo_loc_scales / "
                             "geo_disp_scales, not both (leave the *_scales at 0)")
        if h != base_h:                      # embedding-size sweep: same share
            D = int(round(D * h / base_h / 4.0)) * 4
            D = max(4, min(D, (h - 2 * n_harm) // 4 * 4))
        rest = h - D - 2 * n_harm
        if D % 4 or D < 4 or D > h or rest < 0:
            top = (h - 2 * n_harm) // 4 * 4
            raise ValueError(
                f"geo_loc_dims={D} is not possible with h={h} and {n_harm} "
                f"harmonic(s): it must be a multiple of 4 between 4 and {top} "
                f"({top} = largest, leaving {h - top - 2 * n_harm} radial number(s))")
        n_loc, n_rad = D // 4, rest // 2
    else:
        if not n_harm:
            raise ValueError("geo_harmonics is empty; for a location-only GEO "
                             "set geo_loc_dims = h as well")
        n_loc = e["geo_loc_scales"] or max(h // 8, 1)
        n_rad = e["geo_disp_scales"] or max((h // 2 - 2 * n_harm) // 2, 1)
    out = {"loc": 4 * n_loc, "radial": 2 * n_rad, "angular": 2 * n_harm,
           "n_loc_scales": n_loc, "n_radial_scales": n_rad}
    out["total"] = out["loc"] + out["radial"] + out["angular"]
    out["text"] = (f"loc {out['loc']} | radial {out['radial']} | "
                   f"angular {out['angular']}")
    return out


def geo_parts(h: int | None = None):
    """(location lattice, displacement GEO or None) at total dimension h.

    Defaults (h=128): 16 location scales from 25 m to 200 km (64 dims) and
    30 radial scales + harmonics m=1, m=4 (64 dims). See `geo_layout` for how
    `geo_loc_dims` moves that boundary. The displacement part is None when it
    has no dimensions at all (location-only GEO).
    """
    e = _enc_hp()
    L = geo_layout(h, e)
    n_loc = L["n_loc_scales"]
    lo, hi = e["geo_loc_lam_min_m"], e["geo_loc_lam_max_m"]
    lams = lo * (hi / lo) ** (np.arange(n_loc) / max(n_loc - 1, 1))
    disp = None
    if L["radial"] + L["angular"] > 0:
        disp = DisplacementGEO(n_scales=L["n_radial_scales"],
                               lam_min=e["geo_disp_lam_min_m"],
                               lam_max=e["geo_disp_lam_max_m"],
                               harmonics=tuple(e["geo_harmonics"]))
    return TorusLattice(lams=tuple(lams)), disp


def gpe_encoder(h: int | None = None):
    e = _enc_hp()
    return GPE(h=h or e["h"], eps_over_2pi=e["gpe_eps_over_2pi"])


def time_encoder():
    from ..encoders import CyclicTimeGENEO
    e = _enc_hp()
    return CyclicTimeGENEO(n_hour_harm=e["time_hour_harmonics"],
                           n_dow_harm=e["time_dow_harmonics"])


def speed_encoder():
    from ..encoders import SpeedGEO
    e = _enc_hp()
    return SpeedGEO(n_scales=e["speed_scales"], lo=e["speed_lo"], hi=e["speed_hi"])


def _default_h():
    return _enc_hp()["h"]


def make_geo(h: int | None = None):
    """GEO at total dimension h: half location lattice, half displacement."""
    from .exp_similarity import Tokenizer
    h = h or _default_h()
    loc, disp = geo_parts(h)
    return Tokenizer(loc=loc, disp=disp,
                     name=f"GEO (h={h})" if h != _default_h() else "GEO")


def make_encoder(key: str, h: int | None = None):
    from .exp_similarity import Tokenizer
    h = h or _default_h()
    tag = f" (h={h})" if h != _default_h() else ""
    if key == "gpe":
        return Tokenizer(loc=gpe_encoder(h), name="GPE" + tag)
    if key == "geo":
        return make_geo(h)
    if key == "xy":
        return Tokenizer(loc=RawXY(normalize=True), name="XY")
    if key == "grid":
        return Tokenizer(loc=GridEmbed(150.0), name="Grid")
    if key == "fourier":
        return Tokenizer(loc=LearnedFourier(h), name="TriW")
    if key == "space2vec":
        return Tokenizer(loc=Space2Vec(n_scales=max(h // 6, 1)), name="Space2Vec" + tag)
    raise KeyError(key)


NON_TRANSFERABLE = ("xy", "grid")          # need GPE's space shifting (SS)
FRAME_BASED = ("geo", "space2vec")         # carry a local metric frame


# ==========================================================================
# data
# ==========================================================================
def split_gpe(trajs, n_train, n_test):
    """GPE Sec. 5.1: the final n_test trajectories are the test set."""
    if len(trajs) <= n_test:
        raise ValueError(f"only {len(trajs)} trajectories; need more than "
                         f"n_test={n_test}")
    test = trajs[-n_test:]
    train = trajs[:-n_test]
    return train[-n_train:] if n_train else train, test


def load_corpora(datasets, paths, n_train, n_test, split, max_traj, seed):
    """Load and split every dataset that is available.

    Returns (data, skipped). Missing / unreadable / too-small datasets are
    reported and skipped instead of aborting the whole run.
    """
    from ..data import load_available
    from .exp_similarity import _split

    # n_train=None means "everything not in the test set" (GPE's protocol);
    # the synthetic generator needs a concrete size, so give it 4x the test
    def kw_for(name):
        return ({"n": (n_train or 4 * n_test) + n_test}
                if name.startswith("synthetic") else {"max_traj": max_traj})

    raw, skipped = load_available(datasets, paths, kw_for, required=0,
                                  what="gpe_suite")
    data = {}
    for name, trajs in raw.items():
        try:
            if split == "gpe":
                tr, te = split_gpe(trajs, n_train, n_test)
            else:
                tr, te = _split(trajs, name, split, n_train, n_test, seed)
        except ValueError as e:
            skipped[name] = f"loaded {len(trajs)} trajectories but could not split: {e}"
            print(f"  !! SKIPPED dataset '{name}': {skipped[name]} "
                  f"(lower --n_test or raise --max_traj)")
            continue
        data[name] = (tr, te)
        print(f"  {name}: {len(tr)} train / {len(te)} test")
    if not data:
        detail = "\n".join(f"    {n}: {r}" for n, r in skipped.items())
        raise FileNotFoundError(
            "gpe_suite: none of the requested datasets could be used.\n"
            f"{detail}\nPass --paths name=/path/to/data (or set DATA_<NAME>); "
            "`python -m geomob.cli datasets` shows what is found.")
    return data, skipped


# ==========================================================================
# one training -> one row of Tables 4 / 8
# ==========================================================================
def train_and_evaluate(key, src, data, kind, cfg, h=None, seed=0):
    from .exp_similarity import evaluate, train_contrastive
    train, test = data[src]
    tok = make_encoder(key, h)
    t0 = time.perf_counter()
    tok.fit(train)                       # "position pre-training"
    t_pre = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = train_contrastive(tok, train, kind=kind, epochs=cfg["epochs"],
                              bs=cfg["bs"], lr=cfg["lr"], device=cfg["device"],
                              seed=seed)
    t_model = time.perf_counter() - t0
    row = {"local": evaluate(model, tok, test, cfg["device"]),
           "time_s": {"position": t_pre, "model": t_model,
                      "total": t_pre + t_model},
           "global": {}, "dim": tok.dim}

    for dst, (dtrain, dtest) in data.items():
        if dst == src:
            continue
        if key in NON_TRANSFERABLE:
            shifted = space_shift(dtest, bbox(dtest), bbox(train))
            row["global"][dst] = {"zero-shot (SS)": evaluate(
                model, tok, shifted, cfg["device"])["MRR"]}
            continue
        g = {"zero-shot": evaluate(model, tok, dtest, cfg["device"])["MRR"]}
        if key in FRAME_BASED:
            # re-anchor the local metric frame on the target's own coordinates;
            # no labels, no gradient steps
            t2 = copy.deepcopy(tok)
            if getattr(t2.loc, "ref_deg", None) is not None:
                t2.loc.ref_deg = np.concatenate([t.lonlat for t in dtest]).mean(0)
            g["zero-shot, frame re-anchored"] = evaluate(
                model, t2, dtest, cfg["device"])["MRR"]
        row["global"][dst] = g
    return row


def _avg(rows, field):
    ks = rows[0][field].keys()
    return {k: float(np.mean([r[field][k] for r in rows])) for k in ks}


def run_local_global(data, encoders, kinds, cfg, seeds):
    """Tables 4 and 8 (and the training-time half of Table 6)."""
    out = {}
    for kind in kinds:
        for key in encoders:
            per_ds = {}
            for src in data:
                reps = []
                for seed in seeds:
                    reps.append(train_and_evaluate(key, src, data, kind, cfg,
                                                   seed=seed))
                r = reps[0] if len(reps) == 1 else _merge_seeds(reps)
                r["seeds"] = [{"seed": int(sd), "local": dict(x["local"]),
                               "global": copy.deepcopy(x["global"])}
                              for sd, x in zip(seeds, reps)]
                per_ds[src] = r
                # the last entry is the most favourable legitimate protocol
                # (re-anchored frame where one exists), printed for a glance
                gl = "  ".join(f"{d}={list(v.values())[-1]:.3f}"
                               for d, v in r["global"].items())
                print(f"  [{kind}] {r_name(key):10s} train {src:12s} local "
                      f"MR={r['local']['MR']:.2f} MRR={r['local']['MRR']:.3f} "
                      f"MP={r['local']['MP']:.3f} KP={r['local']['KP@10']:.3f}"
                      f"  | global {gl}  ({r['time_s']['total']:.0f}s)")
            local_avg = _avg(list(per_ds.values()), "local")
            out.setdefault(kind, {})[key] = {
                "name": r_name(key), "per_dataset": per_ds, "local_avg": local_avg,
                "time_avg_s": _avg(list(per_ds.values()), "time_s")}
    return out


def _merge_seeds(reps):
    r = copy.deepcopy(reps[0])
    for f in ("local", "time_s"):
        # iterate over a snapshot of the keys: the loop adds '<k>_std' entries
        for k in list(reps[0][f]):
            v = [x[f][k] for x in reps]
            r[f][k] = float(np.mean(v))
            r[f][k + "_std"] = float(np.std(v))
    for d in r["global"]:
        for k in r["global"][d]:
            r["global"][d][k] = float(np.mean([x["global"][d][k] for x in reps]))
    return r


def r_name(key):
    return {"gpe": "GPE", "geo": "GEO", "xy": "XY", "grid": "Grid",
            "fourier": "TriW", "space2vec": "Space2Vec"}.get(key, key)


def improvement(ours, theirs, metric):
    """GPE's own 'Improv.' convention, recovered from their Tables 4 and 8:
    relative reduction for MR, absolute percentage points for bounded metrics."""
    if metric == "MR":
        return 100.0 * (theirs - ours) / theirs
    return 100.0 * (ours - theirs)


# ==========================================================================
# Table 6 left: embedding size
# ==========================================================================
def run_dims(data, dims, cfg, seed=0, dataset=None):
    from .exp_similarity import evaluate, train_contrastive
    src = dataset or next(iter(data))
    train, test = data[src]
    out = {"dataset": src, "dims": list(dims), "MRR": {}}
    for key in ("gpe", "geo"):
        for h in dims:
            tok = make_encoder(key, h)
            tok.fit(train)
            model = train_contrastive(tok, train, kind="lstm",
                                      epochs=cfg["epochs"], bs=cfg["bs"],
                                      lr=cfg["lr"], device=cfg["device"],
                                      seed=seed)
            m = evaluate(model, tok, test, cfg["device"])["MRR"]
            out["MRR"].setdefault(r_name(key), {})[str(h)] = m
            print(f"  [dims] {r_name(key):4s} h={h:4d} (actual {tok.dim})  MRR={m:.3f}")
    return out


# ==========================================================================
# Table 9: road network, supervised inside an ST2Vec-style model
# ==========================================================================
def run_road(trajs_train, trajs_test, cfg, dim=None, measures=("TP", "DITA", "LCRS", "NetERP"),
             lam=0.5, cell_m=200.0, graph_path=None, seed=0, k=10):
    dim = dim or _default_h()
    from ..graph import (build_node_graph, hit_ratio_at_k, load_osm_graph,
                         node2vec_embeddings, tie_fraction)
    from .exp_lambda import (STModel, _pdist, build_ground_truth, embed_all,
                             train_regressor)

    g = (load_osm_graph(graph_path, cell_m) if graph_path
         else build_node_graph(trajs_train + trajs_test, cell_m=cell_m))

    def to_vertices(trs):
        """Map-match to vertex sequences; keep the vertex coordinates and the
        time the trajectory reached each vertex."""
        out = []
        for t in trs:
            s = g.snap(t.lonlat)
            keep = np.r_[True, np.diff(s) != 0]
            seq, tt = s[keep], t.t[keep]
            if len(seq) >= 4:
                out.append((seq, Traj(enu_to_lonlat(g.xy[seq], g.ref), tt, t.uid)))
        return out
    V_tr, V_te = to_vertices(trajs_train), to_vertices(trajs_test)
    tr_trajs = [v[1] for v in V_tr]
    te_trajs = [v[1] for v in V_te]
    print(f"  [road] graph {g.n_nodes} vertices; {len(V_tr)} train / "
          f"{len(V_te)} test vertex sequences; lambda={lam}")

    n2v_full = node2vec_embeddings(g, dim=dim, seed=seed, device=cfg["device"],
                                   verbose=False).astype(np.float32)
    n2v_half = node2vec_embeddings(g, dim=dim // 2, seed=seed,
                                   device=cfg["device"], verbose=False).astype(np.float32)

    def features(kind):
        """Per-step feature sequences for one vertex-feature scheme."""
        if kind == "node2vec":
            return [n2v_full[s] for s, _ in V_tr], [n2v_full[s] for s, _ in V_te]
        base, half = kind.split("+")[0], "+" in kind
        tok = make_encoder(base, dim // 2 if half else dim)
        tok.fit(tr_trajs)
        Xtr = [tok(t) for t in tr_trajs]
        Xte = [tok(t) for t in te_trajs]
        if half:
            Xtr = [np.concatenate([x, n2v_half[s]], 1) for x, (s, _) in zip(Xtr, V_tr)]
            Xte = [np.concatenate([x, n2v_half[s]], 1) for x, (s, _) in zip(Xte, V_te)]
        return Xtr, Xte

    schemes = ["node2vec", "gpe", "geo", "gpe+node2vec", "geo+node2vec"]
    names = {"node2vec": f"node2vec (h={dim})", "gpe": f"GPE (h={dim})",
             "geo": f"GEO (h={dim})", "gpe+node2vec": f"GPE+node2vec (h={dim//2}+{dim//2})",
             "geo+node2vec": f"GEO+node2vec (h={dim//2}+{dim//2})"}
    T_tr = [t.t.astype(np.float32) for t in tr_trajs]
    T_te = [t.t.astype(np.float32) for t in te_trajs]

    res = {"n_nodes": int(g.n_nodes), "graph": "osm" if graph_path else "trajectory-induced",
           "lambda": lam, "k": k, "dim": dim, "rows": {n: {} for n in names.values()}}
    for m in measures:
        DS_tr, DT_tr, _ = build_ground_truth(tr_trajs, m, cell_m, graph=g,
                                             temporal_mode="absolute", verbose=False)
        DS_te, DT_te, _ = build_ground_truth(te_trajs, m, cell_m, graph=g,
                                             temporal_mode="absolute", verbose=False)
        GT_tr = lam * DS_tr + (1 - lam) * DT_tr
        GT_te = lam * DS_te + (1 - lam) * DT_te
        res.setdefault("gt_tie_fraction", {})[m] = tie_fraction(GT_te, k)
        for sc in schemes:
            Xtr, Xte = features(sc)
            mu = np.concatenate(Xtr).mean(0)
            sd = np.concatenate(Xtr).std(0) + 1e-6
            Xtr = [((x - mu) / sd).astype(np.float32) for x in Xtr]
            Xte = [((x - mu) / sd).astype(np.float32) for x in Xte]
            set_seed(seed)
            model = STModel(Xtr[0].shape[1], "time2vec").to(cfg["device"])
            train_regressor(model, Xtr, T_tr, GT_tr, epochs=cfg["road_epochs"],
                            device=cfg["device"], seed=seed)
            emb = embed_all(model, Xte, T_te, cfg["device"])
            hr = hit_ratio_at_k(-_pdist(emb), GT_te, k)
            res["rows"][names[sc]][f"HR@{k} {m}"] = hr
            print(f"  [road] {m:7s} {names[sc]:30s} HR@{k}={hr:.3f}")
    return res


# ==========================================================================
# entry point
# ==========================================================================
def run(datasets=("geolife",), paths=None, parts=PARTS,
        encoders=("gpe", "geo"), kinds=("lstm", "transformer", "gnn"),
        n_train=None, n_test=5000, split="gpe", max_traj=20000, epochs=5,
        bs=128, lr=1e-3, dims=(32, 64, 128, 256), road_dataset=None,
        road_n_train=300, road_n_test=150, road_epochs=6, road_lambda=0.5,
        cell_m=200.0, graph_path=None, seeds=(0,), device="cpu", out=None):
    paths = paths or {}
    cfg = dict(epochs=epochs, bs=bs, lr=lr, device=device, road_epochs=road_epochs)
    res = {"datasets": list(datasets), "split": split, "n_test": n_test,
           "encoders": [r_name(e) for e in encoders], "config": cfg, "parts": {}}
    print("loading corpora")
    data, skipped = load_corpora(datasets, paths, n_train, n_test, split,
                                 max_traj, seeds[0])
    # "datasets" = what was actually used, so every downstream average, figure
    # and table caption is about the data that really went in
    res["datasets"] = list(data)
    res["requested_datasets"] = list(datasets)
    res["skipped_datasets"] = skipped
    res["notes"] = []

    def note(msg):
        res["notes"].append(msg)
        print(f"  NOTE: {msg}")

    def save():
        if out:
            save_results(res, out)

    if len(data) < 2 and ("local_global" in parts or "arch" in parts):
        note(f"only one usable dataset ({next(iter(data))}): the zero-shot "
             f"global matrix (GPE Table 4, right) needs at least two and is "
             f"left empty; local metrics and timing are still computed")

    if "local_global" in parts or "arch" in parts:
        k_list = kinds if "arch" in parts else ("lstm",)
        res["parts"]["local_global"] = run_local_global(data, encoders, k_list,
                                                        cfg, seeds)
        save()
    if "dims" in parts:
        res["parts"]["dims"] = run_dims(data, dims, cfg, seeds[0])
        save()
    if "road" in parts:
        rd = road_dataset
        if rd and rd not in data:
            note(f"road dataset '{rd}' is not available "
                 f"({skipped.get(rd, 'not in --datasets')}); using "
                 f"'{next(iter(data))}' instead")
            rd = None
        rd = rd or next(iter(data))
        tr, te = data[rd]
        res["parts"]["road"] = run_road(tr[:road_n_train], te[:road_n_test], cfg,
                                        lam=road_lambda, cell_m=cell_m,
                                        graph_path=graph_path, seed=seeds[0])
        res["parts"]["road"]["dataset"] = rd
        save()
    _summary(res)
    save()
    return res


def _summary(res):
    sk = res.get("skipped_datasets") or {}
    if sk:
        print("\n==== datasets NOT used in this run ====")
        for n, r in sk.items():
            print(f"  {n:10s} {r}")
        print(f"  used: {', '.join(res['datasets'])}")
    lg = res["parts"].get("local_global")
    if not lg:
        return
    print("\n==== GPE Table 4 / 8 layout ====")
    for kind, encs in lg.items():
        print(f"[{kind}]  local average over {len(res['datasets'])} dataset(s)")
        for key, r in encs.items():
            a = r["local_avg"]
            print(f"  {r['name']:10s} MR={a['MR']:.3f}  MRR={a['MRR']:.3f}  "
                  f"MP={a['MP']:.3f}  KP@10={a['KP@10']:.3f}  "
                  f"time={r['time_avg_s']['total']:.0f}s "
                  f"({r['time_avg_s']['position']:.1f} + {r['time_avg_s']['model']:.0f})")
        if "gpe" in encs and "geo" in encs:
            g, o = encs["gpe"]["local_avg"], encs["geo"]["local_avg"]
            print("  GEO vs GPE, in GPE's own 'Improv.' convention: " + "  ".join(
                f"{m} {improvement(o[m], g[m], m):+.1f}%" for m in ("MR", "MRR", "MP", "KP@10")))


if __name__ == "__main__":
    run(datasets=("synthetic_hard",), n_test=500)
