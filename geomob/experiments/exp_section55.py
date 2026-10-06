"""GPE's Section 5.5, run for this tokenizer.

Two experiments:

`run_architectures`  -- their Tables 8/19/20. Swap the backbone (LSTM,
    Transformer, GNN over each trajectory's path graph) while holding the
    tokenizer set fixed, and report local and transfer performance for every
    (tokenizer, backbone) pair. The claim under test is that the tokenizer's
    contribution is a property of the *representation*, not an interaction
    with one particular sequence model -- which is what makes it a tokenizer
    rather than an architectural trick.

`run_roadnetwork`    -- their Table 9. Encode road-network vertices with
    node2vec, with the position embedding, and with both concatenated at half
    dimension each, then rank trajectories against network-aware ground truths
    (TP, DITA, LCRS, NetERP) by HR@10. GPE's finding was that the position
    embedding beats node2vec and that the two are complementary: node2vec
    supplies topology, the embedding supplies geometry. The concatenated row
    is the one that matters, and the half-dimension split is what keeps it an
    honest comparison rather than a capacity increase.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..config import save_results
from ..data import load_dataset, odd_even_split
from ..graph import (build_node_graph, ground_truth_matrix, hit_ratio_at_k,
                     node2vec_embeddings)
from ..metrics import similarity_metrics
from ..models import TrajEncoder, nce_loss, pad_batch, set_seed


# ==========================================================================
# 1. architectures
# ==========================================================================
def run_architectures(dataset="synthetic_hard", path=None,
                      which=("gpe", "geo_spatial", "grid", "xy"),
                      kinds=("lstm", "transformer", "gnn"),
                      n_train=2000, n_test=600, epochs=3, device="cpu",
                      seed=0, split="user", transfer_lat=None, out=None,
                      **load_kw):
    """Every (tokenizer, backbone) pair, local and optionally under transfer."""
    from .exp_similarity import (_split, build_tokenizers, embed,
                                 evaluate, train_contrastive)
    from ..data import Traj
    from ..geo import reproject_to_latitude

    trajs = load_dataset(dataset, path, **load_kw)
    if dataset.startswith("synthetic"):
        trajs = trajs[:n_train + n_test]
    train, test = _split(trajs, dataset, split, n_train, n_test, seed)
    dst = ([Traj(reproject_to_latitude(t.lonlat, transfer_lat), t.t, t.uid)
            for t in test] if transfer_lat is not None else None)

    res = {"dataset": dataset, "kinds": list(kinds), "n_train": len(train),
           "n_test": len(test), "transfer_lat": transfer_lat, "cells": {}}
    for kind in kinds:
        for key, tok in build_tokenizers(tuple(which)).items():
            set_seed(seed)
            tok.fit(train)
            print(f"[{kind}] {tok.name} (dim {tok.dim})")
            model = train_contrastive(tok, train, kind=kind, epochs=epochs,
                                      device=device, seed=seed)
            row = {"local": evaluate(model, tok, test, device),
                   "dim": tok.dim}
            if dst is not None:
                row["transfer"] = evaluate(model, tok, dst, device)
            res["cells"].setdefault(tok.name, {})[kind] = row
            msg = f"  local MRR {row['local']['MRR']:.3f}"
            if dst is not None:
                msg += f"   transfer MRR {row['transfer']['MRR']:.3f}"
            print(msg)

    # Does the ranking of tokenizers survive the change of backbone? If it
    # does not, the tokenizer is interacting with one architecture and the
    # "tokenizer" framing is too strong.
    res["rank_agreement"] = _rank_agreement(res["cells"], kinds)
    print(json.dumps(res["rank_agreement"], indent=2, default=float))
    if out:
        save_results(res, out)
    return res


def _rank_agreement(cells, kinds):
    names = list(cells)
    out = {}
    for a in range(len(kinds)):
        for b in range(a + 1, len(kinds)):
            ka, kb = kinds[a], kinds[b]
            va = [cells[n].get(ka, {}).get("local", {}).get("MRR") for n in names]
            vb = [cells[n].get(kb, {}).get("local", {}).get("MRR") for n in names]
            if any(v is None for v in va + vb) or len(names) < 3:
                continue
            out[f"{ka} vs {kb}"] = _spearman(va, vb)
    return out


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


# ==========================================================================
# 2. road network
# ==========================================================================
class NodeFeatureTokenizer:
    """Vertex features for a road-network model: node2vec, position, or both.

    The three modes reproduce GPE's Table 9 rows. In `both`, each source gets
    half the dimension, so the concatenation is not simply a bigger model.
    """

    def __init__(self, graph, mode: str, pos_encoder=None, n2v=None,
                 name: str = None):
        self.graph, self.mode = graph, mode
        self.n2v, self.pos_encoder = n2v, pos_encoder
        self.name = name or mode
        self.node_feats = self._build()
        self.dim = self.node_feats.shape[1]

    def _build(self):
        parts = []
        if self.mode in ("node2vec", "both"):
            parts.append(self.n2v)
        if self.mode in ("position", "both"):
            from ..geo import enu_to_lonlat
            ll = enu_to_lonlat(self.graph.xy, self.graph.ref)
            parts.append(self.pos_encoder.encode(ll).astype(np.float32))
        f = np.concatenate(parts, axis=1).astype(np.float32)
        return (f - f.mean(0)) / (f.std(0) + 1e-6)

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        return self.node_feats[seq]


def run_roadnetwork(dataset="synthetic_hard", path=None, n_train=1500,
                    n_test=200, cell_m=200.0, dim=128, epochs=4,
                    measures=("TP", "DITA", "LCRS", "NetERP"), k=10,
                    device="cpu", seed=0, split="user", graph_path=None,
                    out=None, **load_kw):
    from .exp_similarity import _split
    from ..encoders import GPE, Space2Vec, TorusLattice

    trajs = load_dataset(dataset, path, **load_kw)
    train, test = _split(trajs, dataset, split, n_train, n_test, seed)

    if graph_path:
        from ..graph import load_osm_graph
        g = load_osm_graph(graph_path, cell_m)
    else:
        g = build_node_graph(train + test, cell_m=cell_m)
    D = g.distance_matrix()

    seqs_tr = [_dedup(g.snap(t.lonlat)) for t in train]
    seqs_te = [_dedup(g.snap(t.lonlat)) for t in test]
    seqs_tr = [s for s in seqs_tr if len(s) >= 4]
    seqs_te = [s for s in seqs_te if len(s) >= 4]
    print(f"vertex sequences: {len(seqs_tr)} train / {len(seqs_te)} test, "
          f"median length {int(np.median([len(s) for s in seqs_te]))}")

    print("computing ground-truth distance matrices")
    truth = {m: ground_truth_matrix(D, seqs_te, m) for m in measures}

    print("training node2vec on the network")
    n2v = node2vec_embeddings(g, dim=dim // 2, seed=seed, device=device)
    n2v_full = node2vec_embeddings(g, dim=dim, seed=seed, device=device,
                                   verbose=False)

    ref = g.ref
    configs = [
        ("node2vec", "node2vec", None, n2v_full),
        (f"GEO ({dim})", "position", TorusLattice(ref_deg=ref), None),
        (f"GPE ({dim})", "position", GPE(h=dim), None),
        (f"Space2Vec ({dim})", "position", Space2Vec(n_scales=dim // 6,
                                                     ref_deg=ref), None),
        (f"GEO+node2vec ({dim // 2}+{dim // 2})", "both",
         TorusLattice(lams=tuple(50.0 * 2.0 ** np.arange(dim // 8)),
                      ref_deg=ref), n2v),
        (f"GPE+node2vec ({dim // 2}+{dim // 2})", "both", GPE(h=dim // 2), n2v),
    ]

    res = {"dataset": dataset, "n_nodes": int(g.n_nodes), "cell_m": cell_m,
           "n_test": len(seqs_te), "k": k, "dim": dim,
           "graph": "osm" if graph_path else "trajectory-induced",
           "rows": {}}
    for name, mode, enc, feats in configs:
        set_seed(seed)
        tokz = NodeFeatureTokenizer(g, mode, enc, feats, name)
        model = _train_ranker(tokz, seqs_tr, epochs, device, seed)
        emb = _embed_seqs(model, tokz, seqs_te, device)
        sim = _cos(emb)
        res["rows"][name] = {"dim": tokz.dim}
        for m in measures:
            res["rows"][name][f"HR@{k} {m}"] = hit_ratio_at_k(sim, truth[m], k)
        print(f"  {name:28s} " + "  ".join(
            f"{m} {res['rows'][name][f'HR@{k} {m}']:.3f}" for m in measures))

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


def _dedup(seq):
    keep = np.r_[True, np.diff(seq) != 0]
    return seq[keep]


def _train_ranker(tokz, seqs, epochs, device, seed, bs=64, lr=1e-3):
    """Contrastive training on vertex sequences, the ST2Vec-style setup."""
    rng = np.random.default_rng(seed)
    model = TrajEncoder(tokz.dim, kind="lstm").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = rng.permutation(len(seqs))
        for s in range(0, len(perm) - bs + 1, bs):
            b = [seqs[i] for i in perm[s:s + bs]]
            xa, ma = pad_batch([tokz(s_) for s_ in b], device)
            # positive = a sub-sampled copy, the network analogue of the
            # point-dropping augmentation used elsewhere here
            pos = [s_[rng.random(len(s_)) > 0.15] for s_ in b]
            pos = [p if len(p) >= 3 else s_ for p, s_ in zip(pos, b)]
            xp, mp = pad_batch([tokz(p) for p in pos], device)
            loss = nce_loss(model(xa, ma), model(xp, mp))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _embed_seqs(model, tokz, seqs, device, bs=128):
    model.eval()
    out = []
    for s in range(0, len(seqs), bs):
        x, m = pad_batch([tokz(q) for q in seqs[s:s + bs]], device)
        out.append(model(x, m).cpu().numpy())
    model.train()
    return np.concatenate(out)


def _cos(e):
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    return e @ e.T
