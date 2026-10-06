"""A road-network layer, for the analogue of GPE's Section 5.5 road experiment.

GPE integrate with ST2Vec, which encodes road-network vertices with node2vec,
and ask whether replacing or augmenting those vertex features with a position
embedding helps. To run the same experiment you need a network, a mapping from
GPS points to it, vertex features, and network-aware ground-truth distances.
This module provides all four.

**On the network itself.** Real road graphs (OSM) are the ideal input and
`load_osm_graph` accepts one if you have it. Absent that, `build_node_graph`
induces a network *from the trajectories*: nodes are occupied cells of a grid,
edges join cells that are consecutive in some trip. For a commute corpus this
is a genuinely reasonable proxy -- the induced graph is the road network as
actually travelled, restricted to the roads people use -- but it is a proxy,
and saying so is part of reporting the experiment honestly. Two consequences
to state in any write-up: the graph is derived from the same data the models
see, so it cannot be used to claim independence from the corpus; and it has no
one-way or turn restrictions.

**On the distances.** ST2Vec's ground truths (TP, DITA, LCRS, NetERP) are
defined over road segments. The implementations here are faithful in form but
operate on the induced vertex sequence rather than on matched road segments,
so treat them as network-aware analogues, not reimplementations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .geo import lonlat_to_enu


# ==========================================================================
# the network
# ==========================================================================
@dataclass
class NodeGraph:
    xy: np.ndarray            # (V, 2) node coordinates, metres
    ref: np.ndarray           # lon/lat of the local frame origin
    adj: coo_matrix           # (V, V) edge weights, metres
    cell_m: float
    _index: dict              # cell key -> node id

    @property
    def n_nodes(self):
        return len(self.xy)

    def snap(self, lonlat: np.ndarray) -> np.ndarray:
        """Map GPS points to node ids (the stand-in for map matching).

        Points falling in an unoccupied cell are attached to the nearest
        occupied one, so every trajectory yields a complete vertex sequence.
        """
        xy, _ = lonlat_to_enu(lonlat, self.ref)
        keys = np.floor(xy / self.cell_m).astype(int)
        out = np.empty(len(xy), dtype=int)
        miss = []
        for i, k in enumerate(keys):
            nid = self._index.get((int(k[0]), int(k[1])), -1)
            out[i] = nid
            if nid < 0:
                miss.append(i)
        if miss:
            d = ((xy[miss][:, None, :] - self.xy[None]) ** 2).sum(-1)
            out[miss] = d.argmin(1)
        return out

    def distance_matrix(self, cap: float = np.inf) -> np.ndarray:
        """All-pairs shortest path in metres (dense; keep the graph modest)."""
        D = dijkstra(self.adj.tocsr(), directed=False)
        # Disconnected pairs get a finite penalty so the distance measures
        # stay well defined; report the fraction affected.
        finite = np.isfinite(D)
        fill = min(cap, D[finite].max() * 2 if finite.any() else 1.0)
        D[~finite] = fill
        return D


def build_node_graph(trajs, cell_m: float = 200.0, min_visits: int = 2,
                     ref=None, verbose: bool = True) -> NodeGraph:
    """Induce a network from trajectories: cells as vertices, transitions as edges."""
    ll = np.concatenate([t.lonlat for t in trajs])
    if ref is None:
        ref = ll.mean(0)
    xy, ref = lonlat_to_enu(ll, ref)
    keys = np.floor(xy / cell_m).astype(int)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                                  return_counts=True)
    keep = counts >= min_visits
    uniq = uniq[keep]
    index = {(int(k[0]), int(k[1])): i for i, k in enumerate(uniq)}
    centres = (uniq + 0.5) * cell_m

    rows, cols, w = [], [], []
    off = 0
    for t in trajs:
        n = len(t)
        seq = []
        for k in keys[off:off + n]:
            nid = index.get((int(k[0]), int(k[1])), -1)
            if nid >= 0 and (not seq or seq[-1] != nid):
                seq.append(nid)
        off += n
        for a, b in zip(seq[:-1], seq[1:]):
            d = float(np.linalg.norm(centres[a] - centres[b])) or cell_m
            rows += [a, b]
            cols += [b, a]
            w += [d, d]
    adj = coo_matrix((w, (rows, cols)), shape=(len(uniq), len(uniq)))
    g = NodeGraph(centres, ref, adj, cell_m, index)
    if verbose:
        print(f"induced network: {g.n_nodes} vertices, {len(w) // 2} edges, "
              f"cell {cell_m:.0f} m")
    return g


def load_osm_graph(place_or_path, cell_m: float = 200.0):  # pragma: no cover
    """Use a real OSM road graph instead, if osmnx is installed.

    Prefer this when it is available: the induced graph above is derived from
    the same trajectories the models see, which the write-up must acknowledge.
    """
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError as e:
        raise ImportError("install osmnx and networkx, or use "
                          "build_node_graph()") from e
    G = (ox.load_graphml(place_or_path) if str(place_or_path).endswith(".graphml")
         else ox.graph_from_place(place_or_path, network_type="drive"))
    G = nx.convert_node_labels_to_integers(G)
    lon = np.array([G.nodes[i]["x"] for i in G.nodes], float)
    lat = np.array([G.nodes[i]["y"] for i in G.nodes], float)
    xy, ref = lonlat_to_enu(np.stack([lon, lat], 1))
    rows, cols, w = [], [], []
    for a, b, d in G.edges(data=True):
        length = float(d.get("length", np.linalg.norm(xy[a] - xy[b])))
        rows += [a, b]
        cols += [b, a]
        w += [length, length]
    adj = coo_matrix((w, (rows, cols)), shape=(len(xy), len(xy)))
    return NodeGraph(xy, ref, adj, cell_m, {})


# ==========================================================================
# node2vec
# ==========================================================================
def random_walks(graph: NodeGraph, n_walks: int = 10, length: int = 40,
                 p: float = 1.0, q: float = 1.0, seed: int = 0):
    """Second-order biased walks (Grover & Leskovec)."""
    rng = np.random.default_rng(seed)
    csr = graph.adj.tocsr()
    nbrs = [csr.indices[csr.indptr[i]:csr.indptr[i + 1]] for i in range(graph.n_nodes)]
    nbr_sets = [set(n.tolist()) for n in nbrs]
    walks = []
    for _ in range(n_walks):
        for start in rng.permutation(graph.n_nodes):
            if len(nbrs[start]) == 0:
                continue
            walk = [int(start)]
            while len(walk) < length:
                cur = walk[-1]
                cand = nbrs[cur]
                if len(cand) == 0:
                    break
                if len(walk) == 1 or (p == 1.0 and q == 1.0):
                    nxt = int(rng.choice(cand))
                else:
                    prev = walk[-2]
                    wts = np.ones(len(cand))
                    for j, c in enumerate(cand):
                        if c == prev:
                            wts[j] = 1.0 / p
                        elif c not in nbr_sets[prev]:
                            wts[j] = 1.0 / q
                    nxt = int(rng.choice(cand, p=wts / wts.sum()))
                walk.append(nxt)
            if len(walk) > 2:
                walks.append(walk)
    return walks


def node2vec_embeddings(graph: NodeGraph, dim: int = 128, window: int = 5,
                        epochs: int = 3, n_walks: int = 10, length: int = 40,
                        neg: int = 5, lr: float = 0.01, seed: int = 0,
                        device: str = "cpu", verbose: bool = True):
    """Skip-gram with negative sampling over biased random walks.

    A compact reimplementation rather than a gensim dependency, so the whole
    pipeline runs from `pip install -r requirements.txt`.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    walks = random_walks(graph, n_walks, length, seed=seed)
    pairs = []
    for w in walks:
        for i, c in enumerate(w):
            for j in range(max(0, i - window), min(len(w), i + window + 1)):
                if j != i:
                    pairs.append((c, w[j]))
    if not pairs:
        return np.zeros((graph.n_nodes, dim), dtype=np.float32)
    pairs = np.asarray(pairs)
    V = graph.n_nodes
    emb = nn.Embedding(V, dim).to(device)
    ctx = nn.Embedding(V, dim).to(device)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ctx.parameters()), lr=lr)
    freq = np.bincount(pairs[:, 1], minlength=V).astype(np.float64) ** 0.75
    freq = torch.from_numpy(freq / freq.sum()).float().to(device)
    idx = torch.from_numpy(pairs).long().to(device)
    bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(len(idx), device=device)
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = idx[perm[s:s + bs]]
            u, v = emb(b[:, 0]), ctx(b[:, 1])
            negs = torch.multinomial(freq, len(b) * neg, replacement=True)
            n = ctx(negs).view(len(b), neg, -1)
            pos = torch.nn.functional.logsigmoid((u * v).sum(-1))
            negl = torch.nn.functional.logsigmoid(
                -(n * u.unsqueeze(1)).sum(-1)).sum(-1)
            loss = -(pos + negl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if verbose:
            print(f"  node2vec epoch {ep + 1}/{epochs} loss="
                  f"{tot / max(len(idx) // bs, 1):.4f}")
    return emb.weight.detach().cpu().numpy()


# ==========================================================================
# network-aware trajectory distances (ST2Vec's ground truths, in form)
# ==========================================================================
def _pairwise(D, a, b):
    return D[np.ix_(a, b)]


# --------------------------------------------------------------------------
# The four measures, written against a plain (len_a x len_b) cost matrix M.
#
# ST2Vec applies the same four measures to the spatial sequence (cost =
# network distance between vertices) and to the temporal sequence (cost =
# |t_i - t_j|), then blends them as lambda*D_S + (1-lambda)*D_T. Writing them
# on M rather than on a vertex-indexed matrix means one implementation serves
# both, so the spatial and temporal ground truths cannot silently diverge.
# --------------------------------------------------------------------------
def tp_M(M):
    """Mean cost between index-aligned elements (resampled to equal length)."""
    A, B = M.shape
    n = min(A, B)
    if n == 0:
        return np.inf
    ia = np.linspace(0, A - 1, n).astype(int)
    ib = np.linspace(0, B - 1, n).astype(int)
    return float(M[ia, ib].mean())


def dita_M(M):
    """Symmetric nearest-element matching, the shape DITA's filter takes."""
    return float(0.5 * (M.min(1).mean() + M.min(0).mean()))


def lcrs_M(M, thresh: float = 1.0):
    """Longest common subsequence of 'shared' elements, as a dissimilarity."""
    A, B = M.shape
    hit = M <= thresh
    dp = np.zeros((A + 1, B + 1), dtype=np.int32)
    for i in range(A):
        row, prev = dp[i + 1], dp[i]
        for j in range(B):
            row[j + 1] = prev[j] + 1 if hit[i, j] else max(prev[j + 1], row[j])
    return float(1.0 - dp[A, B] / max(min(A, B), 1))


def neterp_M(M, gap: float | None = None):
    """Edit distance with Real Penalty; `gap` defaults to the mean cost."""
    if gap is None:
        gap = float(M.mean())
    A, B = M.shape
    prev = np.arange(B + 1, dtype=np.float64) * gap
    for i in range(A):
        cur = np.empty(B + 1)
        cur[0] = prev[0] + gap
        for j in range(B):
            cur[j + 1] = min(prev[j] + M[i, j], prev[j + 1] + gap, cur[j] + gap)
        prev = cur
    return float(prev[B] / max(A + B, 1))


MEASURES_M = {"TP": tp_M, "DITA": dita_M, "LCRS": lcrs_M, "NetERP": neterp_M}


# thin wrappers keeping the original vertex-sequence signatures
def dist_TP(D, a, b):
    return tp_M(_pairwise(D, a, b))


def dist_DITA(D, a, b):
    return dita_M(_pairwise(D, a, b))


def dist_LCRS(D, a, b, thresh: float = 1.0):
    return lcrs_M(_pairwise(D, a, b), thresh)


def dist_NetERP(D, a, b, gap: float | None = None):
    return neterp_M(_pairwise(D, a, b), gap)


MEASURES = {"TP": dist_TP, "DITA": dist_DITA, "LCRS": dist_LCRS,
            "NetERP": dist_NetERP}


def ground_truth_matrix(D, seqs, measure: str, verbose: bool = True):
    """Pairwise ground-truth dissimilarity. O(N^2 L^2): keep N in the hundreds."""
    f = MEASURES[measure]
    n = len(seqs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = f(D, seqs[i], seqs[j])
        if verbose and n > 50 and i % max(n // 5, 1) == 0:
            print(f"    {measure}: {i}/{n}")
    return M


def _time_cost(ta, tb, mode: str):
    """Cost between two timestamps, in hours.

    `absolute`  |t_i - t_j|. This is ST2Vec's reading, and it makes two trips
        a week apart maximally dissimilar however alike their schedules are.
        Under it a raw timestamp is close to sufficient and a periodic code is
        at a structural disadvantage -- it throws away exactly the quantity
        being measured.

    `cyclic`  circular distance on time-of-day combined with day-of-week. This
        is the notion the application in ST2Vec's own Figure 1 needs: a
        rideshare match cares that two people leave at 08:00, not that they
        leave in the same calendar week. It is also the notion the C24 x C7
        GENEO is built for. Reporting both is the experiment: each temporal
        code should win on the ground truth whose structure it matches, which
        is the point rather than a confound.
    """
    if mode == "absolute":
        return np.abs(ta[:, None] - tb[None, :])
    hod = np.abs(((ta % 24.0)[:, None] - (tb % 24.0)[None, :] + 12.0) % 24.0 - 12.0)
    dow = np.abs((((ta // 24.0) % 7.0)[:, None]
                  - ((tb // 24.0) % 7.0)[None, :] + 3.5) % 7.0 - 3.5)
    return hod + 24.0 / 7.0 * dow          # day-of-week in hour-equivalents


def temporal_ground_truth(times, measure: str, verbose: bool = True,
                          mode: str = "absolute"):
    """The same four measures applied to the temporal sequences."""
    f = MEASURES_M[measure]
    n = len(times)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = f(_time_cost(times[i], times[j], mode))
        if verbose and n > 50 and i % max(n // 5, 1) == 0:
            print(f"    {measure} (temporal/{mode}): {i}/{n}")
    return out


def hit_ratio_at_k(pred_sim: np.ndarray, truth_dist: np.ndarray, k: int = 10,
                   seed: int = 0, tol: float = 1e-9):
    """HR@k: overlap of the model's top-k with the ground truth's top-k.

    Tie-aware. Repeated trips snap to identical vertex sequences, and pairs
    across disconnected parts of a graph all share the same capped distance,
    so both the ground truth and a model's similarities contain exact ties.
    Breaking them by array index -- what a plain argsort does -- makes the
    result depend on the order of the data rather than on the embedding, and
    identically for every model, which is how it surfaced: different vertex
    features scored the same HR to three decimals.

    Here a retrieved item counts as a hit if it is at least as close as the
    true k-th neighbour (so every member of a tied group at the boundary is
    acceptable), and the model's own ties are broken uniformly at random with
    a fixed seed. With no ties this reduces exactly to the usual definition.
    """
    n = len(pred_sim)
    rng = np.random.default_rng(seed)
    ps = pred_sim.astype(np.float64).copy()
    td = truth_dist.astype(np.float64).copy()
    np.fill_diagonal(ps, -np.inf)
    np.fill_diagonal(td, np.inf)
    k = min(k, n - 1)
    hits = []
    for i in range(n):
        # random tie-breaking among the model's equal similarities
        order = np.lexsort((rng.random(n), -ps[i]))
        top = order[:k]
        kth = np.sort(td[i])[k - 1]                   # true k-th distance
        acceptable = td[i] <= kth + tol
        hits.append(min(int(acceptable[top].sum()), k) / k)
    return float(np.mean(hits))


def tie_fraction(truth_dist: np.ndarray, k: int = 10, tol: float = 1e-9):
    """Share of queries whose true top-k boundary falls inside a tied group --
    reported alongside HR@k so a reader knows how much the fix matters."""
    td = truth_dist.astype(np.float64).copy()
    np.fill_diagonal(td, np.inf)
    k = min(k, len(td) - 1)
    out = 0
    for row in td:
        kth = np.sort(row)[k - 1]
        out += int((np.abs(row - kth) <= tol).sum() > 1)
    return out / len(td)
