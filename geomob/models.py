"""Models.

The backbone deliberately mirrors GPE's base model (2-layer bidirectional
LSTM, three dense SiLU+BatchNorm layers, h = 128, NCE contrastive loss with
their positive-pair recipe) so that swapping only the position encoder
reproduces their experimental design and makes numbers comparable to their
Table 4. `TrajEncoder(kind='transformer')` covers their Table 20.

`EquivariantAggregator` implements Eq. (2) of the proposal: scalar weights
from the *invariant* channel only, applied to the equivariant angular channel,
so that pred(g.T) = R_theta pred(T) holds exactly rather than approximately.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# backbones
# ==========================================================================
class PathGATConv(nn.Module):
    """Graph attention over the path graph of a trajectory.

    GPE's Section 5.5 forms, for each trajectory, "a graph of shape '1'" -- a
    path -- and runs GATconv over it. On a path the neighbourhood of vertex i
    is {i-1, i, i+1}, so the layer reduces to attention over three shifted
    copies of the sequence. Implementing it directly avoids a
    torch-geometric dependency and keeps the batched, padded layout used by
    every other backbone here, so the comparison stays like-for-like.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 concat: bool = True, negative_slope: float = 0.2):
        super().__init__()
        self.h, self.dk = heads, out_dim // heads if concat else out_dim
        self.concat = concat
        self.lin = nn.Linear(in_dim, self.h * self.dk, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, self.h, self.dk))
        self.att_dst = nn.Parameter(torch.empty(1, self.h, self.dk))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.slope = negative_slope

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        h = self.lin(x).view(B, L, self.h, self.dk)
        a_dst = (h * self.att_dst).sum(-1)                     # (B, L, H)
        a_src = (h * self.att_src).sum(-1)
        # neighbours on a path: previous, self, next
        shifts = [(-1, "prev"), (0, "self"), (1, "next")]
        logits, values, valid = [], [], []
        for s, _ in shifts:
            hs = torch.roll(h, shifts=s, dims=1)
            asrc = torch.roll(a_src, shifts=s, dims=1)
            ok = torch.ones(B, L, dtype=torch.bool, device=x.device)
            if s == -1:
                ok[:, -1] = False
            elif s == 1:
                ok[:, 0] = False
            if mask is not None:
                ok = ok & torch.roll(mask, shifts=s, dims=1) & mask
            logits.append(F.leaky_relu(a_dst + asrc, self.slope))
            values.append(hs)
            valid.append(ok)
        logit = torch.stack(logits, 0)                          # (3, B, L, H)
        ok = torch.stack(valid, 0).unsqueeze(-1)
        logit = logit.masked_fill(~ok, float("-inf"))
        alpha = torch.softmax(logit, dim=0)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = (alpha.unsqueeze(-1) * torch.stack(values, 0)).sum(0)
        out = out.reshape(B, L, -1) if self.concat else out.mean(2)
        return out


class TrajEncoder(nn.Module):
    """Sequence model over pre-computed position tokens -> trajectory vector.

    `kind` selects the backbone, mirroring GPE's Section 5.5 ablation:
    'lstm' (their base model), 'transformer' (their Table 20), 'gnn' (their
    Table 19, two GATconv layers over the trajectory's path graph).
    """

    def __init__(self, in_dim: int, hidden: int | None = None,
                 out_dim: int | None = None, kind: str = "lstm",
                 layers: int | None = None, heads: int | None = None,
                 dropout: float | None = None):
        # unset arguments come from the run configuration ([model] section)
        from .config import HP
        m = HP["model"]
        hidden = m["hidden"] if hidden is None else hidden
        out_dim = hidden if out_dim is None else out_dim
        layers = m["layers"] if layers is None else layers
        heads = m["heads"] if heads is None else heads
        dropout = m["dropout"] if dropout is None else dropout
        super().__init__()
        self.kind = kind
        if kind == "gnn":
            self.g1 = PathGATConv(in_dim, hidden, heads=heads)
            self.g2 = PathGATConv(hidden, hidden, heads=heads)
            feat = hidden
        elif kind == "lstm":
            self.seq = nn.LSTM(in_dim, hidden, num_layers=layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout)
            feat = 2 * hidden
        elif kind == "transformer":
            self.proj = nn.Linear(in_dim, hidden)
            layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden,
                                               dropout=dropout,
                                               batch_first=True)
            self.seq = nn.TransformerEncoder(layer, layers)
            feat = hidden
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(
            nn.Linear(feat, hidden), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x, mask=None):
        """x: (B, L, in_dim); mask: (B, L) True where valid."""
        if self.kind == "gnn":
            h = F.elu(self.g1(x, mask))
            h = self.g2(h, mask)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
                pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                pooled = h.mean(1)
            return self.head(pooled)
        if self.kind == "lstm":
            h, _ = self.seq(x)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
                pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                pooled = h.mean(1)
        else:
            h = self.seq(self.proj(x),
                         src_key_padding_mask=None if mask is None else ~mask)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
                pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                pooled = h.mean(1)
        return self.head(pooled)


# ==========================================================================
# Experiment A: equivariant aggregator vs unconstrained baseline
# ==========================================================================
class EquivariantAggregator(nn.Module):
    """pred = sum_i s_i(PE(r_1..r_N)) * u(psi_i).

    s_i is a function of the invariant radial channel only -- never of psi --
    which is what makes (3) exact. The angular channel enters linearly.
    """

    def __init__(self, n_inv: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.n_inv = n_inv
        self.gru = nn.GRU(n_inv, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True)
        self.score = nn.Linear(2 * hidden, 1)

    def forward(self, tokens, mask=None):
        """tokens: (B, L, n_inv + 2), angular channel = last two dims."""
        inv, ang = tokens[..., :self.n_inv], tokens[..., self.n_inv:self.n_inv + 2]
        h, _ = self.gru(inv)
        s = self.score(h).squeeze(-1)                     # (B, L), invariant
        if mask is not None:
            s = s.masked_fill(~mask, 0.0)
        return torch.einsum("bl,blc->bc", s, ang)         # (B, 2), equivariant


class BaselineRegressor(nn.Module):
    """Matched-capacity unconstrained control: raw displacements -> GRU -> MLP."""

    def __init__(self, in_dim: int = 2, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x, mask=None):
        h, _ = self.gru(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.out(pooled)


# ==========================================================================
# Experiment D: relative heading / cyclic-time attention
# ==========================================================================
class RelativeGeoAttention(nn.Module):
    """Rotate query/key pairs by heading psi_i and by the cyclic time phase
    before the dot product, so attention scores depend only on *relative*
    heading and relative time-of-day/week (a RoPE-style construction with the
    geometric group in place of sequence index)."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % (2 * heads) == 0
        self.h, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _rotate(x, phase):
        """x: (B,H,L,Dk) with Dk even; phase: (B,L) -> pairwise rotation."""
        B, H, L, Dk = x.shape
        x = x.view(B, H, L, Dk // 2, 2)
        c = torch.cos(phase)[:, None, :, None]
        s = torch.sin(phase)[:, None, :, None]
        xr = torch.stack([x[..., 0] * c - x[..., 1] * s,
                          x[..., 0] * s + x[..., 1] * c], dim=-1)
        return xr.view(B, H, L, Dk)

    def forward(self, x, psi, tphase, mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = lambda z: z.view(B, L, self.h, self.dk).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        phase = psi + tphase
        q, k = self._rotate(q, phase), self._rotate(k, phase)
        att = (q @ k.transpose(-2, -1)) / self.dk ** 0.5
        if mask is not None:
            att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


# ==========================================================================
# losses
# ==========================================================================
def nce_loss(z_anchor, z_pos, temperature: float = 1.0):
    """GPE's contrastive objective: in-batch negatives, cosine similarity."""
    a = F.normalize(z_anchor, dim=-1)
    p = F.normalize(z_pos, dim=-1)
    logits = a @ p.T / temperature
    target = torch.arange(len(a), device=a.device)
    return F.cross_entropy(logits, target)


# ==========================================================================
# batching
# ==========================================================================
def pad_batch(seqs, device="cpu"):
    """list of (L_i, D) arrays -> (B, Lmax, D) tensor + bool mask."""
    L = max(len(s) for s in seqs)
    D = seqs[0].shape[1]
    x = np.zeros((len(seqs), L, D), dtype=np.float32)
    m = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        m[i, :len(s)] = True
    return (torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
