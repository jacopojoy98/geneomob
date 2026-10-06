"""Trajectory-user linking (TUL), and next-location prediction.

Personal commute corpora rarely carry trip-purpose or mode labels, but they do
carry a user column, and "which of these people made this trip" is a standard,
well-posed downstream task. It is also a better test of a tokenizer than
self-similarity ranking, because the label is not recoverable from the trace's
own geometry by construction.

Two protocols, and the difference matters:

  `split='trip'`   -- held-out *trips* from users seen in training. This is
                      classic TUL. Scores are high, and a large part of that
                      is the model learning where each person lives.
  `split='day'`    -- train on the first weeks, test on the last. Harder and
                      more honest: it cannot be solved by memorising a trip
                      that is nearly duplicated in the training set.

Note what this task rewards. TUL is *helped* by an absolute-position code that
pins down a home address, so a tokenizer that deliberately discards absolute
position will not win it. Report it as a check that the tokens retain
person-level information (Hypothesis 4), not as evidence for equivariance.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from ..config import save_results
from ..models import TrajEncoder, pad_batch, set_seed


def _labels(trajs):
    users = sorted({t.uid for t in trajs})
    idx = {u: i for i, u in enumerate(users)}
    return np.array([idx[t.uid] for t in trajs]), users


def split_by_trip(trajs, frac_train=0.8, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trajs))
    k = int(frac_train * len(trajs))
    return [trajs[i] for i in order[:k]], [trajs[i] for i in order[k:]]


def split_by_day(trajs, frac_train=0.8):
    """Temporal split: the last (1 - frac_train) of the calendar period is the
    test set. Near-duplicate commutes cannot straddle the boundary."""
    t0 = np.array([t.t[0] for t in trajs])
    cut = np.quantile(t0, frac_train)
    return ([t for t, s in zip(trajs, t0) if s <= cut],
            [t for t, s in zip(trajs, t0) if s > cut])


def run_tul(trajs, tokenizers, split="day", epochs=15, bs=64, lr=1e-3,
            device="cpu", seed=0, min_trips_per_user=6, out=None):
    """`tokenizers` is a dict name -> Tokenizer from exp_similarity."""
    trajs = [t for t in trajs if t.uid is not None]
    counts: dict[str, int] = {}
    for t in trajs:
        counts[t.uid] = counts.get(t.uid, 0) + 1
    trajs = [t for t in trajs if counts[t.uid] >= min_trips_per_user]
    if not trajs:
        raise ValueError("no users with enough trips; lower min_trips_per_user")

    y, users = _labels(trajs)
    train, test = (split_by_day(trajs) if split == "day"
                   else split_by_trip(trajs, seed=seed))
    ytr = np.array([users.index(t.uid) for t in train])
    yte = np.array([users.index(t.uid) for t in test])
    majority = float(np.bincount(yte, minlength=len(users)).max() / len(yte))
    print(f"TUL: {len(users)} users, {len(train)} train / {len(test)} test "
          f"trips, split='{split}', majority-class baseline {majority:.3f}")

    res = {"n_users": len(users), "n_train": len(train), "n_test": len(test),
           "split": split, "majority_baseline": majority, "models": {}}

    for name, tok in tokenizers.items():
        set_seed(seed)
        tok.fit(train)
        enc = TrajEncoder(tok.dim, out_dim=128).to(device)
        head = nn.Linear(128, len(users)).to(device)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                                lr=lr)
        Xtr = [tok(t) for t in train]
        Xte = [tok(t) for t in test]
        ytr_t = torch.from_numpy(ytr).long().to(device)
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            perm = rng.permutation(len(Xtr))
            for s in range(0, len(perm) - 1, bs):
                b = perm[s:s + bs]
                if len(b) < 2:
                    continue
                x, m = pad_batch([Xtr[i] for i in b], device)
                loss = nn.functional.cross_entropy(head(enc(x, m)), ytr_t[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        enc.eval()
        with torch.no_grad():
            x, m = pad_batch(Xte, device)
            logits = head(enc(x, m)).cpu().numpy()
        top1 = float((logits.argmax(1) == yte).mean())
        k = min(5, len(users))
        topk = float(np.mean([yte[i] in np.argsort(-logits[i])[:k]
                              for i in range(len(yte))]))
        res["models"][tok.name] = {"acc@1": top1, f"acc@{k}": topk,
                                   "dim": tok.dim}
        print(f"  {tok.name:26s} acc@1={top1:.3f}  acc@{k}={topk:.3f}")

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res
