"""Experiments B and G -- similarity benchmark, and cross-geography transfer.

Backbone, loss and augmentation follow GPE's Section 5.1 exactly, so the
MR / MRR / MP / KP@10 numbers produced here sit in the same table as their
Tables 10-13, and the train-on-A-test-on-B matrix sits in the same table as
their Tables 14-18.

Transfer regimes (Experiment G):
  zero_shot   everything frozen
  refit_scale GEO model: re-fit only the geography-dependent scale field
              (lambda_k of phi_loc) on city B; baseline: re-fit only its
              normalisation/output head. This is the *mechanistic* claim --
              that separating geography-independent from geography-dependent
              channels is what makes transfer cheap. GPE already shows that
              bounded periodic codes transfer; this is the part that is new.
  finetune    small budget of city-B trajectories, both models
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..config import save_results
from ..data import (Traj, augment, bbox, load_dataset, odd_even_split,
                    space_shift)
from ..encoders import (GPE, DisplacementGEO, GridEmbed, LearnedFourier,
                        PatchworkLattice, RawXY, Space2Vec, TorusLattice)
from ..metrics import similarity_metrics
from ..models import TrajEncoder, nce_loss, pad_batch, set_seed


# --------------------------------------------------------------------------
# tokenizers: how a trajectory becomes a (L, D) token sequence
# --------------------------------------------------------------------------
class Tokenizer:
    """Concatenates the channels that a given configuration turns on."""

    def __init__(self, loc=None, disp=None, time=None, speed=None,
                 name="tok"):
        self.loc, self.disp, self.time = loc, disp, time
        self.speed, self.name = speed, name

    @property
    def dim(self):
        return sum(0 if c is None else c.dim
                   for c in (self.loc, self.disp, self.time, self.speed))

    def fit(self, trajs, force: bool = False):
        """Fit the geography-dependent parts: the local frame and, for the
        adaptive regimes, the scale field.

        `force=True` re-fits an already-fitted channel. Without it a second
        call is a no-op, which would silently make Experiment G's
        `refit_scale` regime identical to zero-shot -- the exact confound this
        experiment exists to avoid.
        """
        ll = np.concatenate([t.lonlat for t in trajs])
        ch = self.loc
        if ch is None:
            return self
        if hasattr(ch, "ref_deg") and (ch.ref_deg is None or force):
            ch.ref_deg = ll.mean(0)
        if hasattr(ch, "fit") and (force or not getattr(ch, "_fitted", False)):
            ch.fit(ll)
        return self

    def fit_scale_only(self, trajs):
        """Experiment G regime (2): re-fit ONLY the geography-dependent scale
        field (and local frame). Encoder weights and transformer stay frozen.

        For GPE there is nothing to re-fit -- its frequency ladder is fixed in
        angular units worldwide -- so this regime is a no-op for it by
        construction, which is itself the result.
        """
        return self.fit(trajs, force=True)

    def refittable(self) -> bool:
        return self.loc is not None and (
            hasattr(self.loc, "_fitted") or hasattr(self.loc, "ref_deg"))

    def __call__(self, traj: Traj) -> np.ndarray:
        parts = []
        if self.loc is not None:
            parts.append(self.loc.encode(traj.lonlat))
        if self.disp is not None:
            from ..geo import lonlat_to_enu
            xy = lonlat_to_enu(traj.lonlat, getattr(self.loc, "ref_deg", None))[0]
            d = self.disp.encode_disp(np.diff(xy, axis=0))
            parts.append(np.vstack([np.zeros((1, d.shape[1])), d]))
        if self.speed is not None:
            from ..geo import lonlat_to_enu
            xy = lonlat_to_enu(traj.lonlat, getattr(self.loc, "ref_deg", None))[0]
            sp = self.speed.encode_speed(np.diff(xy, axis=0), np.diff(traj.t))
            parts.append(np.vstack([np.zeros((1, sp.shape[1])), sp]))
        if self.time is not None:
            parts.append(self.time.encode(t=traj.t))
        return np.concatenate(parts, axis=1).astype(np.float32)


def build_tokenizers(which=("gpe", "geo_full", "xy", "grid", "fourier",
                            "space2vec")):
    from ..encoders import CyclicTimeGENEO
    reg = {
        "gpe": lambda: Tokenizer(loc=GPE(h=128), name="GPE"),
        "xy": lambda: Tokenizer(loc=RawXY(normalize=True), name="XY(SS)"),
        "grid": lambda: Tokenizer(loc=GridEmbed(150.0), name="Grid"),
        "fourier": lambda: Tokenizer(loc=LearnedFourier(128), name="TriW"),
        "space2vec": lambda: Tokenizer(loc=Space2Vec(8), name="Space2Vec"),
        "geo_disp": lambda: Tokenizer(disp=DisplacementGEO(), name="GEO-disp"),
        "geo_spatial": lambda: Tokenizer(loc=TorusLattice(),
                                         disp=DisplacementGEO(),
                                         name="GEO(i)+disp"),
        # Dimension-matched to GPE(h=128). Comparing a 38-d tokenizer against
        # a 128-d one and reading the difference as "symmetry helps" (or
        # "hurts") confounds capacity with structure; this is the config to
        # quote in any head-to-head table.
        "geo_spatial128": lambda: Tokenizer(
            loc=TorusLattice(lams=tuple(50.0 * 2.0 ** np.arange(16))),
            disp=DisplacementGEO(n_scales=30, lam_min=10.0, lam_max=50_000.0),
            name="GEO(i)+disp [d=128]"),
        "geo_full": lambda: Tokenizer(loc=TorusLattice(),
                                      disp=DisplacementGEO(),
                                      time=CyclicTimeGENEO(), name="GEO(i)+disp+time"),
        "geo_patch": lambda: Tokenizer(loc=PatchworkLattice(),
                                       disp=DisplacementGEO(),
                                       time=CyclicTimeGENEO(), name="GEO(iii)+disp+time"),
    }
    return {k: reg[k]() for k in which}


# --------------------------------------------------------------------------
# training / evaluation
# --------------------------------------------------------------------------
def train_contrastive(tok, trajs, kind="lstm", epochs=5, bs=128, lr=1e-3,
                      device="cpu", seed=0, model=None, freeze=False):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = model or TrajEncoder(tok.dim, kind=kind).to(device)
    if freeze:
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = rng.permutation(len(trajs))
        tot, nb = 0.0, 0
        for s in range(0, len(perm) - bs + 1, bs):
            b = [trajs[i] for i in perm[s:s + bs]]
            xa, ma = pad_batch([tok(t) for t in b], device)
            xp, mp = pad_batch([tok(augment(t, rng)) for t in b], device)
            loss = nce_loss(model(xa, ma), model(xp, mp))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item()
            nb += 1
        print(f"  epoch {ep + 1}/{epochs} loss={tot / max(nb, 1):.4f}")
    return model


@torch.no_grad()
def embed(model, tok, trajs, device="cpu", bs=256):
    model.eval()
    out = []
    for s in range(0, len(trajs), bs):
        x, m = pad_batch([tok(t) for t in trajs[s:s + bs]], device)
        out.append(model(x, m).cpu().numpy())
    model.train()
    return np.concatenate(out)


def evaluate(model, tok, test, device="cpu", k=10):
    halves = [odd_even_split(t) for t in test]
    A = embed(model, tok, [h[0] for h in halves], device)
    B = embed(model, tok, [h[1] for h in halves], device)
    full = embed(model, tok, test, device)
    return similarity_metrics(A, B, k=k, emb_full=full)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def _split(trajs, dataset, split, n_train, n_test, seed):
    """Hold out test data.

    On a personal-commute corpus a random trip-level split leaks badly: the
    same person's Tuesday and Wednesday commutes are near-duplicates, so a
    model can score well by recognising the individual instead of
    representing movement. `split='user'` is therefore the default whenever
    the corpus carries a user column.
    """
    has_users = any(t.uid is not None for t in trajs)
    if split == "user" and has_users:
        from ..csvdata import user_disjoint_split
        train, test = user_disjoint_split(trajs, seed=seed)
    else:
        if split == "user" and not has_users:
            print("note: no user column in this corpus; falling back to a "
                  "random trip-level split")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(trajs))
        k = min(n_train, len(trajs) - 1)
        train = [trajs[i] for i in order[:k]]
        test = [trajs[i] for i in order[k:]]
    return train[:n_train], test[:n_test]


# --------------------------------------------------------------------------
# Experiment B
# --------------------------------------------------------------------------
def run_benchmark(dataset="synthetic", path=None, n_train=4000, n_test=1000,
                  which=("gpe", "geo_spatial", "xy", "grid"), kind="lstm",
                  epochs=5, device="cpu", seed=0, out=None, split="user",
                  **load_kw):
    trajs = load_dataset(dataset, path, **load_kw)
    if dataset.startswith("synthetic"):
        trajs = trajs[:n_train + n_test]
    train, test = _split(trajs, dataset, split, n_train, n_test, seed)
    print(f"{len(train)} train / {len(test)} test trajectories "
          f"(split='{split}')")

    res = {}
    for key, tok in build_tokenizers(which).items():
        if tok.time is not None:
            print("WARNING: the absolute-time channel leaks trajectory "
                  "identity in the odd/even self-similarity protocol -- both "
                  "halves carry the same timestamps, so a perfect MRR here "
                  "measures the leak, not the tokenizer. GPE's protocol is "
                  "spatial; use 'geo_spatial' for a fair comparison, and test "
                  "the time channel on a downstream task instead.")
        print(f"[{tok.name}] dim={tok.dim}")
        tok.fit(train)
        model = train_contrastive(tok, train, kind=kind, epochs=epochs,
                                  device=device, seed=seed)
        res[tok.name] = evaluate(model, tok, test, device)
        print(f"  {res[tok.name]}")
    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


# --------------------------------------------------------------------------
# Experiment G
# --------------------------------------------------------------------------
def run_transfer(src="synthetic", dst="synthetic", src_path=None, dst_path=None,
                 which=("gpe", "geo_full", "xy", "grid"), n_train=4000,
                 n_test=1000, n_finetune=500, epochs=5, ft_epochs=2,
                 device="cpu", seed=0, dst_lat=None, dst_scale=None, out=None,
                 split="user", **load_kw):
    """Controlled synthetic destinations, for isolating one confound at a time.

    `dst_lat`   -- the source city moved to another latitude with its metric
                   shape preserved. Isolates the cos(lat) factor. Note that
                   the relative/temporal channels are latitude-invariant by
                   construction, so this test does NOT exercise re-fitting;
                   it separates metric from angular encodings.
    `dst_scale` -- the source city rescaled about its centroid, i.e. a
                   different characteristic block size and density. This is
                   what the geography-dependent scale field exists for, so it
                   is the test that actually exercises regime (ii)/(iii)
                   re-fitting. Use `--which geo_patch` for it to mean anything:
                   regime (i) has a fixed lambda ladder and nothing to re-fit.
    """
    from ..geo import enu_to_lonlat, lonlat_to_enu, reproject_to_latitude

    src_trajs = load_dataset(src, src_path, **load_kw)
    if dst_scale is not None:
        dst_trajs = []
        for t in src_trajs:
            xy, ref = lonlat_to_enu(t.lonlat)
            c = xy.mean(0)
            dst_trajs.append(Traj(enu_to_lonlat((xy - c) * dst_scale + c, ref), t.t))
        dst_name = f"{src}x{dst_scale}"
    elif dst_lat is not None:
        dst_trajs = [Traj(reproject_to_latitude(t.lonlat, dst_lat), t.t)
                     for t in src_trajs]
        dst_name = f"{src}@lat{dst_lat}"
    else:
        dst_trajs = load_dataset(dst, dst_path, **load_kw)
        dst_name = dst
    A_train, A_test = _split(src_trajs, src, split, n_train, n_test, seed)
    B_ft, B_test = _split(dst_trajs, dst, split, n_finetune, n_test, seed)

    res = {"src": src, "dst": dst_name}
    for key, tok in build_tokenizers(which).items():
        tok.fit(A_train)
        print(f"[{tok.name}] pretraining on {src}")
        model = train_contrastive(tok, A_train, epochs=epochs, device=device,
                                  seed=seed)
        row = {"in_domain": evaluate(model, tok, A_test, device)}

        # (1) zero-shot: for non-transferable baselines, GPE's SS remap
        tok_zs = tok
        if isinstance(tok.loc, (RawXY, GridEmbed)):
            shifted = space_shift(B_test, bbox(B_test), bbox(A_train))
            row["zero_shot"] = evaluate(model, tok_zs, shifted, device)
            row["zero_shot_protocol"] = "space-shift (SS)"
        else:
            row["zero_shot"] = evaluate(model, tok_zs, B_test, device)
            row["zero_shot_protocol"] = "native"

        # (2) re-fit only the geography-dependent scale field
        import copy
        tok_rs = copy.deepcopy(tok)
        tok_rs.fit_scale_only(B_ft)
        row["refit_scale"] = evaluate(model, tok_rs, B_test, device)
        row["refit_changed_tokenizer"] = bool(
            not np.allclose(tok(B_test[0]), tok_rs(B_test[0])))

        # (3) small-budget full fine-tune
        model_ft = copy.deepcopy(model)
        train_contrastive(tok_rs, B_ft, epochs=ft_epochs, device=device,
                          seed=seed, model=model_ft)
        row["finetune"] = evaluate(model_ft, tok_rs, B_test, device)

        zs, rs, ft = (row["zero_shot"]["MRR"], row["refit_scale"]["MRR"],
                      row["finetune"]["MRR"])
        row["gap_closed_by_refit"] = float((rs - zs) / (ft - zs)) if ft > zs else None
        res[tok.name] = row
        print(f"  {tok.name}: zs={zs:.3f} refit={rs:.3f} ft={ft:.3f} "
              f"gap closed={row['gap_closed_by_refit']}")

    print(json.dumps(res, indent=2, default=float))
    if out:
        save_results(res, out)
    return res


if __name__ == "__main__":
    run_benchmark()
