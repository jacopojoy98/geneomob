"""Experiment A -- controlled synthetic validation (Hypotheses 1, 2).

Destination-displacement regression on biased random walks toward latent hubs.
GEO tokenizer + equivariant aggregator vs a matched-capacity GRU on raw
displacements. Reports

  (i)  equivariance error  E_theta || pred(R.T) - R pred(T) ||
  (ii) validation error vs training-set size
  (iii) explanation-equivariance error (the Crabbe & van der Schaar check the
       proposal cites but never runs)
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..config import save_results
from ..data import make_synthetic
from ..encoders import DisplacementGEO
from ..equivariance import model_equivariance_error, explanation_equivariance_error
from ..geo import lonlat_to_enu
from ..models import BaselineRegressor, EquivariantAggregator, pad_batch, set_seed


def _tokens(geo, xys):
    return [geo.encode_disp(np.diff(x, axis=0)).astype(np.float32) for x in xys]


def _train(model, X, Y, epochs=60, lr=1e-3, bs=128, device="cpu", seed=0):
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
    n = len(X)
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            x, m = pad_batch([X[i] for i in b], device)
            loss = torch.nn.functional.mse_loss(model(x, m), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _rmse(model, X, Y, device="cpu"):
    with torch.no_grad():
        x, m = pad_batch(X, device)
        p = model(x, m).cpu().numpy()
    return float(np.sqrt(((p - Y) ** 2).sum(1).mean()))


def _save(results, out):
    if out:
        save_results(results, out)


def run(sizes=(200, 500, 1000, 2000, 4000), n_val=1000, epochs=60,
        device="cpu", seed=0, out=None, n_attrib=32, n_probe=256):
    """Results are written to `out` after every training-set size, so a run
    that is interrupted still leaves usable partial results."""
    geo = DisplacementGEO(n_scales=8, lam_min=10.0, lam_max=20_000.0,
                          harmonics=(1,), gate=True)

    trajs, Y, _ = make_synthetic(n=max(sizes) + n_val, seed=seed)
    xys = [t.xy(ref=np.array([0.0, 41.15])) for t in trajs]
    scale = np.std(np.concatenate(xys, 0))
    xys = [x / scale for x in xys]
    Y = Y / scale

    va = slice(len(xys) - n_val, len(xys))
    Xva_geo, Yva = _tokens(geo, xys[va]), Y[va]
    Xva_raw = [np.diff(x, axis=0).astype(np.float32) for x in xys[va]]

    results = {"sizes": list(sizes), "geo_rmse": [], "base_rmse": []}
    geo_model = base_model = None
    for n in sizes:
        tr = slice(0, n)
        geo_model = _train(EquivariantAggregator(n_inv=geo.n_inv),
                           _tokens(geo, xys[tr]), Y[tr], epochs, device=device,
                           seed=seed)
        base_model = _train(BaselineRegressor(in_dim=2),
                            [np.diff(x, axis=0).astype(np.float32) for x in xys[tr]],
                            Y[tr], epochs, device=device, seed=seed)
        results["geo_rmse"].append(_rmse(geo_model, Xva_geo, Yva, device))
        results["base_rmse"].append(_rmse(base_model, Xva_raw, Yva, device))
        print(f"n={n:5d}  GEO rmse={results['geo_rmse'][-1]:.4f}  "
              f"baseline rmse={results['base_rmse'][-1]:.4f}", flush=True)
        _save(results, out)

    # ---- (i) model-level equivariance error, on the largest-trained models
    def geo_predict(traj_list):
        with torch.no_grad():
            x, m = pad_batch(_tokens(geo, traj_list), device)
            return geo_model(x, m).cpu().numpy()

    def base_predict(traj_list):
        with torch.no_grad():
            x, m = pad_batch([np.diff(t, axis=0).astype(np.float32)
                              for t in traj_list], device)
            return base_model(x, m).cpu().numpy()

    probe = xys[va][:n_probe]
    results["equivariance_error"] = {
        "geo": model_equivariance_error(geo_predict, probe, seed=seed),
        "baseline": model_equivariance_error(base_predict, probe, seed=seed)}
    print("equivariance:", results["equivariance_error"], flush=True)
    _save(results, out)

    # ---- (iii) explanation equivariance: gradient attribution on positions
    def make_attrib(model, tokenize):
        def attrib(xy):
            x, m = pad_batch([tokenize(xy)], device)
            x.requires_grad_(True)
            model(x, m).sum().backward()
            g = x.grad[0, :len(xy) - 1].cpu().numpy()
            # project token-space gradient back onto the 2-d step direction
            d = np.diff(xy, axis=0)
            w = np.linalg.norm(g, axis=1, keepdims=True)
            return w * d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        return attrib

    def tok_geo(xy):
        return geo.encode_disp(np.diff(xy, axis=0)).astype(np.float32)

    def tok_raw(xy):
        return np.diff(xy, axis=0).astype(np.float32)

    results["explanation_equivariance_error"] = {
        "geo": explanation_equivariance_error(
            make_attrib(geo_model, tok_geo), probe[:n_attrib], seed=seed),
        "baseline": explanation_equivariance_error(
            make_attrib(base_model, tok_raw), probe[:n_attrib], seed=seed)}
    _save(results, out)

    print(json.dumps({k: v for k, v in results.items()
                      if k != "sizes"}, indent=2, default=float))
    _save(results, out)
    return results


if __name__ == "__main__":
    run()
