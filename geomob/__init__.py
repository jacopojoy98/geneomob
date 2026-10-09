"""Equivariant tokenization of human mobility trajectories: experiment code.

Modules
    geo           local metric frames and the action of G = SE(2) x Theta
    encoders      GPE, Space2Vec, torus regimes (i)/(ii)/(iii), displacement
                  GEO, cyclic-time GENEO, non-equivariant baselines
    equivariance  encoder / model / explanation equivariance audits
    codebook      discrete codebooks and the corrected Experiment C
    data          synthetic generator and Porto/T-drive/GeoLife/AIS loaders
    metrics       GPE's MR/MRR/MP/KP protocol, mobility-statistic probes
    models        GPE-comparable backbone, equivariant aggregator (torch)
"""
__version__ = "0.8.0"
