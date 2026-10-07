"""Turn results JSON into manuscript figures.

    python -m geomob.figures results/ -o figures/
    python -m geomob.figures results/audit.json -o figures/ --format pdf png

File type is detected from content, not filename, so renamed or merged result
files still work. Every figure is written as PDF (for \\includegraphics) and
PNG (for quick viewing), and a `figures.tex` snippet is emitted with a ready
figure environment and caption for each one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import builders as B
from .style import save, use_paper_style

# Detection rules: (predicate on the parsed JSON, builder, default figure name)
RULES = [
    (lambda d: d.get("relocate") is True and "datasets" in d, "relocate", None),
    (lambda d: "tasks" in d and "backbones" in d, "downstream", None),
    (lambda d: "parts" in d and "datasets" in d, "gpe_suite", None),
    (lambda d: "encoders" in d and "latitude" in d, "audit", None),
    (lambda d: "geo_rmse" in d or ("models" in d and "sizes" in d),
     "expa", None),
    (lambda d: "test_1_lattice_period" in d, "tokens", None),
    (lambda d: "models" in d and "majority_baseline" in d, "tul", None),
    (lambda d: "cells" in d and "kinds" in d, "arch", None),
    (lambda d: "runs" in d and "expected" in d, "anomaly", None),
    (lambda d: "grid" in d and "time_kinds" in d, "lambda", None),
    (lambda d: "rows" in d and "n_nodes" in d, "roadnet", None),
    (lambda d: any(isinstance(v, dict) and "zero_shot" in v
                   for v in d.values()), "transfer", None),
    (lambda d: any(isinstance(v, dict) and "MRR" in v for v in d.values()),
     "similarity", None),
    (lambda d: any(isinstance(v, dict) and "radius_of_gyration" in v
                   for v in d.values()), "probe", None),
]

CAPTIONS = {
    "fig_relocate":
        "Relocated-city test. Each model is trained once on the city where it "
        "is; the test set is then moved as a rigid body to another latitude, "
        "turned about its centre, or shifted east, leaving every distance and "
        "angle on the ground unchanged. All rows are zero-shot. The "
        "canonical-frame rows re-fit their frame (centroid and principal axes) "
        "on the moved city without labels and are flat by construction; the "
        "grey row feeds GPE the same canonical coordinates, separating the "
        "effect of the frame from that of the encoding.",
    "fig_paired":
        "GEO against GPE with matched channels, one row per setting (dataset, "
        "task, sequence model, metric). The point is the mean relative "
        "improvement of GEO over seeds, the bar its 95\\% interval; filled "
        "points are significant in a paired $t$-test over seeds after Holm "
        "correction across all settings.",
    "fig_by_dataset":
        "The same task on every dataset, one panel per dataset. Bars are the "
        "encoders, hatching the sequence model. Each panel has its own scale: "
        "compare encoders within a panel, not heights across panels.",
    "fig_similarity_by_dataset":
        "Local trajectory-similarity metrics for each dataset separately "
        "(the averages over datasets are in the GPE-layout table).",
    "fig_cd":
        "Critical-difference diagram. Methods are ranked within each block "
        "(one dataset, task, sequence model and metric; rank 1 is best) and "
        "placed at their average rank. Methods joined by a thick bar are not "
        "significantly different (Nemenyi test, $\\alpha=0.05$); the CD bar "
        "shows the rank gap needed for significance. Blocks that share a "
        "dataset are not independent, so the test is indicative.",
    "fig_equivariance_error":
        "Encoder equivariance error under translation and rotation, measured "
        "against the analytic representation where one exists and against the "
        "best-fit orthogonal representation otherwise. GPE is already exactly "
        "translation-equivariant; the displacement channel adds exact rotation "
        "equivariance.",
    "fig_latitude":
        "Distortion of each code when the same trip, with its metric shape "
        "preserved, is relocated to a different latitude. Dotted lines re-fit "
        "the encoder's local frame, which is available to metric codes and not "
        "to angular ones.",
    "fig_noise":
        "Stability of each code under positional noise. Encoders whose finest "
        "wavelength falls below the GPS error floor decorrelate.",
    "fig_experiment_a":
        "Downstream-task error against training-set size, and equivariance "
        "error of the prediction and of the attribution map under rotation of "
        "the input. Backbone, head and optimisation are identical across rows; "
        "only the tokenizer changes. Greyed rows never beat the constant "
        "predictor and are not usable as a comparison.",
    "fig_experiment_a_destination":
        "Destination prediction: given the observed prefix of a trip, predict "
        "the remaining displacement. The target rotates with the input, so the "
        "required behaviour is equivariance, $\\rho=R_\\theta$. Left: error "
        "against training-set size, with the constant predictor marked. Right: "
        "relative error of the prediction and of the attribution map under "
        "rotation. Backbone, head and optimisation are identical across rows; "
        "only the tokenizer changes.",
    "fig_experiment_a_eta":
        "Arrival-time estimation: given the observed prefix of a trip, predict "
        "the remaining travel time. The target is unchanged by rotation, so "
        "the required behaviour is invariance, $\\rho=I$. Otherwise identical "
        "in setup to the destination task, and reported alongside it because a "
        "construction tested on only one symmetry type is under-tested.",
    "fig_architectures_": None,
    "fig_token_identity":
        "Discrete-token preservation under translation. An equivariant code "
        "preserves its token exactly when the translation is a lattice period "
        "(left) and is not expected to otherwise (right).",
    "fig_equivariant_codebook":
        "Equivariance of the discrete tokens. Left: measured agreement with "
        "the permutation predicted by $\\rho(v)$, against translation "
        "magnitude; translations that are multiples of the cell size are exact "
        "by construction. Right: agreement of the induced index map on points "
        "it was not fitted on, which distinguishes a group action from an "
        "empirical coincidence.",
    "fig_gauge_defect":
        "Deviation of the separable-warp regime from its first-order gauge "
        "prediction, against the predicted quadratic law. Below the shaded "
        "region the measurement is limited by the resolution of the "
        "interpolant rather than by the construction.",
    "fig_probe":
        "Held-out probe $R^2$ for mobility statistics recovered from pooled "
        "tokens. Each channel loses what it was designed to lose and the "
        "concatenation recovers both.",
    "fig_similarity":
        "Trajectory-similarity performance under the odd/even self-similarity "
        "protocol.",
    "fig_transfer":
        "Transfer to an unseen corpus under three regimes. The claim under "
        "test is that re-fitting only the geography-dependent scale field "
        "closes most of the gap between zero-shot and full fine-tuning.",
    "fig_architectures":
        "Tokenizer performance across sequence architectures. A stable "
        "ordering across backbones indicates that the effect belongs to the "
        "representation rather than to an interaction with one model.",
    "fig_roadnetwork":
        "Road-network vertex features evaluated by HR@10 against "
        "network-aware ground-truth distances. Concatenated rows split the "
        "dimension between the two feature sources so that the comparison is "
        "not a capacity increase.",
    "fig_anomaly":
        "Anomaly detection AUROC by type. Rows below the rule are one-line "
        "statistical detectors, included as a guard: where one of them matches "
        "the tokenizer, that column says nothing about the representation.",
    "fig_anomaly_orientation":
        "Orientation control. A 45-degree rotation breaks alignment with a "
        "square street grid while a 90-degree rotation preserves it, and both "
        "displace absolute positions equally, so only a positive gap between "
        "them is evidence of genuine orientation sensitivity.",
    "fig_lambda":
        "Left: retrieval fidelity against the spatial/temporal blending weight "
        "of the ground truth. Right: relative drift of the embedding when every "
        "timestamp is shifted by one week, a period of the cyclic group.",
    "fig_downstream_tul":
        "Trajectory-user linking with each encoding fed to an LSTM or a "
        "Transformer. Dashed lines are the majority-class and random baselines.",
    "fig_downstream_eta":
        "Travel-time estimation. Paths are resampled at a fixed arc-length "
        "step and the time channel carries only the departure time, so neither "
        "the number of points nor the timestamps reveal the duration.",
    "fig_downstream_mode":
        "Transportation-mode detection on the labelled GeoLife users, "
        "user-disjoint split, with a hand-crafted kinematic baseline.",
    "fig_downstream_mode_per_class":
        "Per-class F1 for mode detection.",
    "fig_downstream_anomaly":
        "Anomaly detection AUROC per injected type, for an autoencoder trained "
        "on clean trajectories only. Rows below the rule are one-line "
        "statistical detectors.",
    "fig_gpe_local":
        "Local trajectory-similarity metrics averaged over datasets, per "
        "backbone (layout of GPE Tables 4 and 8).",
    "fig_gpe_global":
        "Zero-shot cross-dataset MRR: train on the row, test on the column; "
        "the diagonal is the local result.",
    "fig_gpe_time":
        "Training time, split into position fitting and model training.",
    "fig_gpe_dims":
        "Local MRR as the embedding size varies.",
    "fig_gpe_road":
        "Road-network HR@10 with vertex features from node2vec, GPE, GEO or "
        "their half-dimension concatenation, supervised inside an ST2Vec-style "
        "model.",
    "fig_tul":
        "Trajectory-user linking accuracy, reported against the majority-class "
        "baseline.",
}


TABLES: list[str] = []      # LaTeX tables collected while building figures


def build_from(path: Path):
    """Return (name, figure) pairs for one results file."""
    data = json.loads(Path(path).read_text())
    stem = Path(path).stem
    for pred, kind, _ in RULES:
        try:
            if not pred(data):
                continue
        except Exception:
            continue
        if kind == "relocate":
            from . import builders_compare as C
            TABLES.append(C.tex_relocate(data))
            return C.fig_relocate(data)
        if kind == "downstream":
            from . import builders_suite as S
            TABLES.append(S.tex_downstream(data))
            return S.fig_downstream(data)
        if kind == "gpe_suite":
            from . import builders_suite as S
            TABLES.append(S.tex_gpe_suite(data))
            return S.fig_gpe_suite(data)
        if kind == "audit":
            return (B.fig_equivariance_error(data) + B.fig_latitude(data)
                    + B.fig_noise(data))
        if kind == "expa":
            return B.fig_experiment_a(data)
        if kind == "tokens":
            return (B.fig_token_identity(data) + B.fig_gauge_defect(data)
                    + B.fig_equivariant_codebook(data))
        if kind == "tul":
            return B.fig_tul(data)
        if kind == "arch":
            return B.fig_architectures(data)
        if kind == "anomaly":
            return B.fig_anomaly(data)
        if kind == "lambda":
            return B.fig_lambda(data)
        if kind == "roadnet":
            return B.fig_roadnetwork(data)
        if kind == "transfer":
            return B.fig_transfer(data, name=f"fig_{stem}")
        if kind == "similarity":
            return B.fig_similarity(data, name=f"fig_{stem}")
        if kind == "probe":
            return B.fig_probe(data)
    return []


def caption_for(name: str) -> str:
    """Longest-prefix lookup, so content-tagged names still find a caption.

    Figure names carry a discriminator (`fig_experiment_a_destination`,
    `fig_roadnetwork_osm`) so that two runs of the same experiment do not
    overwrite each other; captions are keyed on the base name.
    """
    hits = [k for k in CAPTIONS if name.startswith(k)]
    return CAPTIONS[max(hits, key=len)] if hits else name.replace("_", " ")


def write_tex(names, out_dir: Path, width=r"\linewidth"):
    lines = ["% Generated by geomob.figures -- paste into the manuscript.",
             "% Requires \\usepackage{graphicx}.", ""]
    for n in names:
        cap = caption_for(n)
        lines += [r"\begin{figure}[t]", r"  \centering",
                  rf"  \includegraphics[width={width}]{{figures/{n}.pdf}}",
                  rf"  \caption{{{cap}}}", rf"  \label{{{n.replace('_', ':')}}}",
                  r"\end{figure}", ""]
    (out_dir / "figures.tex").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="geomob.figures",
        description="Build manuscript figures from experiment results JSON.")
    p.add_argument("inputs", nargs="+",
                   help="results .json files, or a directory of them")
    p.add_argument("-o", "--out-dir", default="figures")
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--font", default="serif", choices=["serif", "sans-serif"])
    p.add_argument("--base-font-size", type=float, default=8.0)
    p.add_argument("--no-tex", action="store_true",
                   help="skip writing the figures.tex snippet")
    args = p.parse_args(argv)

    use_paper_style(font=args.font, base=args.base_font_size)
    out = Path(args.out_dir)

    files = []
    for i in args.inputs:
        q = Path(i)
        files += sorted(q.glob("*.json")) if q.is_dir() else [q]

    # Build everything first, then resolve name clashes, so that a batch
    # containing two runs of the same experiment writes both rather than
    # silently overwriting one with the other.
    pending = []
    for f in files:
        try:
            figs = build_from(f)
        except Exception as e:  # a malformed file should not kill the batch
            print(f"  ! {f.name}: {type(e).__name__}: {e}")
            continue
        if not figs:
            print(f"  - {f.name}: no figure builder matched")
            continue
        pending += [(f, name, fig) for name, fig in figs]

    # cross-file comparison: per-dataset panels and critical-difference
    # diagrams need every results file of the batch at once
    try:
        from . import builders_compare as C
        parsed = []
        for f in files:
            try:
                parsed.append(json.loads(Path(f).read_text()))
            except Exception:
                pass
        cfigs, ctabs = C.build(parsed, out)
        pending += [(Path("comparison"), name, fig) for name, fig in cfigs]
        TABLES.extend(t for t in ctabs if t)
    except Exception as e:  # never lose the per-file figures to this
        import traceback
        traceback.print_exc()
        print(f"  ! comparison figures: {type(e).__name__}: {e}")

    seen: dict[str, int] = {}
    for _, name, _ in pending:
        seen[name] = seen.get(name, 0) + 1

    written, used = [], set()
    for f, name, fig in pending:
        final = name
        if seen[name] > 1:                       # disambiguate by source file
            final = f"{name}__{f.stem}"
            while final in used:
                final += "_x"
            print(f"    (name clash on '{name}' -> '{final}')")
        used.add(final)
        paths = save(fig, out, final, formats=tuple(args.format))
        written.append(final)
        print(f"  + {f.name} -> {', '.join(p.name for p in paths)}")

    if TABLES:
        (out / "tables.tex").write_text(
            "% Generated by geomob.figures -- requires \\usepackage{booktabs}.\n\n"
            + "\n".join(t for t in TABLES if t))
        print(f"wrote {sum(1 for t in TABLES if t)} table group(s) to {out}/tables.tex")
    if written and not args.no_tex:
        write_tex(written, out)
        print(f"\nwrote {len(written)} figures to {out}/ "
              f"and a LaTeX snippet in {out}/figures.tex")
    return written


if __name__ == "__main__":
    main()
