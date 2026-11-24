#!/usr/bin/env bash
# Every experiment of the paper, after one hyperparameter search for GEO.
#
#   bash run_experiments.sh                 # the full run
#   QUICK=1 bash run_experiments.sh         # tiny sizes: checks the whole pipeline first
#   N_EXP=7 bash run_experiments.sh         # results7/, figures/IMG7/
#
# Relaunching skips every step whose results file already exists, so an
# interrupted run continues where it stopped (delete a file to redo a step).
# A failed step is reported at the end; the other steps still run.
#
#  step                     research question it answers
#  1  search                (GEO's settings, scored on validation data only)
#  2  gpe_suite             in domain: on GPE's own benchmark, is GEO at least as good?
#  3  downstream            in domain: do movement and time help real tasks?
#  4  robust, city scope    robustness: does a trained model survive a moved city?
#  5  transfer_suite        transfer: does a model trained on one dataset work on
#                           another (zero-shot, few labels)? + channel ablation
#  6  robust, trip scope    robustness of the channels themselves + channel ablation
#  7  ablation in domain    which channel matters for which task (LSTM only)
#  8  relocate              appendix: the city moved to other latitudes
#  9  anomaly (synthetic)   case study: anomaly types aligned to channels
# 10  figures and tables    for all of the above, plus the search analysis
set -o pipefail

[ -f .venv/bin/activate ] && source .venv/bin/activate

N_EXP=${N_EXP:-6}
CFG=${CFG:-geomob.toml}
QUICK=${QUICK:-0}
# 1: transfer_suite picks GEO's settings itself, from the same [search_space],
#    on in-domain AND zero-shot validation scores (select_on in the config);
# 0: transfer_suite uses the settings of step 1, like every other experiment.
TRANSFER_SEARCH=${TRANSFER_SEARCH:-1}

export DATA_PORTO=${DATA_PORTO:-$HOME/data/Porto/train.csv}     # Kaggle, POLYLINE column
export DATA_TDRIVE=${DATA_TDRIVE:-$HOME/data/TDrive/release/taxi_log_2008_by_id}
export DATA_GEOLIFE=${DATA_GEOLIFE:-$HOME/data/Geolife/Data}

OUT=results$N_EXP
FIG=figures/IMG$N_EXP
if [ "$QUICK" = 1 ]; then OUT=${OUT}_quick; FIG=${FIG}_quick; fi
BEST=$OUT/search/best.toml
mkdir -p "$OUT/ablation" "$OUT/logs" "$FIG"

# datasets that are actually there
DATASETS=()
for d in tdrive porto geolife; do
    v=DATA_${d^^}
    if [ -e "${!v}" ]; then DATASETS+=("$d"); else echo "!! $d not found at ${!v}: skipped"; fi
done
[ ${#DATASETS[@]} -ge 1 ] || { echo "no dataset found"; exit 1; }

# tiny sizes for QUICK=1 (command-line flags beat the config file)
if [ "$QUICK" = 1 ]; then
    Q_SEARCH="--epochs 1 --n-train 300 --n-val 100 --n-test 300 --max-traj 2000"
    Q_GPE="--kinds lstm --n-test 300 --max-traj 2000 --epochs 1 --dims 32 64 --road-n-train 80 --road-n-test 40 --road-epochs 1 --seeds 0"
    Q_DOWN="--backbones lstm --epochs 1 --max-traj 1500 --sim-n-test 200 --seeds 0"
    Q_ROBUST="--epochs 1 --n-test 200 --max-traj 1500 --seeds 0"
    Q_TRANSFER="--epochs 1 --n-test 200 --max-traj 1500 --seeds 0 --fractions 50 1.0 --min-steps 20 --search-epochs 1"
    Q_RELOC="--epochs 1 --n-test 200 --max-traj 1500 --latitudes 0 40 70 --angles 0 45 --shifts-km 0 100 --seeds 0"
    Q_ANOM="--n-train 400 --n-test 200 --severities 1.0"
fi

FAILED=()
step() {   # step <name> <results file> <command...>
    local name=$1 out=$2; shift 2
    if [ -s "$out" ]; then echo "== $name: done already ($out)"; return 0; fi
    echo; echo "==================== $name ===================="
    if ! "$@" 2>&1 | tee "$OUT/logs/$name.log"; then
        FAILED+=("$name (log: $OUT/logs/$name.log)")
    fi
}
geomob() { python -m geomob.cli "$@"; }

python -m geomob.cli selfcheck || exit 1
geomob datasets --config "$CFG"

# 1 ---- search: one GEO configuration, chosen on validation data
step search "$BEST" \
    geomob search --config "$CFG" --out-dir "$OUT/search" $Q_SEARCH
[ -s "$BEST" ] || { echo "the search did not write $BEST: stopping"; exit 1; }

# 2 ---- GPE's own benchmark
step gpe_suite "$OUT/gpe_suite.json" \
    geomob gpe_suite --config "$BEST" --datasets "${DATASETS[@]}" \
        --out "$OUT/gpe_suite.json" $Q_GPE

# 3 ---- downstream tasks, one file per dataset
for d in "${DATASETS[@]}"; do
    step "downstream_$d" "$OUT/downstream_$d.json" \
        geomob downstream --config "$BEST" --dataset "$d" \
            --out "$OUT/downstream_$d.json" $Q_DOWN
done

# 4 ---- robustness: the whole test city moved (+ GEO without re-fitting,
#        GPE in GEO's frame)
step robust_city "$OUT/robust_city.json" \
    geomob robust --config "$BEST" --datasets "${DATASETS[@]}" --scope city \
        --diagnostics --out "$OUT/robust_city.json" $Q_ROBUST

# 5 ---- transfer across datasets (+ from scratch, diagnostics, ablation)
if [ ${#DATASETS[@]} -ge 2 ]; then
    if [ "$TRANSFER_SEARCH" = 1 ]; then
        # the original config: it has the [search_space] (best.toml does not)
        step transfer_suite "$OUT/transfer_suite.json" \
            geomob transfer_suite --config "$CFG" --datasets "${DATASETS[@]}" \
                --search --out "$OUT/transfer_suite.json" $Q_TRANSFER
    else
        step transfer_suite "$OUT/transfer_suite.json" \
            geomob transfer_suite --config "$BEST" --datasets "${DATASETS[@]}" \
                --out "$OUT/transfer_suite.json" $Q_TRANSFER
    fi
else
    echo "!! transfer_suite needs two datasets: skipped"
fi

# 6 ---- robustness of the channels: every trip moved on its own, + ablation
step robust_trip "$OUT/robust_trip.json" \
    geomob robust --config "$BEST" --datasets "${DATASETS[@]}" --scope trip \
        --ablations drop --out "$OUT/robust_trip.json" $Q_ROBUST

# 7 ---- channel ablation in domain (LSTM, GEO with every channel)
for d in "${DATASETS[@]}"; do
    step "ablation_$d" "$OUT/ablation/downstream_$d.json" \
        geomob downstream --config "$BEST" --dataset "$d" --encoders geo_ts \
            --backbones lstm --ablations drop \
            --out "$OUT/ablation/downstream_$d.json" $Q_DOWN
done

# 8 ---- appendix: latitude
step relocate "$OUT/relocate.json" \
    geomob relocate --config "$BEST" --datasets "${DATASETS[@]}" \
        --out "$OUT/relocate.json" $Q_RELOC

# 9 ---- case study: channel-aligned anomaly types (synthetic, needs no data)
step anomaly_synthetic "$OUT/anomaly_synthetic.json" \
    geomob anomaly --config "$BEST" --out "$OUT/anomaly_synthetic.json" $Q_ANOM

# 10 ---- figures and tables (rebuilt every time: cheap)
echo; echo "==================== figures ===================="
python -m geomob.figures "$OUT" -o "$FIG" 2>&1 | tee "$OUT/logs/figures.log" \
    || FAILED+=("figures")
python -m geomob.figures "$OUT/ablation" -o "$FIG/ablation_in_domain" 2>&1 \
    | tee "$OUT/logs/figures_ablation.log" || FAILED+=("figures (ablation)")
if [ -f analyze_search.py ] && [ -s "$OUT/search/search_results.csv" ]; then
    # needs at least 4 trials; with fewer it only explains why in the log
    python analyze_search.py "$OUT/search/search_results.csv" -o "$FIG/search" \
        > "$OUT/logs/search_analysis.log" 2>&1 \
        || echo "(search analysis not made: see $OUT/logs/search_analysis.log)"
fi

echo
echo "results in $OUT/, figures and tables.tex in $FIG/"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED steps (relaunch to retry only these):"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "all steps done"
