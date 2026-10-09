#!/usr/bin/env bash
# Hyperparameter search for GEO, then the full experiments with the winner.
#   1. edit [search] and [search_space] in geomob.toml
#   2. bash run_search.sh
set -e
source ../.geneomob/bin/activate

export N_EXP="Asearch"
export DATA_PORTO=~/data/Porto/train.csv        # Kaggle taxi trajectory, POLYLINE column
export DATA_TDRIVE=~/data/TDrive/release/taxi_log_2008_by_id
export DATA_GEOLIFE=~/data/Geolife/Data

OUT="/home/$USER/results/$N_EXP"
mkdir -p "$OUT"
python -m geomob.cli datasets

# 1) the search: every trial is scored on validation data only.
#    Safe to interrupt and relaunch: finished trials are not repeated.
python -m geomob.cli search --config geomob.toml --out-dir "$OUT/search"

# 2) the full comparison with the best settings found (also builds the figures)
python -m geomob.cli run_all --config "$OUT/search/best.toml" \
       --out-dir "$OUT" --fig-dir "$OUT/IMG"
