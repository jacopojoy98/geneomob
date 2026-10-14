set -e
source ../.geneomob/bin/activate

export N_EXP="robust1"
export DATA_PORTO=~/data/Porto/train.csv        # Kaggle taxi trajectory, POLYLINE column
export DATA_TDRIVE=~/data/TDrive/release/taxi_log_2008_by_id
export DATA_GEOLIFE=~/data/Geolife/Data

OUT=/home/jcolombini/results/$N_EXP
mkdir -p "$OUT"

#python -m geomob.cli robust --config geomob.toml --seeds 0 1 2 --out results/robust.json
python -m geomob.cli robust --config geomob.toml --search --seeds 0 1 2  --out $OUT/robust.json   # tune GEO first
python -m geomob.figures results/robust.json -o $OUT/figures
