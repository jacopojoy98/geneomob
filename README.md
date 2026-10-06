# Equivariant tokenization of mobility trajectories — experiment code

Code for the experiments in *Equivariant Tokenization of Human Mobility
Trajectories via Group Equivariant Operators*, including the comparisons
against **GPE** (Liu et al., *Global Position Embedding for Trajectory
Similarity Computation*, KDD '25).

## 0. Install check and partial data (read this first)

**Always replace the whole `geomob/` folder** when updating; never copy single
files over an older copy. A mixed install produces errors such as
`cannot import name 'SpeedGEO' from 'geomob.encoders'` or
`No module named 'geomob.figures'`. One command finds every such problem:

```bash
python -m geomob.cli selfcheck        # or: python -m geomob.selfcheck
```

**Running with only some datasets.** Nothing needs every corpus:

```bash
# what is found, and where (add --load to actually read each one)
python -m geomob.cli datasets --paths geolife=/data/Geolife roma=/data/roma.txt

# everything, on whatever is available; missing datasets are skipped and named
python -m geomob.cli run_all --paths geolife=/data/Geolife roma=/data/roma.txt
python -m geomob.cli run_all --quick --paths geolife=/data/Geolife   # smoke test
```

* `gpe_suite` and `run_all` load each requested dataset, print
  `!! SKIPPED dataset 'porto': <reason>` for the ones they cannot use, and
  carry on. The results JSON records them under `skipped_datasets`, and table
  captions / figure titles say "Not included (data unavailable): porto, ...",
  so a partial run is never mistaken for the full comparison.
* With a single usable dataset the zero-shot global matrix (GPE Table 4,
  right) is omitted with a note; local metrics, timing, dims, architectures
  and the road-network part still run. If `--road-dataset` is missing, the
  first available dataset is used instead (noted in the JSON).
* `downstream` skips, and records under `skipped_tasks`, any task the corpus
  cannot support (e.g. `mode` without `labels.txt`), and runs the others.
* `run_all` runs: gpe_suite on the available real corpora, downstream on
  GeoLife (skipped if absent), the synthetic channel-aligned anomaly test
  (needs no data), then figures + `tables.tex`. `run_all_status.json` lists
  what ran, what was skipped and why.
* Single-dataset commands (`benchmark`, `anomaly --dataset porto`, ...) stop
  with a one-line reason instead of a traceback when their dataset is absent.

## Config file: every hyperparameter in one place, saved with the results

```bash
python -m geomob.cli config --write geomob.toml     # regenerate the template
python -m geomob.cli run_all   --config geomob.toml
python -m geomob.cli gpe_suite --config geomob.toml --epochs 2   # a flag beats the file
```

`geomob.toml` (shipped next to this README, generated from the CLI itself)
lists every option of every command commented out at its default, plus:

* `[paths]` dataset locations (these beat exported `DATA_<NAME>` variables);
* `[common]` options applied to every command that has them (`device`, `seeds`);
* `[model]` sequence-model architecture (`hidden`, `layers`, `heads`, `dropout`),
  used by every experiment;
* `[encoders]` GPE / GEO / time / speed hyperparameters (`h`, wavelength
  ranges, number of scales, angular harmonics, ...) used by `gpe_suite`,
  `downstream` and `run_all`;
* `[<command>]` that command's options, named like its flags with `-` -> `_`.
  Under `run_all`, the `[gpe_suite]`, `[downstream]` and `[anomaly]` sections
  configure the corresponding steps (they beat `--quick`'s sizes).

Precedence: defaults < `[common]` < `[<command>]` < command-line flags.
Unknown sections, keys or values are rejected with a suggestion
(`unknown key 'epoch' in [gpe_suite]. Did you mean: epochs?`).

**What is saved.** Every results JSON gains a `_run` block: command and argv,
the config file path *and its full text*, every resolved option (including
defaults nobody touched), `[model]` / `[encoders]`, the data paths, and the
versions of geomob, Python, numpy, torch, ... Beside `results/x.json` a
`results/x.config.toml` is written that reproduces the run:

```bash
python -m geomob.cli gpe_suite --config results/gpe_suite.config.toml
```

Under `run_all`, each step's JSON records the options that step actually ran
with, and `run_all_status.json` records the `run_all` invocation itself.

## Results per dataset, and critical-difference diagrams

`run_all` now runs the downstream tasks on **every available dataset**, not
only GeoLife, writing `downstream_<dataset>.json` for each (trajectory-user
linking uses the taxi / vessel / person id; mode detection runs only where
labels exist and is reported as skipped elsewhere). To run one by hand:
`python -m geomob.cli downstream --dataset porto --out results/downstream_porto.json`.

Point the figure builder at the results folder and, besides the per-file
figures, it builds from the whole batch:

| Output | What it shows |
|---|---|
| `fig_by_dataset_<task>` | one panel per dataset: every encoder x sequence model on that task |
| `fig_similarity_by_dataset` | MR / MRR / MP / KP@10 for each dataset separately |
| `fig_cd_similarity`, `fig_cd_downstream`, `fig_cd_downstream_<task>` | critical-difference diagrams |
| `results_by_dataset.csv` | every headline number: dataset, task, model, metric, method, value |
| `cd_stats.json` | average ranks, wins, test, p-value, CD and the list of blocks behind each diagram |
| `tables.tex` | adds one table per task (datasets across) and the table of average ranks |

**Reading a CD diagram.** Methods are ranked inside each *block* (one dataset
x task x sequence model x metric; rank 1 = best, ties share a rank) and placed
at their average rank, best on the left. Two methods joined by a thick bar are
*not* significantly different (Nemenyi, alpha = 0.05); the bar labelled CD
shows the rank gap needed. The title gives the number of blocks and the
overall test (Friedman for three or more methods, Wilcoxon signed-rank for
two). Blocks that share a dataset or differ only in the sequence model are
not independent, so treat the test as indicative; more datasets is what makes
it stronger. A diagram is skipped when fewer than 3 complete blocks exist.

## Hyperparameter search for GEO

```bash
python -m geomob.cli search --config geomob.toml --out-dir results5/search
python -m geomob.cli run_all --config results5/search/best.toml --out-dir results5
# or simply:  bash run_search.sh
```

In `geomob.toml`, list the values to try for any `[encoders]` or `[model]` key:

```toml
[search]
objective = "similarity"   # or "tul", "eta", "mode" (GeoLife downstream tasks)
metric = "KP@10"           # what "best" means; see `search --help` for the list
method = "grid"            # "random" runs n_trials of the combinations
epochs = 3                 # per trial

[search_space]
geo_loc_lam_min_m  = [10.0, 25.0, 50.0]
geo_loc_lam_max_m  = [50000.0, 200000.0]
geo_disp_lam_min_m = [2.0, 5.0, 10.0]
geo_harmonics      = [[1], [1, 4], [1, 2, 4]]
```

* Every trial trains GEO with one combination and scores it on a **validation
  split**. The test data reported by `gpe_suite` / `downstream` is never used,
  so the final numbers are not tuned on their own test set. For similarity,
  each corpus is cut as `[train | n_val | n_test]` with the final `n_test`
  held out (keep `[search] n_test` equal to `[gpe_suite] n_test`); for the
  downstream objectives the task's own split is applied twice.
* The current settings ("base") and GPE are always run as references.
* Impossible combinations (e.g. `lam_min >= lam_max`) are skipped and listed.
* Trials are appended to `trials.jsonl`; relaunching the same search resumes.
* Outputs: `search_results.csv` (all trials, best first, every metric),
  `search_results.json` (plus the run record and a per-value sensitivity
  summary), and `best.toml` = your config with the winning `[encoders]` /
  `[model]`, ready for `run_all --config`.
* **Location vs displacement share.** `geo_loc_dims` says how many of GEO's `h`
  numbers go to location; displacement gets the rest (angular harmonics first,
  the remainder radial), so the total is always exactly `h`. With `h = 128` and
  two harmonics: `64` is the default half/half, `96` gives 96 + 28 radial + 4
  angular, `124` removes the radial part and keeps only the angular one, and
  `128` together with `geo_harmonics = []` is a location-only GEO. Search it
  with `geo_loc_dims = [64, 80, 96, 112, 124]`; each trial's split is printed
  and saved in the `geo_layout` column.
* Leaving `geo_loc_scales` / `geo_disp_scales` at 0 keeps GEO the same size as
  GPE in every trial; the `token_dim` column shows the size if you search them.

## The two comparison runners (start here)

Everything the paper needs is produced by two commands. The rest of this
README documents the individual experiments they build on.

### 1. `gpe_suite` -- every facet of GPE's own evaluation

```bash
export DATA_TDRIVE=... DATA_PORTO=... DATA_ROMA=... DATA_AIS=... DATA_GEOLIFE=...
python -m geomob.cli gpe_suite --out results/gpe_suite.json
python -m geomob.cli figures results/ -o figures/     # figures + tables.tex
```

| GPE table | what is reproduced | `--parts` |
|---|---|---|
| 4 (left), 10-13 | local MR, MRR, MP, KP@10 per dataset and averaged | `local_global` |
| 4 (right), 14-18 | zero-shot MRR, train on each dataset / test on every other | `local_global` |
| 6 (right) | training time = position fitting + model training | `local_global` |
| 6 (left) | local MRR vs embedding size h | `dims` |
| 8, 19, 20 | all of the above with LSTM, Transformer and GNN backbones | `arch` |
| 9 | road-network HR@10 on TP, DITA, LCRS, NetERP | `road` |

Protocol kept identical to GPE Sec. 5.1: the **final 5,000** trajectories of
each corpus are the test set (`--split gpe`, `--n-test 5000`), the odd/even
self-similarity metrics, NCE contrastive training with their positive
recipe, and space-shifting for non-transferable baselines. One training per
(dataset, encoder, backbone) yields a whole row -- local on its own test set,
zero-shot on every other -- so the cross-dataset matrix costs no extra
training. `tables.tex` reproduces Table 4's layout with GPE's own "Improv."
convention (relative for MR, percentage points for the bounded metrics).

GEO works in a local metric frame, so zero-shot results are reported both with
the frame left where training put it and re-anchored on the target city's own
coordinates (no labels, no gradient steps). Table 9 is run the way GPE ran it:
supervised inside an ST2Vec-style model, map-matched vertex sequences, the
Time2Vec temporal module kept for every row so only the spatial vertex
features differ (node2vec, GPE, GEO, or half-and-half concatenations).

**HR@k is tie-aware.** Repeated commutes snap to identical vertex sequences,
and all pairs across disconnected parts of a graph share one capped distance,
so the ground truth is full of exact ties. A plain `argsort` breaks them by
array index -- identically for every model -- and on the fixture that made
five different vertex-feature schemes score the same HR@10 to three decimals.
A retrieval now counts as a hit if it is at least as close as the true k-th
neighbour, and the model's own ties are broken at random with a fixed seed.
With no ties this is exactly the usual HR@k. The share of queries whose
top-k boundary falls in a tie is saved as `gt_tie_fraction` so you can see how
much it matters on your data. GPE's and ST2Vec's papers do not say how they
handle ties, so their HR@10 values may not be exactly reproducible on data
with many repeated routes.

### 2. `downstream` -- the encodings as input to an LSTM or Transformer

```bash
python -m geomob.cli downstream --dataset geolife --path /data/Geolife \
       --backbones lstm transformer --seeds 0 1 2 --out results/downstream.json
```

| task | metrics | baselines in the same table |
|---|---|---|
| trajectory-user linking | acc@1, acc@5, macro P / R / F1 | majority, random |
| travel-time estimation | MAE, RMSE, MAPE (minutes) | constant; least squares on length + departure time |
| mode detection | accuracy, macro P / R / F1, per-class F1 | majority; hand-crafted kinematics + logistic regression |
| anomaly detection | per-type AUROC with SE, registered prediction checks | five one-line statistical detectors |

The encoder set is a 2 x 2 factorial, which is what makes it interpretable:

|  | spatial only | + time + speed channels |
|---|---|---|
| **GPE** | `gpe` | `gpe_ts` |
| **GEO** | `geo` | `geo_ts` |

GPE has no temporal channel, so comparing it against GEO-with-time would
confound the spatial encoding with extra information. `gpe_ts` gives GPE the
*same* time and speed channels: `gpe -> geo` and `gpe_ts -> geo_ts` isolate the
spatial encoding, `gpe -> gpe_ts` measures what the extra channels are worth.
Spatial dimensions are matched at 128. Backbone, head, optimiser, epochs,
clipping and token standardisation are identical across rows.

**Leakage guards** -- each of these would otherwise inflate a result:

* *ETA.* GeoLife samples at a near-fixed interval, so the number of points
  encodes the duration. Paths are resampled at a fixed arc-length step (30 m;
  wider for very long paths so point count stays a function of length only).
  The time channel carries only the departure time, and the speed channel is
  removed -- both would hand the model the answer.
* *TUL.* Split per user in time (earliest trips train, latest test), so every
  user has training data and near-duplicate commutes cannot straddle the split.
* *Mode.* User-disjoint split by default; `--mode-split trip` for the looser
  protocol common in the literature. Report which you used.
* *Anomaly.* A sequence **autoencoder** (LSTM or Transformer encoder, 16-d
  bottleneck, matching decoder) is trained on clean trajectories only; it never
  sees an anomaly or a label, since supervised training on injected anomalies
  would learn the injection procedure. Scores: reconstruction error, and k-NN
  distance of the latent code to clean codes. A third, **no-model** score (k-NN
  on pooled tokens) shows what the LSTM/Transformer adds over the raw encoding.
  The split here is per-user temporal, not user-disjoint -- see below.

**Why the anomaly split is not user-disjoint.** Anomaly detection asks whether
a trip deviates from established behaviour, so normal test trips must come
from the same population as training. Under a user-disjoint split every normal
test trip belongs to an unseen person with unseen homes and workplaces -- it is
already an "unseen area" -- and location-aware encoders are penalised for
knowing where things are. Measured on the GeoLife fixture, that split put every
encoder at chance (AUROC ~0.48); the per-user temporal split restored
0.65-0.80.

**Sample size for the anomaly verdicts.** Each registered prediction is
decided only if its AUROC is more than two standard errors from the threshold.
With ~7 anomalies per type the SE is ~0.11 and most verdicts come out
*inconclusive*; with the default 20,000 GeoLife trajectories the test split
yields ~160 per type and SE ~0.03.

## Install

```bash
pip install -r requirements.txt
```

That is numpy, pandas, scipy and matplotlib (all four needed for the
numpy-only experiments and the figures) plus torch, which only the trained-model
experiments use. If you want to start without torch, `pip install numpy pandas
scipy matplotlib` is enough for `audit`, `tokens`, `probe`, `anomaly`,
`inspect` and `figures`.

Run everything from the directory containing `geomob/`; there is nothing to
build or install beyond the dependencies.

## Quick start

Nothing below needs a dataset — every command defaults to the built-in
synthetic corpus, so you can run the whole suite before touching real data.
Timings are wall-clock on 4 CPU cores.

```bash
# --- tier 1: no torch, seconds, fully deterministic -----------------------
python -m geomob.cli audit    --out results/audit.json    #  ~16 s
python -m geomob.cli tokens   --out results/tokens.json   #  ~57 s
python -m geomob.cli probe    --out results/probe.json    #   ~4 s
python -m geomob.cli anomaly  --out results/anomaly.json  #  ~30 s (3 severities)

# --- tier 2: needs torch, minutes ----------------------------------------
python -m geomob.cli lambda    --out results/lambda.json  #  ~50 s (3 lambdas x 3 epochs)
python -m geomob.cli exp_a     --out results/expa.json
python -m geomob.cli benchmark --out results/bench.json --dataset synthetic_hard
python -m geomob.cli transfer  --out results/transfer.json --src synthetic --dst-lat 0

# --- turn every results file into manuscript figures ---------------------
python -m geomob.cli figures results/ -o figures/        #  ~15 s
```

`figures` matches files to builders **by content, not filename**, so it picks
up whatever is in `results/` and skips what it does not recognise. It writes
PDF + PNG and a `figures.tex` snippet with a caption and label per figure.

Every command takes `--out`, `--seed` and `--dataset`. Start with tier 1: those
four are deterministic, need no GPU, and carry the claims that do not depend on
a model converging.

### Running on real data

Point `--dataset` at a corpus and give it a path (or set the environment
variable in the next section):

```bash
python -m geomob.cli anomaly   --dataset geolife --path /data/Geolife
python -m geomob.cli lambda    --dataset geolife --path /data/Geolife --measure TP
python -m geomob.cli benchmark --dataset porto   --path /data/porto/train.csv --n-test 5000
python -m geomob.cli anomaly   --dataset csv     --path commute.csv
```

Two flags matter on real data. `--split user` (the default wherever a user
column exists) keeps one person's repeated trips out of both halves; use
`--split trip` only on corpora with too few users to split. And `--n-test 5000`
is required for any table meant to sit alongside GPE's published numbers, since
MR and MRR both depend on the candidate-pool size.

### GeoLife specifics

GeoLife has **182 users**, of which **73** ship a `labels.txt` with
transportation-mode annotations. Two things are worth knowing:

* `.plt` files are found **recursively**, so `--path` can point at any level at
  or above `Data/` — the extracted archive root, `Data/` itself, or even a
  single user directory.
* `max_traj` (default 20000) spreads its budget **evenly across users** rather
  than taking the first N trajectories. Taking a prefix would stop partway
  through the first handful of users, and since the labelled users are mostly
  numbered well above those, you would get a corpus with ~11 users and zero
  mode labels — which also silently ruins `--split user`. When the cap binds,
  the loader says so and reports how many trajectories per user it kept.

```bash
python -m geomob.cli probe --dataset geolife --path /data/Geolife
# GeoLife: 20000 trajectories, 182/182 users, 4127 mode-labelled  {...}
#   (max_traj=20000 reached; sampled up to 109 trajectories per user ...)
```

Pass `max_traj=None` for the whole corpus, or `labelled_only=True` to restrict
to the 73 labelled users for the travel-mode experiment.

### If a run is too slow or too small

The trained experiments scale with `--n-train`, `--n-test` and `--epochs`; the
λ sweep additionally scales quadratically in `--n-test`, because the ground
truth is an all-pairs distance matrix, so keep it in the low hundreds. The
anomaly experiment refuses to run on fewer than 50 test trajectories rather
than reporting noise, and the exp_a runner marks any model that never beat a
constant predictor as `warning_failed_to_fit` and greys it out in the figure.

## Real data

Set an environment variable per corpus, or pass `--path`:

```bash
export DATA_PORTO=/data/porto/train.csv        # Kaggle taxi trajectory, POLYLINE column
export DATA_TDRIVE=/data/tdrive/release/taxi_log_2008_by_id
export DATA_GEOLIFE=/data/Geolife            # any level at or above 'Data'

export DATA_ROMA=/data/roma/taxi_february.txt  # CRAWDAD roma/taxi
export DATA_AIS=/data/ais/AIS_2024_01_01.csv   # marinecadastre.gov
```

All six corpora plus your own CSV go through one interface, and three things
are now handled uniformly across them:

* **user ids** on every loader (GeoLife directory, T-drive file, Porto
  `TAXI_ID`, Roma taxi id, AIS MMSI), so `--split user` works everywhere;
* **GeoLife travel-mode labels** from each user's `labels.txt`, canonicalised
  to walk / bike / bus / car / rail / other and attached to the window they
  cover — pass `labelled_only=True` for the mode experiment;
* **local wall-clock time.** Every loader returns hours since the epoch on the
  corpus's *own* clock, never converted to UTC. Calling `datetime.timestamp()`
  on a naive timestamp silently applies the host machine's timezone, which
  would put T-drive's 08:00 Beijing rush hour at 00:00 on a UTC host and make
  hour-of-day features machine-dependent. AIS stays UTC because it spans many
  longitudes, and that is stated rather than hidden.

Roma's WKT lists **latitude first** (`POINT(lat lon)`), so the loader defaults
to `latlon_order=True` and prints the resulting bounding box — Rome is near
lon 12.5, lat 41.9, so a swapped file is obvious at a glance.

For your own CSV use `--dataset csv --path yourfile.csv`; see the section
above.

Preprocessing mirrors GPE's Section 5.1 (gap splitting at 30 min / 1 min / 1 h,
cut at 200 points with 100 overlap, discard below length 20), so numbers land
in the same table as their Tables 10–18. **To be directly comparable to their
MR/MRR/MP/KP figures, use `--n-test 5000`** — mean rank depends on the size of
the candidate pool, and their protocol uses 5,000 held-out trajectories.

## Using your own commute CSV

Built and tested against exactly this header:

```
uid, DAY, HH24, lat, lng, SPEED, HEADING, QUALITY,
ID_PANELSESSION, DELTAPOS, datetime, tid, trip_id, trip_id_segment, user
```

Nothing is hard-coded to it — `infer_schema` guesses, and `CsvSchema` overrides
— but with that header everything works with no configuration:

```bash
# 1. look before you leap: stats + hyperparameters fitted to YOUR data
python -m geomob.cli inspect --path commute.csv

# 2. the experiments, with a user-disjoint split by default
python -m geomob.cli benchmark --dataset csv --path commute.csv --epochs 5
python -m geomob.cli tul       --path commute.csv --split day
python -m geomob.cli transfer  --src csv --src-path commute.csv --dst-lat 0
```

To try the pipeline before touching real data:

```python
from geomob.csvdata import write_example_csv
write_example_csv("fake_commute.csv")     # same header, 12 fake commuters
```

### What `inspect` gives you

Median step, sampling interval, trip lengths, spatial extent, hour-of-day
histogram — and from those, **concrete values for every scale parameter**:
`DisplacementGEO.lam_min/lam_max`, `TorusLattice.lams`,
`PatchworkLattice.tile_m`, `load_csv.gap_min`, and a `GPE.eps_over_2pi` whose
finest wavelength matches your median step. These are physical lengths, so
copying them from a paper about Beijing taxis is a mistake; fit them to your
corpus. On the example corpus (15 s fixes, 57 m median step, 7 × 6 km) it
recommends `lam_min = 14 m`, a torus ladder from 60 m to 14 km, and
`eps/2π = 1.4e-6`.

It also runs the **heading check**. `HEADING` is a free ground truth for the
equivariant angular channel: ψ = 90° − HEADING if it is course over ground in
compass degrees. The check reports the residual binned by step length, because
positional noise randomises the direction of short steps — a large error at
small steps is the noise floor, a large error at *long* steps means the lon/lat
order, the sign convention, or the column's meaning is wrong. Catch that before
it surfaces as a mysterious result in the rotation experiments.

### Three things about commute data that change the protocol

**Split on users, not trips.** A person's Tuesday and Wednesday commutes are
near-duplicates. A random trip-level split puts them on both sides and the
score measures memorisation. `--split user` is the default whenever a user
column is present; `--split trip` is the weaker fallback. For TUL, where every
user must appear in training, `--split day` (train on early weeks, test on
late) plays the same role.

**Windows from one trip stay together.** `load_csv` cuts at 200 points with
100 overlap, following GPE, so overlapping windows of the same trip are
near-duplicates too. The splitters keep them on one side.

**Time finally matters.** DAY and HH24 are exactly the C24 × C7 coordinates the
cyclic-time GENEO is defined on, so this is the first corpus in this repo where
that channel is testable. But heed the leak warning: in the odd/even similarity
protocol both halves carry identical timestamps, so use `geo_spatial` there and
test the time channel on TUL or next-location prediction instead.

### Trajectory-user linking

Commute corpora rarely carry purpose or mode labels, but the user column gives
a well-posed downstream task that self-similarity ranking cannot substitute
for. `geomob.cli tul` runs it across tokenizers and reports accuracy against
the majority-class baseline. Read it as a Hypothesis-4 check that person-level
information survives tokenization — not as evidence for equivariance, since
TUL is *helped* by an absolute code that pins down a home address.

### One early signal worth chasing

On the toy commute corpus with a user-disjoint split (2 epochs, tiny — treat as
a hypothesis, not a result), the ordering **reverses** relative to the in-domain
synthetic benchmark:

| tokenizer | MR | MRR | KP@10 |
|---|---|---|---|
| GPE (h=128) | 10.58 | 0.230 | 0.560 |
| GEO(i)+disp (d=38) | **5.55** | **0.348** | **0.743** |

If this holds on your real data it is the strongest argument in the paper: when
the test users' homes were never seen in training, a fine-grained absolute
position code generalises worse, and the relative channel carries the load.
That is the same mechanism as the cross-city story, at the scale of people
rather than cities. Run it properly before believing it.

## GPE's Section 5.5, run for this tokenizer

```bash
python -m geomob.cli architectures --dataset csv --path commute.csv --transfer-lat 0
python -m geomob.cli roadnetwork   --dataset csv --path commute.csv
python -m geomob.cli roadnetwork   --graph-path city.graphml   # real OSM graph
```

### Architectures (their Tables 8, 19, 20)

Swap the backbone while holding the tokenizer set fixed: LSTM, Transformer,
and a GNN running two GAT layers over each trajectory's path graph. `PathGATConv`
implements that directly — on a path the neighbourhood of vertex *i* is
{*i*−1, *i*, *i*+1}, so the layer is attention over three shifted copies of the
sequence. No torch-geometric dependency, and the batched padded layout is
identical to the other backbones, so the comparison stays like-for-like.

The claim under test is that the tokenizer's contribution is a property of the
**representation**, not an interaction with one sequence model — which is what
makes it a tokenizer rather than an architectural trick. The runner therefore
reports Spearman rank agreement between backbones, so that judgement isn't left
to the eye. On the toy run: lstm↔transformer ρ = 0.60, transformer↔gnn ρ = 0.80,
**lstm↔gnn ρ = 0.00**. At 2 epochs on 800 trips that's noise, but it is exactly
the number that has to come out high before the word "tokenizer" is earned.

`--transfer-lat` adds a second panel: the same test corpus relocated to another
latitude. On the toy run the local panel is flat (every tokenizer 0.43–0.58,
Grid and XY among the best) while the transfer panel separates: GEO 0.25/0.27/0.27
across the three backbones against Grid 0.10/0.20/0.03 and XY 0.14/0.23/0.02.
**The local panel does not discriminate and the transfer panel does** — which is
the same lesson as everywhere else in this repo.

### Road network (their Table 9)

GPE integrate with ST2Vec and ask whether replacing or augmenting node2vec
vertex features with a position embedding helps. `geomob/graph.py` supplies the
four things that needs: a network, a map from GPS to it, vertex features, and
network-aware ground truths.

**On the network.** Pass `--graph-path` for a real OSM graph if you have one
(needs `osmnx`). Otherwise the network is *induced from the trajectories*:
vertices are occupied grid cells, edges join cells consecutive in some trip. For
a commute corpus that is a fair proxy — the road network as actually travelled —
but two things must be said in any write-up: it shares its support with the data
the models see, so it cannot support a claim of independence from the corpus, and
it has no one-way or turn restrictions.

**On the distances.** TP, DITA, LCRS and NetERP are implemented faithfully in
form but over the induced vertex sequence rather than matched road segments, so
they are network-aware analogues, not reimplementations. node2vec is a compact
reimplementation (biased walks plus skip-gram with negative sampling) so the
pipeline runs without gensim.

Concatenated rows split the dimension in half between the two sources, following
GPE, so the comparison is not simply a capacity increase.

**The toy run does not reproduce their headline finding, and that is worth
knowing before you run it on real data.** Position embeddings beat node2vec, as
they report (GPE 0.77 vs node2vec 0.51 on TP). But concatenation *hurt* here —
GPE+node2vec 0.69 against GPE alone 0.77 — where GPE report complementarity.
The likely reason is the induced graph: its topology is derived from the same
trips, so node2vec carries little information the position code does not already
have. A real OSM graph is the test that would settle it. Note also that DITA
inverts the ordering entirely (node2vec best at 0.44, GPE worst at 0.40), which
is a reminder that these four ground truths are genuinely different measures and
should be reported as four columns, never averaged.

## Anomaly detection with a channel-aligned taxonomy

```bash
python -m geomob.cli anomaly --dataset geolife --path /data/Geolife
python -m geomob.cli anomaly --dataset csv --path commute.csv --severities 1.0 0.5 0.25
```

Seven anomaly types, each tied to a channel, with the prediction of *which
channel should catch it* written into `anomaly.EXPECTED` before anything runs.
Detection is unsupervised (k-NN distance in pooled-token space, fit on clean
trajectories only) and identical across representations, so the comparison is
about the representation alone.

| anomaly | preserves exactly | predicted to be missed by |
|---|---|---|
| `detour` | endpoints | — |
| `kinematic` | the entire path | everything without a speed channel |
| `wrong_time` | all geometry | any spatial-only tokenizer |
| `off_grid` (45°) | all rotation invariants | a rotation-**invariant** model |
| `off_grid_control` (90°) | all invariants *and* grid alignment | an orientation-aware model |
| `teleport` | — | — |
| `unseen_area` | shape, duration, every relative quantity | a purely relative model |

Two guards against the usual objection to synthetic anomalies: one-line
statistical detectors (path length, mean speed, duration, bbox area, start
hour) appear in the same table below a rule, and every verdict carries a
Hanley–McNeil standard error, with calls inside two SE marked *inconclusive*
rather than decided.

**Three findings changed the code rather than being written around.**

*A re-timed trajectory was invisible to the whole construction.* φ(Δx) encodes
displacement magnitude and φ_time encodes absolute time-of-day; neither sees
Δt. A one-line mean-speed detector scored 1.000 where the tokenizer scored
0.54. `SpeedGEO` — log-speed and log-Δt on a geometric ladder, invariant under
all of SE(2), so it costs the construction nothing in symmetry — closes it to
1.000. This is a genuine gap in the proposal, not a weak anomaly.

*Grid orientation needs the m=4 angular harmonic.* With m=1 the `off_grid`
signal was entirely positional: the 90° control scored **higher** than the 45°
anomaly, at every grid strength up to a perfectly grid-locked corpus. Mean+std
pooling keeps the net heading and discards heading-mod-90, which is what
alignment *is*. Adding m=4 reverses the sign of the gap. The lesson is sharper
than the fix: equivariant tokens are necessary but not sufficient — the
harmonic order must match the symmetry, and the aggregation must preserve it.

*The 90° control is what makes the 45° row interpretable.* Both rotations
displace absolute positions equally, so a position-only detector scores the
same on each; only a genuinely orientation-aware one scores higher on 45°.
Report the **gap**, never either number alone.

## The λ sweep: whose time embedding, and how much does time matter?

```bash
python -m geomob.cli lambda --dataset geolife --path /data/Geolife --measure TP
python -m geomob.cli lambda --dataset csv --path commute.csv --temporal-mode absolute
```

ST2Vec's learning target is $\mathcal{D} = \lambda\mathcal{D}_S + (1-\lambda)\mathcal{D}_T$
with λ fixed at 0.5. Sweeping it turns a number into a curve, and the curve is
the experiment. Training here is **supervised** — ST2Vec's triplet loss (their
Eq. 12) regressing $e^{-\|v_a-v_b\|}$ onto $e^{-\alpha\mathcal{D}}$ — because
that is what HR@10 grades. Self-supervised contrastive training answers a
different question and is not offered.

Four time channels compete: `none` (the GPE position — it encodes (lon, lat)
and nothing else, so it cannot participate at the temporal end at all), `raw`,
`time2vec` (ST2Vec's Eq. 3 = Time2Vec, learnable ω and φ plus a linear term),
and `cyclic` (the C24 × C7 GENEO).

**Two temporal ground truths, and the choice matters more than it looks.**
`--temporal-mode absolute` is ST2Vec's $|t_i - t_j|$, under which two trips a
week apart are maximally dissimilar however alike their schedules — a raw
timestamp is then near-sufficient and a periodic code is structurally
handicapped. `--temporal-mode cyclic` (the default here) uses circular
time-of-day plus day-of-week, which is the notion ST2Vec's own Figure 1
motivates: a rideshare match cares that two people leave at 08:00, not that
they leave in the same calendar week. Report both — each code should win on the
ground truth whose structure it matches, and that *is* the thesis.

On the GeoLife fixture, cyclic ground truth, HR@10:

| λ | none | raw | time2vec | **cyclic** |
|---|---|---|---|---|
| 1.0 (spatial) | 0.597 | 0.591 | 0.178 | 0.410 |
| 0.5 | 0.305 | 0.338 | 0.167 | **0.607** |
| 0.0 (temporal) | 0.191 | 0.214 | 0.197 | **0.567** |

**The shift test is the sharp one.** `--shift-test-h 168` shifts every
timestamp by one week — *a period of C24 × C7*. An exactly periodic code is
unchanged to floating point; a code with a non-periodic linear term drifts. On
the synthetic corpus the cyclic channel showed **0.000** relative embedding
drift at every λ while Time2Vec drifted 0.53 → 1.11 as the target became more
temporal. Shifting by a non-period (5 hours, say) tests nothing, since every
honest time code should react to a real change of hour.

## Figures

Every experiment writes JSON with `--out`; `geomob.figures` turns that JSON
into manuscript-ready figures.

```bash
python -m geomob.cli audit     --out results/audit.json
python -m geomob.cli tokens    --out results/tokens.json
python -m geomob.cli probe     --out results/probe.json
python -m geomob.cli exp_a     --out results/expa.json
python -m geomob.cli benchmark --out results/bench.json
python -m geomob.cli transfer  --out results/transfer.json --src ... --dst ...
python -m geomob.cli tul       --out results/tul.json --path commute.csv

python -m geomob.cli figures results/ -o figures/
```

Each file is matched to a builder **by content, not filename**, so renamed or
merged result files still work, and a partial file (interrupted run, subset of
tokenizers) produces the figures it can rather than failing.

Figure names carry a discriminator taken from the results themselves, so two
runs of the same experiment do not overwrite each other:
`fig_experiment_a_destination` and `fig_experiment_a_eta`,
`fig_roadnetwork_induced` and `fig_roadnetwork_osm`,
`fig_architectures_<dataset>`. If two files would still produce the same name
the driver appends the source stem and says so. Captions are matched by longest
prefix, so a tagged name still finds its caption, and the task-specific ones
state the relevant $\rho$ ($R_\theta$ for destination, $I$ for ETA). Output is PDF for
`\includegraphics` plus PNG for quick viewing, at 400 dpi with Type-42
embedded fonts, and a `figures.tex` snippet with a ready figure environment,
caption and label for each one.

| figure | from | shows |
|---|---|---|
| `fig_equivariance_error` | audit | translation and rotation error per encoder, log scale |
| `fig_latitude` | audit | Gram distortion vs latitude, with and without re-fitting |
| `fig_noise` | audit | code stability vs GPS jitter |
| `fig_experiment_a_*` | expa | task error vs training size; prediction and attribution equivariance |
| `fig_token_identity` | tokens | token preservation, lattice-period vs arbitrary `v` |
| `fig_gauge_defect` | tokens | measured defect vs the O(‖v‖²) prediction |
| `fig_probe` | probe | heat map of probe R² by representation and statistic |
| `fig_<name>` | bench_* | MRR / MP / KP and mean rank per tokenizer |
| `fig_<name>` | transfer_* | zero-shot vs re-fit vs fine-tune |
| `fig_architectures_*` | arch | MRR per tokenizer per backbone, local and transfer |
| `fig_roadnetwork_*` | roadnet | HR@10 per vertex-feature set per ground truth |
| `fig_anomaly` | anomaly | AUROC heat map by type, with the trivial-detector guard |
| `fig_anomaly_orientation` | anomaly | the 45° vs 90° orientation control |
| `fig_lambda` | lambda | HR@k vs λ per time channel, plus period-shift drift |
| `fig_tul` | tul | user-linking accuracy vs the majority baseline |

Style lives in one place, `geomob/figures/style.py`: single- and double-column
widths, the Okabe-Ito palette (distinguishable under the common forms of colour
blindness and in greyscale), and a stable method-to-colour map so "orange is
GPE" holds across every figure in the paper. Nothing relies on colour alone —
series also differ by marker and line style, bar groups by hatch. Change
`--font sans-serif` or `--base-font-size` to match your template.

## What maps to what

| Command | Proposal | Hypotheses | Needs torch |
|---|---|---|---|
| `audit` | new: translation/rotation/latitude/noise audits | 1, 6, 7 | no |
| `tokens` | Experiment C (corrected) | 1, 4 | no |
| `probe` | Experiment F | 4 | no |
| `exp_a` | Experiment A | 1, 2 | yes |
| `benchmark` | Experiment B | 2, 3 | yes |
| `transfer` | Experiment G | 5 | yes |
| `inspect` | — (setup for your CSV) | — | no |
| `tul` | new: trajectory-user linking | 4 | yes |
| `architectures` | GPE Sec. 5.5 (Tables 8/19/20) | 3 | yes |
| `roadnetwork` | GPE Sec. 5.5 (Table 9) | 3, 4 | yes |
| `gpe_suite` | GPE Tables 4, 6, 8, 9 in one run | 2, 3, 5 | yes |
| `downstream` | TUL, ETA, mode, anomaly x LSTM/Transformer | 2, 3, 4 | yes |
| `anomaly` | channel-aligned anomaly detection, no model (k-NN on tokens) | 1, 4 | no |
| `lambda` | new: ST2Vec λ sweep, time-channel head-to-head | 1, 3, 4 | yes |
| `figures` | — (plots any results JSON) | — | no (needs matplotlib) |

Experiment D (relative vs. absolute positional encoding) has its building
block in `models.RelativeGeoAttention` but no driver script yet.

## Positioning against GPE, in one table

`audit` produces this directly. Synthetic corpus centred at 41.15°N:

| | translation | rotation | latitude (Gram distortion, 41°→0°) | finest λ |
|---|---|---|---|---|
| GPE (h=128) | **1e-11**, analytic ρ | 0.87 | 0.041 | 38 m |
| GPE (h=256) | **5e-11**, analytic ρ | 0.94 | 0.033 | 6.9 m |
| Space2Vec | **1e-11**, analytic ρ | 1.14 | 0.126 → **0.0** refit | — |
| Torus lattice (i) | **4e-12**, analytic ρ | 1.06 | 0.135 → **0.0** refit | — |
| DisplacementGEO | invariant | **2e-15** | **0.0** | — |

Three things this settles:

1. **GPE is already an exactly translation-equivariant GEO.** Its "equal
   distance" and "equal similarity" theorems are corollaries of ρ being
   orthogonal. It should be framed as a special case this framework explains,
   not as a competitor.
2. **Rotation is where the displacement channel is new** — but note raw XY
   also scores well on rotation, since rotation is itself orthogonal. The
   claim to make is symmetry *together with* boundedness and multi-scale
   resolution, not symmetry alone.
3. **Latitude is where the separation of channels pays.** Metric codes go to
   exactly zero distortion once their local frame is re-fitted; GPE has
   nothing to re-fit, because its ladder is fixed in angular units worldwide.
   GPE's own corpora sit at 40–42°N (two of them are the same city), so their
   cross-city experiments never vary cos(lat) by more than ~3%.

## Corrections to the proposal, implemented here

**Experiment C's prediction was wrong as written.** Under translation by `v` an
equivariant code moves by ρ(v); it does not stay put. So "regime (i)/(iii)
should score at or very near 100%" is false for arbitrary `v`. Two repairs,
both falsifiable, both implemented:

- `test_1_lattice_period`: restrict `v` to the lattice. Regime (i) scores
  **exactly 1.000**; GPE and naive k-means score 0.003–0.03.
- `test_2_equivariant_codebook`: use a codebook ρ(v) permutes. The prediction
  becomes "the index changes by the predicted permutation".

### The equivariant codebook

`EquivariantTorusCodebook` is a usable discrete tokenizer, not just a
diagnostic: uniform bins on the torus phase, `encode` returning integer tokens
for a transformer over discrete inputs. Translation shifts the phase by a
constant, so it **permutes** the cells rather than scrambling them — ρ descends
from the continuous code to the discrete tokens. A k-means codebook fitted on
the same equivariant code does not have this property: the vectors move
correctly but the cell boundaries are arbitrary, so the induced index map is
not a group action.

Three results, all analytic and all hit exactly:

| ‖v‖ | on lattice | predicted | measured |
|---|---|---|---|
| 31.25 m | no | 0.500 | **0.501** |
| 46.88 m | no | 0.750 | **0.752** |
| 62.5 m | yes | 1.000 | **1.000** |
| 93.75 m | no | 0.500 | **0.499** |
| ≥ 125 m | yes | 1.000 | **1.000** |

The prediction is a *number to hit*, not a bound to stay under. On-lattice
translations (multiples of the cell size) are exact by construction. Off-lattice
ones must match $\prod_a (1-|r_a|)$, where $r_a$ is the rounding residual in
cells — the fraction of points sitting within the residual of a boundary. Both
regimes match to three decimals, which is a far stronger claim than "tokens are
mostly preserved".

`permutation_transfer_test` separates a group action from a coincidence: fit the
index map empirically on half the points, apply it to the other half. The
equivariant codebook scores 1.000 fitted and 1.000 held out, and the
empirically fitted map turns out to **be** the analytic ρ(v) (100% agreement).
k-means on the same code scores 0.55 held out; k-means on GPE, 0.83.

Read held-out agreement, not the generalisation gap: for a coarse codebook and
a small `v` the best empirical map is near the identity, which generalises fine
while agreeing only partially. A near-zero gap there means the map is trivial,
not that the codebook is equivariant. Only the equivariant codebook admits an
analytic permutation at all.

Regime (iii)'s period is the *tile-local* λ_k, so a single global `v` is the
wrong test for it. `test_1c` uses per-tile periods: **0.981 mean, 0.949 min**,
with the shortfall entirely at tile boundaries — the measured cost of an atlas.

**Regime (ii)'s gauge defect really is O(‖v‖²)**: measured tail slope **2.002**.
Getting there required deriving the gauge J from the same interpolant `encode()`
uses for η. Mixing the interpolant with the underlying field leaves an O(‖v‖)
discretisation residual that masquerades as a failure of Eq. (5).

**A single λ in regime (i) aliases every λ**, so the code is not unique and the
draft's own uniqueness requirement fails. `TorusLattice` therefore defaults to a
geometric ladder whose longest period exceeds the domain diameter. Still exactly
equivariant — a direct sum of equivariant blocks is equivariant — and it is
exactly the trick GPE uses, with a fundamental period of one globe.

**u(ψ) is discontinuous as r → 0**, and stay points produce many near-zero
displacements. `DisplacementGEO(gate=True)` multiplies the angular channel by a
smooth invariant g(r) with g(0) = 0: continuity restored, SO(2)-equivariance
still exact, because g(r) is invariant. GPE lists continuity as a core property
and reviewers will check yours.

**The composite symmetry of Sec. 2.6 is overstated.** A square-lattice φ_loc is
equivariant under rotations by multiples of 90° only, so the full token is
equivariant under ℝ² ⋊ C4, not all of SE(2). Only the relative channel carries
full SE(2). GPE's axis-aligned encoding has the same limitation, unacknowledged.

## The result that should shape the paper

On plain **in-domain similarity ranking**, GPE wins. On the hard synthetic
preset, at matched dimension (126 vs 128) and matched backbone:

| tokenizer | MR | MRR | MP | KP@10 |
|---|---|---|---|---|
| GPE (h=128) | **3.48** | **0.551** | **0.352** | **0.718** |
| GEO(i)+disp [d=128] | 4.19 | 0.487 | 0.293 | 0.682 |

Take this seriously rather than tuning around it. A fine-grained absolute
position ladder is very well suited to ranking two halves of the same trip,
and the invariant/equivariant split deliberately throws away some of exactly
that. The case for this tokenizer is **not** in-domain similarity; it is
transfer, latitude robustness, noise stability, and equivariant attribution.
Frame the paper on those axes and report the similarity numbers honestly, or
reviewers who know GPE will find this themselves.

Two caveats before over-reading it: these are 2–3 epochs on synthetic data,
and the synthetic generator is not a proxy for real corpora. Re-run on Porto
and T-drive before quoting anything.

### Two methodological traps caught here

**The time channel leaks.** In the odd/even self-similarity protocol both
halves carry the same timestamps, so an absolute-time channel gives MRR =
1.000 — measuring the leak, not the tokenizer. `benchmark` warns if you
include it, and defaults to `geo_spatial`. Test the time channel on a
downstream task instead.

**Dimension must be matched.** GEO(i)+disp defaults to 38 dimensions against
GPE's 128. Reading that gap as evidence about symmetry confounds capacity with
structure. Use `geo_spatial128`.

**The synthetic default saturates.** With well-separated hubs every tokenizer
scores MRR ≈ 1.0 and nothing is discriminable; use `--dataset synthetic_hard`.

## Results from the smoke runs (synthetic)

### Experiment A

Now a **real downstream task with the same baselines used everywhere else**.
Backbone, head, optimiser, gradient clipping and target scaling are identical
across rows; only the tokenizer changes. A separate row adds the equivariant
aggregator on top of the GEO tokens, so the contribution of the tokenizer and
of the equivariant head can be read apart.

```bash
python -m geomob.cli exp_a --dataset csv --path commute.csv --task destination
python -m geomob.cli exp_a --dataset csv --path commute.csv --task eta
```

**Two targets with different symmetry types**, which makes the equivariance
measurement much sharper than a single task can:

| task | target | required behaviour under rotation | $\rho$ |
|---|---|---|---|
| `destination` | remaining displacement from the last observed point | the answer must rotate with the trip | $R_\theta$ |
| `eta` | remaining travel time | the answer must not change | $I$ |

Equivariance and invariance are the same property with different $\rho$, and a
construction tested on only one of them is under-tested. Run both: the
destination task is naturally a displacement and so favours the relative
channel, while ETA depends on absolute position through traffic and geography
and is where the absolute codes should do well. Winning both is the claim
worth making; winning only the favourable one is not.

Two fairness guards, both learned the hard way here. Targets are standardised
to a common scale fitted on the training set — without it every linear-head
model parks on the constant predictor and the aggregator appears to win
because its output is a combination of unit vectors and starts on the right
scale, which has nothing to do with symmetry. And gradients are clipped
identically for all rows, because a diverged baseline is evidence that the
baseline was not tuned, not evidence about symmetry.

The runner **refuses to let a failed row pass as a comparison**: any model that
never beats the constant predictor is listed in `warning_failed_to_fit`,
printed loudly, and greyed out in the figure. On the 336-trip toy corpus GPE
and Grid both trip this — a few hundred trips is not enough for a 128-dim
tokenizer through a shared backbone. Use a real corpus before reporting these
rows.

(The older synthetic two-model version, destination-displacement regression,
8 epochs:)

| | n=200 | n=1000 | n=4000 | equivariance err | explanation equivariance err |
|---|---|---|---|---|---|
| GEO + equivariant aggregator | **0.0525** | **0.0465** | **0.0250** | **6.2e-08** | **3.9e-05** |
| matched-capacity GRU baseline | 0.0599 | 0.0506 | 0.0454 | 0.388 | 0.043 |

The equivariance figures reproduce the draft's reported ~1e-6 vs 0.18–0.51.
The explanation-equivariance column is new: a ~1000× gap, and the one axis no
non-GEO competitor can match by construction.

Experiment F probes (held-out R²) show the designed losses landing exactly
where predicted:

| representation | total dist. | centroid dist. | heading (cos) | r. of gyration |
|---|---|---|---|---|
| GEO(i)+disp | **1.00** | **0.98** | **0.90** | **0.99** |
| GEO(i) only | −0.09 | 0.98 | 0.21 | 0.99 |
| disp only | **1.00** | 0.15 | **0.90** | 0.76 |
| GPE | −0.11 | 0.98 | 0.31 | 0.99 |
| raw xy | −0.01 | 0.22 | 0.10 | 0.91 |

Absolute-position codes (GPE, torus-only) cannot recover path length or
heading; the displacement channel cannot recover absolute position. Each loses
exactly what it was designed to lose, and the concatenation loses neither.

Noise saturation (cosine similarity at σ = 15 m GPS jitter): GPE(h=256) falls
to 0.87, GPE(h=128) to 0.94, torus lattice holds 0.98. GPE's own ε ablation
found finer settings worse; this is the mechanism.

## Layout

```
geomob/
  geo.py           ENU frames, action of G = SE(2) × Θ, latitude reprojection
  encoders.py      GPE (Algorithm 1), Space2Vec, φ_loc regimes (i)/(ii)/(iii),
                   DisplacementGEO, CyclicTimeGENEO, XY/Grid/TriW baselines
  equivariance.py  encoder / model / explanation equivariance, Lipschitz, noise
  codebook.py      k-means and torus-phase codebooks, corrected Experiment C
  data.py          synthetic generator, Porto/T-drive/GeoLife/AIS, GPE's SS protocol
  graph.py         induced or OSM road network, node2vec, map snapping,
                   TP/DITA/LCRS/NetERP network distances, HR@k
  csvdata.py       schema-driven CSV loader, corpus inspector, heading check,
                   user-disjoint splitting, example-corpus generator
  anomaly.py       seven channel-aligned injectors, the occupancy model, the
                   trivial-detector guard, and the predictions in EXPECTED
  figures/         style.py (palette, sizes), builders.py (one per result
                   file), __main__.py (auto-detect driver + LaTeX snippet)
  metrics.py       MR/MRR/MP/KP@10, mobility statistics, ridge probes
  models.py        GPE-matched backbone, EquivariantAggregator, relative-geo attention
  experiments/     exp_a_task (A, real task + shared baselines), exp_audit,
                   exp_similarity (B+G), exp_tokens_probe (C+F), exp_tul,
                   exp_section55 (architectures + road network)
  cli.py
```

Every encoder exposes an analytic `rho(v)` where one exists. Where none does,
`equivariance.py` fits the **best possible** orthogonal ρ by Procrustes on a
held-out split and reports the residual — baselines get the benefit of the
doubt and still show one.
