# Reverse drug-combination dose prediction

This project implements the [model architecture](docs/architecture.md): an
independently trained, frozen cell-metadata VAE encoder, shared drug encoding,
two-coordinate tensor factorization, a shared deep branch, and a correlated
bivariate mixture density network. Given a cell, Drug A, Drug B and target
inhibition, it returns a distribution over the two drug doses, candidate
coordinates, reproducible samples and an optional probability map.

The [protocol](docs/protocol.md) defines source processing, feature construction,
training objectives, splits and metrics. The [design decisions](docs/design.md)
explain the configuration. Predicted coordinates are research outputs;
coordinate accuracy alone does not establish a requested biological response.
The [assessment protocol](docs/evaluation_assessment.md),
[evaluation results](docs/evaluation_results.md) and
[source validation](docs/source_validation.md) describe evaluation and provenance.

## Install and inspect

Run commands from this repository root. Python 3.11–3.14 is supported; the reference
environment uses Python 3.12 and CPU. Install the dependencies recorded in `uv.lock`:

```bash
uv sync --extra dev --locked
uv run reverse-dose --help
uv run reverse-dose inspect --config configs/astra_zeneca.yaml
uv run pytest -q
```

The [dataset description](dataset/dataset.md) and
generated `dataset/manifests/AstraZeneca–DREAM/audit_report.json` audit describe the downloaded source files.
`inspect` records the audit, validated cell mapping, effective configuration and
runtime in `dataset/manifests/AstraZeneca–DREAM/inspection.json`. To rerun the independent source
audit, use `uv run python audit_dataset.py`.

Dataset paths use `<dataset_root>/<artifact type>/<dataset_name>/`. All supplied
configurations select `dataset_root: dataset` and `dataset_name: AstraZeneca–DREAM`
(with an en dash). See the [directory layout](dataset/dataset.md) for adding
separate dataset locations and for notes on existing artifacts after migration.

The default pipeline uses the corrected DREAM matrices and CMP RNA-seq. It never
rewrites `dataset/raw/AstraZeneca–DREAM/`. RNA-seq extraction streams the full archive and retains
every model catalogued with Broad or Sanger cell-line RNA-seq (1,643 candidates
in the current catalog); it does not extract the 6.1 GB CSV to disk. Preparation
first fixes the final response split, groups related CMP identities, removes all
validation/test-only identity groups from the complete offline VAE population,
then splits every remaining profile between VAE training and internal validation.
Only VAE-training profiles fit gene selection, imputation and scaling.

Before those splits, source preprocessing computes inhibition from the released
percent-control response and keeps only the inclusive range `[-2.0, 2.0]`.
Outliers are removed, never clipped, and written to `exclusions.parquet` with
reason `response_outside_quality_range`, their original response, computed
inhibition, source matrix and grid coordinates. The fixed policy is recorded in
and hashed by the source schema.

The supplied chemistry includes 42 numeric PubChem identifiers requiring exact
CID resolution. With `sources.resolve_pubchem: true`, the first preparation
retrieves structures for those released identifiers and preserves the response
bytes, requested/returned CID, URL and hashes under
`dataset/processed/AstraZeneca–DREAM/drug_features/pubchem/v1/`. Subsequent runs verify and reuse
that cache offline. Keep it with the experiment. Setting
`sources.resolve_pubchem: false` uses only locally released parseable SMILES and
explicitly produces a smaller chemistry-supported cohort. No compound-name
searches or surrogate structures are used. An unsuccessful request fails rather
than silently reducing the selected cohort.

## Run the workflow

Perform the short real-source integration check:

```bash
uv run reverse-dose run --config configs/astra_zeneca_smoke.yaml --run-dir runs/integration
```

The smoke configuration uses 240 systematically selected matrices, small feature
and network dimensions, and two epochs per stage. Its metrics check execution and
artifact generation; they are not scientific performance estimates.

Train the default model with:

```bash
uv run reverse-dose run --config configs/astra_zeneca.yaml --run-dir runs/default
```

`run` prepares the source tables and expression, constructs grouped splits and
fitted transforms, trains the offline VAE, freezes its selected encoder, trains
the final model, and evaluates its selected checkpoint on validation and test
data. Use `--skip-test` to defer test evaluation until a selected experiment is ready for test assessment.
The full run may be long; progress, validation scores, learning rates, gradient
norms, MDN covariance diagnostics and checkpoints are visible in the terminal
and persistent structured event logs. The full configuration regularizes only
the final stage with L2 weight decay `1e-3`; this setting limits memorization and
covariance collapse on cell-disjoint splits.

Resume a run created with the current schema without changing its settings:

```bash
uv run reverse-dose run --config configs/astra_zeneca_smoke.yaml --run-dir runs/integration --resume
uv run reverse-dose run --config configs/astra_zeneca.yaml --run-dir runs/default --resume
```

SIGINT/SIGTERM requests checkpoint creation after the active minibatch completes.
Checkpoints also save periodically and at every completed epoch. A hard process
kill resumes from the most recent completed checkpoint. Existing runs require
`--resume`; a changed dataset, configuration, encoder, code or runtime is rejected
as a different experiment. Start a new output directory for changed experiments.

Check each run's saved configuration, dataset fingerprint and source identity
before reuse. Use a separate output directory whenever experiment settings change.

## Execute stages separately

Every stage has an executable entry point. For explicit artifact paths:

```bash
uv run reverse-dose preprocess --config configs/astra_zeneca.yaml \
  --source-dir dataset/processed/AstraZeneca–DREAM/dose_response/manual
uv run reverse-dose extract-expression --config configs/astra_zeneca.yaml
uv run reverse-dose prepare --config configs/astra_zeneca.yaml \
  --source-dir dataset/processed/AstraZeneca–DREAM/dose_response/manual \
  --data-dir dataset/splits/AstraZeneca–DREAM/manual
uv run reverse-dose train-vae --config configs/astra_zeneca.yaml \
  --data-dir dataset/splits/AstraZeneca–DREAM/manual --run-dir runs/stages
uv run reverse-dose train --config configs/astra_zeneca.yaml \
  --data-dir dataset/splits/AstraZeneca–DREAM/manual --run-dir runs/stages \
  --vae-checkpoint runs/stages/vae/best.pt
```

`prepare` verifies and reuses completed source/expression artifacts. It refuses
changed or incomplete artifacts in an occupied destination. The default source
and prepared directory names contain configuration-derived identifiers; the
resolved prepared path for a pipeline run is recorded in `runs/default/run.json`.

For an individual stage, resume with an explicit checkpoint:

```bash
uv run reverse-dose train --config configs/astra_zeneca.yaml \
  --data-dir dataset/splits/AstraZeneca–DREAM/manual --run-dir runs/stages \
  --vae-checkpoint runs/stages/vae/best.pt \
  --resume runs/stages/model/last.pt
```

`last.pt` and `epoch_XXXX.pt` preserve training continuity. `best.pt` contains
selected weights and is for encoder transfer, evaluation and inference; it cannot
be used as a resumption checkpoint. `--stop-after-steps N` on either training
command provides a controlled interruption for verifying restart behavior.

## Evaluate, export and predict

Use the prepared path recorded by the completed pipeline:

```bash
PREPARED_DATA=$(uv run python -c 'import json; print(json.load(open("runs/default/run.json"))["data_dir"])')
uv run reverse-dose evaluate --config configs/astra_zeneca.yaml \
  --data-dir "$PREPARED_DATA" --checkpoint runs/default/model/best.pt \
  --split test --output runs/default/evaluation/test_recheck
uv run reverse-dose export --data-dir "$PREPARED_DATA" \
  --checkpoint runs/default/model/best.pt --output runs/default/export
uv run reverse-dose predict --bundle runs/default/export \
  --cell A549 --drug-a AKT_1 --drug-b MAP2K_1 --inhibition 0.5 \
  --samples 1000 --seed 42 --grid-size 100 --output runs/default/prediction_a549
```

Drug and cell names must be canonical identifiers in the prepared artifacts;
`cells.parquet` and the `features.npz` ID arrays list them. Inhibition uses fraction
units: `0.5` means 50%. Doses and optional `--bounds A_MIN A_MAX B_MIN B_MAX` use
micromolar units. By default, grid bounds use training dose ranges only.

The export contains the model and its verified, self-contained preprocessing
bundle, so inference does not require the raw data or an internet connection.
Predictions include all component weights, means, standard deviations and
correlations, dose candidates and their nonnegativity flags. Samples and grid
densities go to `distribution.npz`; Matplotlib produces `probability_map.png`.
Gaussian predictions are not clipped or renormalized to the plotted rectangle.

For a cell absent from training, supply its measured expression:

```bash
uv run reverse-dose predict --bundle runs/default/export \
  --cell my_new_cell --expression my_new_cell_tpm.csv \
  --drug-a AKT_1 --drug-b MAP2K_1 --inhibition 0.5 \
  --samples 1000 --grid-size 100 --output runs/default/prediction_new_cell
```

The CSV columns are `gene_id,rsem_tpm`, using CMP `gene_id` identifiers and raw
nonnegative TPM, before log transformation or standardization. Include every
selected gene identifier from `expression_transform.npz`; additional genes and
arbitrary row order are supported. Duplicate gene measurements are averaged.
Explicit missing TPM is imputed using the saved training mean. An omitted selected
identifier, negative/infinite TPM or a vector missing all selected measurements
fails clearly. No statistics are refitted for the new cell.

## Experiments and artifacts

Reassess completed, frozen checkpoints without retraining or overwriting their
original evaluations:

```bash
uv run --frozen reverse-dose assess-suite --suite runs/five_split_two_train_seed_run \
  --output runs/suite_assessment
```

Run a frozen Cartesian design with **independent** partition and training seeds:

```bash
uv run --frozen reverse-dose experiments --config configs/astra_zeneca.yaml \
  --output runs/factorial --split-seeds 42 43 44 45 46 \
  --training-seeds 42 43 --protocols cell cell_pair
```

This declares 20 combinations (five partitions × two training seeds × two
protocols). Add `--reference-suite runs/five_split_two_train_seed_run` to reuse
the local completed suite. Matching scientific configurations are reused with
verified provenance and do not count as new training repetitions.
Append `--resume` with the identical command to
continue; the frozen design cannot change in place.

Both commands write separate `cell_results.csv`, `cell_pair_results.csv` and
`summary.json`. Factorial runs additionally retain `design.json` and incremental
`progress.json`. Every evaluation writes an `assessment/` directory with
primary macro-cell, inhibition-stratified, near-target and worst-group metrics;
identity-cluster intervals; density/covariance diagnostics; and response-blind
target choices scored against the original held-out matrices. Raw workbooks are
required for the target assessment; a portable bundle alone is insufficient to
reconstruct missing/QC-excluded grid coordinates safely.

These are retrospective robustness analyses, not new independent biological
replication. OI inclusion is a separate, explicit experiment: copy the configuration and set
`sources.sources: [dream, oi]`. The original raw matrices are never pooled with
the corrected source releases. `cell_pair` groups all replicates of a cell/unordered
drug pair and reports overlap with training cells; its interpretation differs
from the cell-disjoint cold-start baseline.

| Location                                      | Contents                                                                                              |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `dataset/raw/AstraZeneca–DREAM/` | Immutable downloaded sources |
| `dataset/processed/AstraZeneca–DREAM/dose_response/source_*/` | Observations, canonical cell mapping, chemistry features, exclusions and source manifest |
| `dataset/processed/AstraZeneca–DREAM/cell_expression/expression_*/` | Raw TPM means, gene/model IDs, measurement counts and extraction provenance |
| `dataset/processed/AstraZeneca–DREAM/drug_features/pubchem/v1/` | Immutable exact-CID response cache and provenance |
| `dataset/splits/AstraZeneca–DREAM/split_*/` | Split memberships, portable feature arrays, fitted transforms and artifact hashes |
| `dataset/manifests/AstraZeneca–DREAM/logs/` | Standalone preprocessing logs |
| `runs/<run>/` | `run.json`, terminal log, `result.json`, stage directories and evaluations |
| `runs/<run>/{vae,model}/` | Effective experiment metadata, `events.jsonl`, selected/continuation/periodic checkpoints |
| `runs/<run>/evaluation/<split>/` | Metrics JSON, per-observation predictions, mixture parameters, per-cell/pair/matrix/inhibition tables |

Source exclusions remain explicit. The seven agents without released usable
small-molecule structures are `ADAM17`, `CD19`, `PDGFR_Ab`, `TNFSF10`, `ATR_3`,
`PIP5K1` and `SLC16A4`. Unresolved `KMSII` and conflicting `BT474`/`BT474C` labels
are quarantined unless an explicit evidence-bearing alias is configured. The
released `Gemcitibine` synonym is resolved to `Gemcitabine` with its metadata
evidence. Inspect each source manifest and `exclusions.parquet` for the actual
retained population, response-quality exclusions and source-specific issues.

The package contains source ingestion (`preprocess.py`), expression and feature
preparation (`data.py`), architecture (`models.py`), training, evaluation and
inference modules. `assessment.py`, `experiments.py` and `reporting.py` handle
target scoring, experiment designs and summaries. Configuration, artifact
utilities and command-line entry points live in `config.py`, `utils.py` and `cli.py`.
Tests cover source parsing, feature leakage, mathematical densities and branch
symmetry, frozen encoder behavior, serialization, exact CPU interruption/resume,
inference and the complete synthetic workflow.
