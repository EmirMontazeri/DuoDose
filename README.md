# Reverse drug-combination dose prediction — V2

This project implements the architecture in [proposal.md](proposal.md): an
independently trained, frozen cell-metadata VAE encoder, shared drug encoding,
two-coordinate tensor factorization, a shared deep branch, and a correlated
bivariate mixture density network. Given a cell, Drug A, Drug B and target
inhibition, it returns a distribution over the two drug doses, candidate
coordinates, reproducible samples and an optional probability map.

The proposal specifies the architecture but does not define numerical training
settings, feature construction, losses, split ratios or benchmark metrics. The
executable baseline choices are identified in [the protocol](docs/protocol.md)
and [decision register](docs/implementation_plan.md). Predicted coordinates are
research outputs; coordinate accuracy alone does not establish achievement of a
requested biological response.

Historical recorded runs and verification evidence are described in
[validation results](docs/validation_results.md). Those artifacts predate the
expanded, identity-grouped VAE population and remain under `runs/` for audit;
they are not checkpoints of the current preprocessing schema. Raw-data exclusions
and severe out-of-range response/prediction cases remain visible in those results.
The [complete source validation](docs/source_validation.md) records all DREAM
and OI matrix counts, retained populations, label conflicts and missing grids.

## Install and inspect

Run commands from this repository root. Python 3.11–3.14 is supported; the reference
environment uses Python 3.12 and CPU. Install the dependencies recorded in `uv.lock`:

```bash
uv sync --extra dev --locked
uv run reverse-dose --help
uv run reverse-dose inspect --config configs/v2.yaml
uv run pytest -q
```

The existing [dataset description](dataset/dataset.md) and
[audit](dataset/manifests/audit_report.json) describe the downloaded source files.
`inspect` records the audit, validated cell mapping, effective configuration and
runtime in `dataset/manifests/v2_inspection.json`. To rerun the independent source
audit, use `uv run python audit_dataset.py`.

The default pipeline uses the corrected DREAM matrices and CMP RNA-seq. It never
rewrites `dataset/raw/`. RNA-seq extraction streams the full archive and retains
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
and hashed by source schema `source-v2.4`.

The supplied chemistry includes 42 numeric PubChem identifiers requiring exact
CID resolution. With `sources.resolve_pubchem: true`, the first preparation
retrieves structures for those released identifiers and preserves the response
bytes, requested/returned CID, URL and hashes under
`dataset/processed/drug_features/pubchem/v1/`. Subsequent runs verify and reuse
that cache offline. Keep it with the experiment. Setting
`sources.resolve_pubchem: false` uses only locally released parseable SMILES and
explicitly produces a smaller chemistry-supported cohort. No compound-name
searches or surrogate structures are used. An unsuccessful request fails rather
than silently reducing the selected cohort.

## Run the workflow

First perform the short real-source integration check:

```bash
uv run reverse-dose run --config configs/smoke.yaml --run-dir runs/smoke-qc-expanded-vae
```

The smoke configuration uses 240 systematically selected matrices, small feature
and network dimensions, and two epochs per stage. Its metrics check execution and
artifact generation; they are not scientific performance estimates.

Train the complete default baseline with:

```bash
uv run reverse-dose run --config configs/v2.yaml --run-dir runs/v2-qc-expanded-vae-regularized
```

`run` prepares the source tables and expression, constructs grouped splits and
fitted transforms, trains the offline VAE, freezes its selected encoder, trains
the final model, and evaluates its selected checkpoint on validation and test
data. Use `--skip-test` to defer test evaluation while developing an experiment.
The full run may be long; progress, validation scores, learning rates, gradient
norms, MDN covariance diagnostics and checkpoints are visible in the terminal
and persistent structured event logs. The full configuration regularizes only
the final stage with L2 weight decay `1e-3`; this setting was selected without
test access to prevent memorization and covariance collapse on cell-cold splits.

Resume a run created with the current schema without changing its settings:

```bash
uv run reverse-dose run --config configs/smoke.yaml --run-dir runs/smoke-qc-expanded-vae --resume
uv run reverse-dose run --config configs/v2.yaml --run-dir runs/v2-qc-expanded-vae-regularized --resume
```

SIGINT/SIGTERM requests checkpoint creation after the active minibatch completes.
Checkpoints also save periodically and at every completed epoch. A hard process
kill resumes from the most recent completed checkpoint. Existing runs require
`--resume`; a changed dataset, configuration, encoder, code or runtime is rejected
as a different experiment. Start a new output directory for changed experiments.

The included `runs/smoke`, `runs/v2` and `runs/suite` artifacts use the earlier
85-model expression extraction and intentionally fail current continuation
identity checks. Use fresh run directories for the expanded-VAE quality-controlled
pipeline.

## Execute stages separately

Every stage has an executable entry point. For explicit artifact paths:

```bash
uv run reverse-dose preprocess --config configs/v2.yaml \
  --source-dir dataset/processed/dose_response/manual_v2
uv run reverse-dose extract-expression --config configs/v2.yaml
uv run reverse-dose prepare --config configs/v2.yaml \
  --source-dir dataset/processed/dose_response/manual_v2 \
  --data-dir dataset/splits/manual_v2
uv run reverse-dose train-vae --config configs/v2.yaml \
  --data-dir dataset/splits/manual_v2 --run-dir runs/stages_v2
uv run reverse-dose train --config configs/v2.yaml \
  --data-dir dataset/splits/manual_v2 --run-dir runs/stages_v2 \
  --vae-checkpoint runs/stages_v2/vae/best.pt
```

`prepare` verifies and reuses completed source/expression artifacts. It refuses
changed or incomplete artifacts in an occupied destination. The default source
and prepared directory names contain configuration-derived identifiers; the
resolved prepared path for a pipeline run is recorded in `runs/v2/run.json`.

For an individual stage, resume with an explicit checkpoint:

```bash
uv run reverse-dose train --config configs/v2.yaml \
  --data-dir dataset/splits/manual_v2 --run-dir runs/stages_v2 \
  --vae-checkpoint runs/stages_v2/vae/best.pt \
  --resume runs/stages_v2/model/last.pt
```

`last.pt` and `epoch_XXXX.pt` preserve training continuity. `best.pt` contains
selected weights and is for encoder transfer, evaluation and inference; it cannot
be used as a resumption checkpoint. `--stop-after-steps N` on either training
command provides a controlled interruption for verifying restart behavior.

## Evaluate, export and predict

Use the prepared path recorded by the completed pipeline:

```bash
V2_DATA=$(uv run python -c 'import json; print(json.load(open("runs/v2/run.json"))["data_dir"])')
uv run reverse-dose evaluate --config configs/v2.yaml \
  --data-dir "$V2_DATA" --checkpoint runs/v2/model/best.pt \
  --split test --output runs/v2/evaluation/test_recheck
uv run reverse-dose export --data-dir "$V2_DATA" \
  --checkpoint runs/v2/model/best.pt --output runs/v2/export
uv run reverse-dose predict --bundle runs/v2/export \
  --cell A549 --drug-a AKT_1 --drug-b MAP2K_1 --inhibition 0.5 \
  --samples 1000 --seed 42 --grid-size 100 --output runs/v2/prediction_a549
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
uv run reverse-dose predict --bundle runs/v2/export \
  --cell my_new_cell --expression my_new_cell_tpm.csv \
  --drug-a AKT_1 --drug-b MAP2K_1 --inhibition 0.5 \
  --samples 1000 --grid-size 100 --output runs/v2/prediction_new_cell
```

The CSV columns are `gene_id,rsem_tpm`, using CMP `gene_id` identifiers and raw
nonnegative TPM, before log transformation or standardization. Include every
selected gene identifier from `expression_transform.npz`; additional genes and
arbitrary row order are supported. Duplicate gene measurements are averaged.
Explicit missing TPM is imputed using the saved training mean. An omitted selected
identifier, negative/infinite TPM or a vector missing all selected measurements
fails clearly. No statistics are refitted for the new cell.

## Experiments and artifacts

Run a declared multi-seed cold-start and warm-completion comparison with:

```bash
uv run reverse-dose experiments --config configs/v2.yaml --output runs/suite \
  --seeds 42 43 44 --protocols cell cell_pair
```

The suite writes `experiments.csv` and `suite.json`; each constituent run retains
its full artifacts. Append `--resume` to continue an existing suite. OI inclusion
is a separate, explicit experiment: copy the configuration and set
`sources.sources: [dream, oi]`. The original raw matrices are never pooled with
their corrected versions. `cell_pair` groups all replicates of a cell/unordered
drug pair and reports overlap with training cells; its interpretation differs
from the cell-disjoint cold-start baseline.

| Location                                      | Contents                                                                                              |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `dataset/raw/`                                | Immutable downloaded sources                                                                          |
| `dataset/processed/dose_response/v2_*/`       | Observations, canonical cell mapping, chemistry features, exclusions and source manifest              |
| `dataset/processed/cell_expression/v2_*/`     | Raw TPM means, gene/model IDs, measurement counts and extraction provenance                           |
| `dataset/processed/drug_features/pubchem/v1/` | Immutable exact-CID response cache and provenance                                                     |
| `dataset/splits/v2_*/`                        | Split memberships, portable feature arrays, fitted transforms and artifact hashes                     |
| `dataset/manifests/v2_logs/`                  | Standalone preprocessing logs                                                                         |
| `runs/<run>/`                                 | `run.json`, terminal log, `result.json`, stage directories and evaluations                            |
| `runs/<run>/{vae,model}/`                     | Effective experiment metadata, `events.jsonl`, selected/continuation/periodic checkpoints             |
| `runs/<run>/evaluation/<split>/`              | Metrics JSON, per-observation predictions, mixture parameters, per-cell/pair/matrix/inhibition tables |

Source exclusions remain explicit. The seven agents without released usable
small-molecule structures are `ADAM17`, `CD19`, `PDGFR_Ab`, `TNFSF10`, `ATR_3`,
`PIP5K1` and `SLC16A4`. Unresolved `KMSII` and conflicting `BT474`/`BT474C` labels
are quarantined unless an explicit evidence-bearing alias is configured. The
released `Gemcitibine` synonym is resolved to `Gemcitabine` with its metadata
evidence. Inspect each source manifest and `exclusions.parquet` for the actual
retained population, response-quality exclusions and source-specific issues.

The package is organized into `preprocess.py` (source ingestion), `data.py`
(expression/splits/transforms), `models.py` (architecture and densities),
`training.py`, `evaluation.py`, `inference.py`, `config.py`, `utils.py` and `cli.py`.
Tests cover source parsing, feature leakage, mathematical densities and branch
symmetry, frozen encoder behavior, serialization, exact CPU interruption/resume,
inference and the complete synthetic workflow.
