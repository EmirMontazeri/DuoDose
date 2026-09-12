# Recorded results and verification

> Historical-result notice (pipeline schema 1): these runs used the earlier
> 85-model expression extraction and 48/8 cold-split VAE cohorts. The current
> pipeline uses all eligible CMP cell-line RNA-seq models with identity-grouped
> leakage exclusion. Existing metrics remain an audit of the old checkpoints and
> must not be presented as results of the expanded-VAE pipeline.

This review, dated 2026-09-11, inspected the completed artifacts already present
in `runs/`. It did not start another full research training run. The full DREAM
baseline, six configured experiments, and reduced engineering smoke run have
saved completion records, selected models, continuation checkpoints, and
validation/test outputs. These results use the explicit baseline decisions in
[protocol.md](protocol.md); the proposal itself does not specify these split
ratios, hyperparameters, losses, or evaluation metrics.

## Full DREAM baseline

[runs/v2/result.json](../runs/v2/result.json) records a completed CPU experiment
using [configs/v2.yaml](../configs/v2.yaml). Its prepared dataset is
`dataset/splits/v2_477d7b35d901`, fingerprint
`ec80392b8c9b79a97cad6d41d78e745b7ac1007cc9fae288c69b54b92c0c7623`.

The source pass inspected all 11,608 corrected DREAM matrices and retained
9,931 matrices containing 357,516 observations across 80 resolved cells and
110 observed drugs. The saved drug feature dictionary contains 112 supported
agents, including two absent from retained observations. The source manifest
records 1,102 unsupported-structure, 615 cell-label-mismatch, and 634
unresolved-cell exclusion reasons. Reasons can overlap on the same matrix;
these counts must not be summed to obtain the number of excluded matrices.
Raw values, identifiers, exclusions, and source hashes remain in the
[prepared source manifest](../dataset/splits/v2_477d7b35d901/source_manifest.json).

| Cohort           | Cells | Observations |
| ---------------- | ----: | -----------: |
| Final training   |    56 |      226,692 |
| Final validation |    12 |       77,292 |
| Test             |    12 |       53,532 |

The offline VAE uses 48 of the final-training cells for fitting and the other
8 for metadata-only validation. Gene selection, imputation, and expression
scaling use only those 48 VAE-training cells. Final validation/test cells do
not overlap final training. Exact cell lists are recorded in the
[split manifest](../dataset/splits/v2_477d7b35d901/split_manifest.json).

The VAE completed 72 epochs and selected epoch 57. The final model completed
17 epochs and selected epoch 2. Both stopped after 15 validation epochs without
improvement. These are completed runs under the configured early-stopping
procedure, rather than claims that the maximum 200/100 epochs were executed.

| Metric                                     |         Validation |         Test |
| ------------------------------------------ | -----------------: | -----------: |
| NLL, standardized log-dose coordinates     |        0.331693441 |  0.303809047 |
| NLL, original µM coordinates               |       -0.931728142 | -0.945128087 |
| Coordinate MAE, standardized log dose      |        0.540850005 |  0.514621610 |
| Coordinate MAE, µM                         | 1.888677467 × 10¹⁶ |  0.367976225 |
| Coordinate RMSE, µM                        | 6.148822081 × 10¹⁸ |  0.730080578 |
| Point summaries with a negative coordinate |          0.499405% |    0.788314% |

The original-unit NLL includes the dose-transform Jacobian. Continuous-density
NLL can be negative. The point summary is the inverse transform of the
mixture mean in standardized log-dose coordinates; it is not the mean of the
distribution in µM, nor a guarantee of target inhibition. Table entries are
rounded; JSON and per-observation Parquet files preserve the recorded precision.

The test subset with measured inhibition within `0.5 ± 0.05` contains 3,752
observations across 12 cells and 348 drug pairs. Its standardized-coordinate
NLL is 1.625021100, original-unit NLL is 0.621994864, and coordinate MAE is
0.560964828 µM. This evaluates observed near-target rows; it does not establish
that newly proposed doses achieve 50% inhibition. Full per-cell, pair, matrix,
and inhibition-range tables are under
[runs/v2/evaluation](../runs/v2/evaluation).

## Severe validation outliers remain visible

The very large validation errors above are present in the saved results and
must not be omitted when describing performance. The most extreme row belongs
to HCT-116 with IAP/PTK2, source
`dataset/raw/az_dream/DREAM_Challenge_matrices_corrected/IAP&PTK2&HCT116&Exp1.xlsx`.
Reading that workbook directly confirms the percent-control response
`-4096.5` at grid row 4, column 3. The documented conversion therefore gives
`1 - (-4096.5)/100 = 41.965` inhibition. This is a source value, not a
percent/fraction conversion or axis-order error. The immutable workbook SHA256 is
`238905530e9b691d75982459cf8835eda911801f67aa020f084bce0b74738130`.

In canonical IAP/PTK2 order, that observation has measured doses
`(0.003, 0.3)` µM. Its stored predicted standardized coordinates are approximately
`(62.347492, 117.661285)`, which become
`(2.393997980 × 10¹¹, 2.355198801 × 10²¹)` µM after exponentiation.
The training inhibition range is `[-1.283, 1.213]`, so this conditioning value
is far beyond the training range. Large predicted log coordinates followed by
the exponential inverse transform explain the extraordinary original-unit
errors. The underlying experimental cause of these source responses is not
established by the available records.

There are 15 validation observations with mean coordinate absolute error above
`10⁹` µM: 14 from HCT-116 IAP/PTK2 and one from HCT-116 IAP/PARP1. The validation
inhibition range is `[-2.505, 41.965]`; the test range is `[-1.011, 1.098]`.
Across train/validation/test there are respectively 32,772 / 9,666 / 7,261
measured inhibitions outside `[0,1]`. The implemented baseline preserves these
values without clipping or silently excluding them. Their presence is a major
limitation of interpreting aggregate metrics and extrapolated dose predictions.

## Six recorded experiments

[runs/suite/experiments.csv](../runs/suite/experiments.csv) and
[suite.json](../runs/suite/suite.json) contain the following completed test
results. `cell` holds out entire cells. `cell_pair` holds out all replicates of
a cell/unordered-drug-pair group while allowing cell metadata to overlap
training; it measures warm completion. The two protocols do not constitute the
same prediction task. The `cell`, seed-42 result matches `runs/v2` and should
not be counted as an additional independent replicate.

| Protocol  | Seed | Test rows | Test cells | NLL, scaled |   NLL, µM | Coordinate MAE, µM |
| --------- | ---: | --------: | ---------: | ----------: | --------: | -----------------: |
| cell      |   42 |    53,532 |         12 |    0.303809 | -0.945128 |           0.367976 |
| cell      |   43 |    49,572 |         12 |    0.366338 | -0.953702 |           0.413213 |
| cell      |   44 |    61,776 |         12 |    0.231218 | -1.090111 |           0.412103 |
| cell_pair |   42 |    53,892 |         79 |   -1.075678 | -2.383975 |           0.377937 |
| cell_pair |   43 |    53,640 |         80 |   -0.697140 | -2.005075 |           0.364698 |
| cell_pair |   44 |    53,676 |         80 |   -1.828570 | -3.135844 |           0.371122 |

Each cell-disjoint experiment has 56/12/12 training/validation/test cells and
48/8 offline VAE training/validation cells. Each warm experiment has
6,816/1,461/1,461 cell-pair groups and 68/12 offline VAE training/validation
cells. All use the same 357,516 retained observations; row counts differ by
the grouped assignments. Scaled density values depend on the training-fitted
dose transform, and the test populations differ across seeds. This table is
an experiment record, not a statistical superiority claim.

## Engineering smoke and software checks

[runs/smoke/result.json](../runs/smoke/result.json) records the completed
[smoke configuration](../configs/smoke.yaml): at most 240 source matrices,
64 gene features, reduced model widths, and two epochs per stage. Its prepared
data contain 7,524 observations, split into 5,364/1,080/1,080 rows across
47/10/10 cells, with 40/7 offline VAE training/validation cells. Test NLL is
1.589423776 in standardized coordinates, and coordinate MAE is 0.440632200 µM.
This is an integration check, separate from the full experiments above.

The historical smoke bundle is `dataset/splits/v2_02710809f6d5`, fingerprint
`57b0da81e5249d2185a42843049934c9b00af61e8b42492c514f45a158f1d87d`.
The historical full bundle is `dataset/splits/v2_477d7b35d901`. Both runs
were regenerated using the final code, including source schema `source-v2.3`.

## Pre-QC expanded-VAE pipeline verification

Earlier on 2026-09-12, full preparation under data schema 2 produced
`dataset/splits/v2_f36085a4152b`, fingerprint
`68afd06a9a93634e8419b1e81e40da079a3f1db1f49ca1522f8e88c32accc36b`.
The expression source contains all 1,643 catalogued Broad/Sanger cell-line
RNA-seq models. Twenty-five models in 24 final validation/test identity groups
were excluded, including one related external profile. All 1,618 remaining
models were assigned: 1,379 models (1,303 identity groups) to VAE training and
239 models (230 groups) to internal VAE validation. Pairwise intersections among
VAE-training, VAE-validation and excluded identity groups are all zero. The final
response train/validation/test identity-group intersections are also all zero.

A real-source two-epoch integration run is saved under
`runs/smoke-expanded-vae`. It used 1,375/246 VAE train/internal-validation
profiles and 4,716/1,080/1,728 final train/validation/test response rows. The
offline VAE and frozen-encoder final stage both completed; validation NLL was
1.777403712 in standardized log-dose coordinates. Test evaluation was
deliberately skipped, so this run verifies integration and does not report or
use test performance.

These artifacts predate inhibition quality control and are retained only as an
integration record. They must not be resumed or reported as results of the
current source schema.

## Current inhibition-QC pipeline verification

Source schema `source-v2.4` applies the fixed inclusive inhibition range
`[-2.0,2.0]` before splitting. The complete DREAM rebuild removed 54 of 357,516
otherwise eligible observations and retained 357,462. Every removed row is in
`dataset/processed/dose_response/v2_2a3cd80276be/exclusions.parquet` under
`response_outside_quality_range`, with its original response and unmodified
computed inhibition. Retained inhibition spans `[-2.0,1.873]`; the removed values
span `[-2.505,41.965]`.

The current full prepared bundle is `dataset/splits/v2_6fb47dab2bd0`, fingerprint
`d858bc8152c49bed3e4049f2fcb5230abcb1e1e7ec4a79d50db86da047eae7e2`.
Because QC precedes grouped allocation, its 357,462 observations split into
244,008 training, 54,306 validation and 59,148 test rows. The VAE population
remains 1,379 training, 239 internal-validation and 25 final-evaluation-identity
excluded profiles.

The current two-epoch integration run is `runs/smoke-qc-expanded-vae`, using
prepared bundle `dataset/splits/v2_220e77d01e9a`, fingerprint
`593f9030d9b9be3ec9cf112eedea1390ab09e24189d582c21a8195145b00dbf0`.
Its systematic 240-matrix subset contained no inhibition beyond the quality
range, so its 7,524 rows and validation NLL `1.777403712` are unchanged. Test
evaluation was deliberately skipped.

The historical recorded check passed 93 tests. The expanded-VAE implementation
with inhibition quality control currently passes 100 tests and Ruff checks.
Relevant checks include:

- Source parsing, concentration units, identifier provenance, source
  immutability, missing-data policies, and preprocessing artifact integrity.
- Train-only feature fitting, held-out-cell leakage tests, grouped replicate
  isolation, and supplied-gene order/alignment at inference.
- Gaussian-mixture log density against PyTorch's full covariance distribution,
  analytic VAE terms, finite gradients, frozen VAE behavior, exact paired-branch
  drug swaps, and a counterexample to unrestricted-head output equivariance.
- Exact CPU interrupted-versus-uninterrupted VAE/final runs, including model,
  optimizer, scheduler, sampler, RNG, model selection, and early-stop state;
  rejection of changed datasets/configuration/source identities before writing.
- Completed-VAE resumption and reconstruction of missing selection artifacts
  without changing the final experiment's semantic encoder identity.
- Synthetic training through evaluation, inference and export; physical
  density/Jacobian agreement with an independent SciPy Gaussian calculation;
  rectangular grid mass against numerical Gaussian integration; caller-order
  restoration; missing TPM imputation; deterministic sampling; and exported
  inference while the original experiment directory is unavailable.

These checks establish software behavior and recorded experiment continuity.
They do not establish biological validity, target-achievement calibration,
clinical utility, or robustness to all unseen cells and experimental conditions.
The final MDN remains unrestricted: for the full test run its direct
model-order diagnostic gives mean coordinate disagreement 0.028822631 in
standardized units and mean absolute log-density disagreement 0.630231440.
The public predictor instead canonicalizes input order and restores caller
axes explicitly; this API convention does not alter the specified network.

## Reproducing the recorded code identity

At the time of the 2026-09-11 review, all 11 Python module hashes in the eight
model experiment records (`runs/smoke`, `runs/v2`, and the six suite runs)
matched that revision. They no longer match the expanded-VAE implementation and
cannot be resumed under the current prepared-data schema. Their response source
schema remains `source-v2.3`.

Reproduce the current expanded-VAE protocol in fresh directories:

```bash
uv run --frozen reverse-dose run --config configs/smoke.yaml --run-dir runs/smoke-qc-expanded-vae
uv run --frozen reverse-dose run --config configs/v2.yaml --run-dir runs/v2-qc-expanded-vae
```

To reproduce the declared protocol as a new experiment, choose a new run
directory. Preserve the source files, `uv.lock`, configuration, raw inputs,
exact-CID response cache and prepared artifact identity. Source or runtime changes are intentionally rejected on strict continuation. `last.pt` preserves training state; `best.pt` is the selected evaluation/inference artifact.
