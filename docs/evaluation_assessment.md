# Evaluation assessment protocol

The assessment reports coordinate density, target achievement, uncertainty and robustness using measured response assays and the [model architecture](architecture.md). Recorded outcomes are in [evaluation results](evaluation_results.md).

## Primary reporting set and biological units

Report `cell` and `cell_pair` in separate tables throughout:

| Protocol | Held-out unit | Evidence supported |
| --- | --- | --- |
| `cell` | Complete CMP identity component, all its pairs/matrices/rows | Cold-start to unseen cell identities |
| `cell_pair` | CMP identity component × unordered drug pair, all replicates | Warm completion of held-out combinations in a population largely observed during training |

The prepared manifest records overlap; do not infer cold-start evidence from
the generally stronger warm results. CMP identity components, not aliases or
individual dose-grid rows, define cell-level biological support. Every result
records row, identity, identity/pair and matrix counts.

The primary reporting set contains all of the following, without choosing the
most favorable result after test evaluation:

1. Equal-identity NLL, coordinate MAE and squared error, in scaled log-dose and
   original µM units, with identity-cluster intervals. Physical-unit NLL includes
   the saved transform's Jacobian and is preferable for cross-partition density
   comparisons when fitted dose scales differ. Group CSVs also report RMSE.
2. Fixed inhibition strata: `[-inf,0)`, `[0,0.25)`, `[0.25,0.5)`, `[0.5,0.75)`,
   `[0.75,1)` and `[1,inf)`. QC restricts actual observations to `[-2,2]`.
   Show each stratum's row fraction, identity support, equal-identity NLL and
   MAE with intervals. The equal-bin NLL averages nonempty bin means, with
   empty bins explicitly listed; it is not a deployment-prevalence estimate.
3. Near-target coordinate-density results for observed inhibition within
   `0.5 ± 0.05`, including the number of contributing rows and identities.
   This subset is distinct from making a recommendation at fixed target 0.5.
4. Per-identity, drug-pair, identity/pair and matrix tables with support counts;
   maximum group error and the mean of the worst `ceil(0.10 × group_count)`
   groups, separately ranked by NLL and MAE. Small-support failures remain visible.
5. Fixed-target achievement, calibration/coverage and covariance diagnostics
   below. Overall row-weighted metrics remain available, not privileged as the
   only summary. No reweighting changes the trained objective.

Macro-identity metrics first average all eligible rows within each identity,
then average identities equally. They do not additionally equalize drug pairs
within each identity. Target reporting provides both matrix-within-identity and
pair-within-identity weighting explicitly. These estimands answer different
questions and must be labeled rather than mixed.

## Leakage-safe retrospective target achievement

The fixed policy is declared in code/configuration before evaluation: request inhibition 0.5; summarize the frozen MDN by its mixture mean
in training-standardized log-dose space; snap to the nearest original grid
coordinate in Euclidean distance in that space; score a hit when the revealed
response lies within 0.05 inhibition units of the request.
The assessment permits only float64 round-off at the inclusive window
edges (below 2e-15 at the default target), preventing exact 45% or 55% responses
from falling outside the window due to binary subtraction. The same predicate is used for
near-target reporting, target hits and the oracle availability diagnostic.

For each matrix already included in the prepared held-out split:

1. Verify the original workbook SHA-256 against the saved source manifest.
   Reconstruct its full Cartesian dose axes, canonicalizing drug order exactly
   as in preprocessing. Preserve original coordinates and observation IDs.
   Apply only the configured zero-dose inclusion policy. Do not infer the grid
   from surviving response rows: missing and QC-excluded coordinates must still
   be candidates. Completely excluded matrices remain outside the fixed cohort.
2. Pass only the whitelisted cell/drug identities, matrix/grid IDs and dose
   coordinates to the selector. It rejects extra columns such as inhibition.
   One model query per distinct cell/pair at fixed target supplies the prediction
   for each of that context's matrices. Held-out expression is transformed only
   by the saved training-fitted preprocessing; nothing is fitted here.
3. Select by distance alone. Ties resolve by dose A, dose B, original row and
   column. Save `target_choices.parquet` before joining any response. Never
   inspect attainable responses to select a component, coordinate or policy.
   No clipping of the unsnapped mean is performed.
4. Join the QC-retained response by the exact selected observation ID and save
   `target_scores.parquet`. Missing/QC-excluded selections are unscorable and
   are never replaced. Report their counts, success among scorable selections,
   and all-matrix success bounds `[hits/N, (hits+unknown)/N]`.
5. Report equal-identity hit rates, equal-pair-within-identity hit/error rates,
   intact-identity intervals, snap distance, and the invalid unsnapped-mean
   fraction. The fraction of matrices containing any observed near-target point
   is an explicitly labeled oracle availability diagnostic, not a predictor or
   an eligibility filter. Unattainable matrices remain in the denominator.

Measured responses are used only for final scoring and diagnostics. Assay dose
axes are permitted evaluation design information, not a fitted response model.
The result establishes the retrospective behavior of this **grid-snapped
recommendation policy on retained matrices**, not response at the continuous
generated mean, interpolation validity, causality or prospective effectiveness.
There is no fabricated response between grid points and no need for new assays.
Portable bundles without raw axes explicitly report target assessment unavailable
instead of silently constructing a response-dependent candidate set.

## Hierarchical uncertainty and limitations

For each fixed checkpoint/split, draw 2,000 bootstrap samples of its `G` CMP
identity components, sampling `G` components with replacement. Retain all pairs,
matrices and dose rows of each sampled component together. Macro estimates
average sampled identity means; row-weighted estimates recompute the ratio of
sampled cluster sums to sampled cluster sizes. The 2.5th and 97.5th percentiles
form the default 95% interval. The bootstrap seed is independently fixed to
20260912. Settings are validated and saved in `assessment.json`.

No iid-row standard error or row-level goodness-of-fit p-value is reported.
Duplicating every row a thousand times cannot narrow a macro-identity interval;
a regression test enforces this. Fewer than two identities gives no interval;
fewer than twenty sets a warning. An empty near-target/stratum subset is not
imputed as success or zero error.

These are conditional cluster-bootstrap intervals, not a guarantee of nominal
coverage with twelve cold-test identities. They retain within-cell dependence
but do not model shared-drug-pair dependence across identities or assay noise
separately. They condition on the assayed pair collection and assume identity
clusters are exchangeable for that limited estimand. Matrix rows, warm cells
already observed in training, and repeated appearances of the same identities
across partitions do not create new independent cold biological evidence.
Worst-group ranks and maxima are descriptive, not selection-adjusted confidence
bounds. Pair groups represented in only one identity are labeled accordingly.

## Independent partition and training randomness

The factorial runner materializes the full Cartesian product, never a zip of
the seed lists. `split.seed` determines response partitions, VAE metadata split
and the resulting train-fitted transforms. Top-level `seed` controls VAE
initialization/sampling and final-model initialization/minibatch randomness.
This measures full-pipeline training sensitivity, not final-MDN initialization
alone. A training-seed comparison within a partition uses the same prepared
bundle. Architecture, optimization settings and selection policy stay fixed.

The factorial design uses five split seeds (42–46), two training seeds (42,43)
and both protocols: ten combinations per protocol. Matching completed runs can
be reused with verified configuration and artifact provenance. Reused fits are
identified explicitly and do not count as independent training repetitions.

`design.json` freezes configuration, combinations, reference-suite path and test
policy with a fingerprint. `progress.json` counts completed/planned combinations.
Resume requires exactly the same design and preserves the trainer's existing
checkpoint identity checks. Evaluate every declared run; do not prune failures
or change hyperparameters in response to test scores.

For a complete design within each protocol, `summary.json` reports the SD of
partition means, root-mean-square within-partition training SD, SD of training
seed means and split-by-training interaction RMS. These are descriptive
decompositions, not unbiased population variance-component estimates. With one
fit per combination, interaction cannot be distinguished from other residual
variation. Two training seeds remain modest support. Across-run SD is not a
standard error of independent biology, and overlapping tests are not pooled to
manufacture a larger effective sample size.

The suite provides retrospective robustness evidence. Every run maintains its
train/validation/test fitting boundary; repetitions share biological identities.

## Density and covariance characterization

For each saved, index-aligned mixture, compute exact marginal PIT values for A
and B and the A-then-B Rosenblatt transform. The latter conditions mixture
weights on observed A before evaluating B's conditional CDF; prior component
weights would be incorrect. Report PIT histograms, means and variances and
central marginal coverage at 50%, 80%, 90% and 95%. Joint coverage uses a
Rosenblatt rectangle with each marginal side probability `sqrt(nominal)`.
Coverage has both group summaries and macro-identity bootstrap intervals.

Report minimum component scale, minimum component covariance eigenvalue,
mixture-weighted component log determinant, mixing-weight entropy, maximum
absolute correlation, NLL and swap-log-density quantiles through the 99th
percentile and maximum. The minimum covariance eigenvalue uses determinant
divided by the largest eigenvalue to avoid subtractive cancellation. Minima
include tiny-weight components; these are warning diagnostics, not by themselves
proof of a high-probability failure. Mixing-weight entropy is not differential
entropy of the continuous mixture.

The observations lie on discrete experimental grids. Continuous PIT uniformity
is therefore a calibration diagnostic for the stated density, not continuous
biological ground truth. No post-hoc calibration is fitted, no covariance is
changed, and no conclusion that negative-dose probability has been repaired is
implied. Per-row diagnostics permit inspection of joint NLL, error, covariance
and swap failures for the same biological cases.

## Reproduction and artifacts

Use the commands in [README](../README.md#experiments-and-artifacts). Standalone
reassessment verifies the selected checkpoint against the original evaluation,
configuration, dataset fingerprint and observation membership. It records hashes
of the selected checkpoint, source predictions and mixture arrays, plus runtime
and source identity. Original evaluations are not overwritten.

Each assessment directory contains `assessment.json`, `diagnostic_rows.parquet`,
five `primary_by_*.csv` tables, and (when raw axes are available)
`target_choices.parquet` / `target_scores.parquet`. Training-generated assessments
sit beside the original evaluation's `metrics.json`, which records checkpoint
and dataset identity and promotes the primary assessment into its results.
Suite CSV/JSON files separate protocols and retain each run's actual path and
both seeds. Always check factorial completion counts before calling its summary
a completed ten-run-per-protocol result.
