# Current retrospective evaluation results

Assessment date: 2026-09-12. Source: the six completed, quality-controlled,
expanded-VAE runs in `runs/suite`, with 357,462 total retained response rows
across 80 mapped cell identities. These are not the historical numbers in
[validation_results.md](validation_results.md). Architecture, likelihood,
optimization settings and selected weights are unchanged by this reassessment.

The authoritative corrected outputs are
[suite-assessment-v1.1](../runs/suite-assessment-v1.1/summary.json), with
[cold results](../runs/suite-assessment-v1.1/cell_results.csv) and
[warm results](../runs/suite-assessment-v1.1/cell_pair_results.csv).
The preliminary `suite-assessment-v1` is superseded: schema v1.1 corrects binary
round-off at the inclusive 45%/55% boundaries. Original predictions, model
checkpoints and preliminary artifacts were preserved, not overwritten.
See [the assessment protocol](evaluation_assessment.md) for exact estimands,
leakage safeguards, provenance, formulas and reproducible commands.

## Cold-start evidence: `cell`

Each test partition contains only **12 independent held-out identity groups**.
Counts of 51,804–59,148 grid rows are not independent biological sample sizes.
NLL below is in standardized log-dose space; higher is worse. Macro-cell means
equal-weight identity groups rather than observation rows.

| Split/training seed | Test rows | Row NLL | Macro-cell NLL | Near-50% macro-cell NLL | Target hit rate, macro-cell (95% cluster interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42/42 | 59,148 | 0.546 | 0.652 | 1.839 | 8.79% (5.91–12.27%) |
| 43/43 | 55,080 | 0.190 | 0.380 | 0.844 | 12.24% (8.79–15.64%) |
| 44/44 | 51,804 | 0.206 | 0.368 | 1.759 | 12.38% (8.28–16.39%) |

Near-target subsets contain 3,586, 4,324 and 3,923 rows respectively, each
spanning all twelve test identities. Overall macro-cell NLL is worse than row
NLL in every cold run. Near-target NLL is worse still. These discrepancies are
now primary results, not hidden behind an overall average.

## Warm-completion evidence: `cell_pair`

The 79–80 test identities below are largely represented in training through
other drug pairs. They are **not** 79–80 unseen-cell tests.

| Split/training seed | Test rows | Identities | Row NLL | Macro-cell NLL | Near-50% macro-cell NLL | Target hit rate, macro-cell (95% cluster interval) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42/42 | 53,748 | 80 | 0.005 | 0.004 | 1.012 | 11.12% (8.97–13.64%) |
| 43/43 | 53,568 | 79 | 0.104 | 0.087 | 0.981 | 8.67% (7.10–10.41%) |
| 44/44 | 53,662 | 80 | −0.001 | 0.101 | 0.773 | 9.37% (7.40–11.50%) |

Near-target subsets contain 3,491, 3,264 and 3,204 observations respectively.
Near-target density performance is again worse than overall performance.
Stronger warm density scores do not establish equivalent cold-cell performance
or better biological target achievement.

## Target achievement and response imbalance

The recommendation policy queries the frozen model at exactly 50% inhibition,
then snaps its transformed-space mean to the nearest original assay coordinate
using dose distance only. Responses are revealed after choices are saved. Hits
require measured inhibition in the inclusive 45–55% window. The table therefore
tests a discrete grid recommendation, not an unmeasured continuous dose.

Cold test sets have 1,643, 1,530 and 1,439 matrices; all selected coordinates are
scorable. Respectively 66.28%, 73.40% and 67.69% of those matrices contain *some*
measured near-target coordinate. Warm sets contain 1,493, 1,488 and 1,491 matrices,
with near-target-coordinate availability 67.65%, 66.20% and 66.20%. This oracle
availability is a matrix-weighted diagnostic, not the macro-cell estimand and
not a response-informed recommendation rule. The low actual hit rates show that
target selection remains poor even though many assayed grids contain a hit.

One warm seed-44 selected coordinate is QC-excluded/missing and remains
unscorable: no substitute is selected. Its all-matrix, matrix-weighted hit-rate
bounds are 8.853–8.920%, explicitly separate from the macro-cell interval above.
Matrices without any near-target response remain in the assessment population.

Inhibition below 25% accounts for 46.98–52.06% of cold test rows and 49.38–50.50%
of warm test rows. It is the largest combined response region, but should not be
described as nearly all the data. For cold seed 42, stratum-specific macro-cell
NLL is −0.854 in `[0,0.25)`, 1.108 in `[0.25,0.5)`, 1.866 in `[0.5,0.75)` and
2.060 in `[0.75,1)`. Each run now publishes fixed-stratum proportions, support
counts and cluster intervals, plus an explicitly equal-bin summary. This
corrects how performance is represented; it does not improve the trained model.

## Failure groups and density diagnostics

All cell/pair/matrix tables retain failures with support counts. Examples:

| Protocol/seed | Worst cell | Rows in cell | Cell NLL | Worst 10% cells, mean NLL | Worst 10% pairs, mean NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| cell/42 | NCI-H2085 | 3,024 | 2.428 | 1.967 | 2.933 |
| cell/43 | KATOIII | 936 | 1.232 | 1.209 | 2.014 |
| cell/44 | VCaP | 432 | 3.179 | 2.094 | 2.535 |
| cell_pair/42 | VCaP | 180 | 3.788 | 1.776 | 2.033 |
| cell_pair/43 | EVSA-T | 180 | 1.856 | 1.223 | 2.154 |
| cell_pair/44 | VCaP | 72 | 2.839 | 1.554 | 1.836 |

For example, cold seed 42's Docetaxel/IAP group has NLL 8.832 across 180 rows
but only five identities. Cold seed 44's AR/PIK3CB_PIK3CD group has NLL 5.000
across 36 rows from one identity. Neither 180 nor 36 is its independent cell
sample size. These examples illustrate the need for support-aware reporting,
not a selection-adjusted statistical ranking of inherently difficult drugs.

Macro-cell joint coverage at nominal 90% is 86.85%, 89.78% and 90.86% in the
cold runs, versus 93.10%, 93.53% and 91.43% in the warm runs. The corresponding
cluster intervals, all marginal/other joint levels and PIT histograms are saved.
With twelve cold identities, deviations from 90% must not automatically be
called statistically established miscalibration.

Cold NLL's 99th percentiles are 8.682, 7.255 and 6.292; its maxima are 230.653,
17.467 and 25.684. Median minimum component covariance eigenvalues are
approximately 6.05e-4, 3.87e-6 and 7.58e-4. Cold seed 42's maximum swap log-density
disagreement is 69.07; seed 43's 99th-percentile disagreement is 6.25. These
diagnostics reveal density tails and covariance variation that aggregate MAE
does not capture. Minimum-eigenvalue diagnostics include low-weight components
and are not alone proof of failure in high-probability regions. No density
recalibration, truncation or architectural fix has been claimed.

## Independent-seed extension: progress, not a completed robustness claim

[The frozen design](../runs/factorial-v1.1/design.json) declares five partition
seeds × two training seeds for each protocol: twenty combinations total.
[Progress](../runs/factorial-v1.1/progress.json) is updated after each completed
combination. Check that it reads 20/20 before treating
[its summary](../runs/factorial-v1.1/summary.json) as the completed design.
Intermediate summaries describe only their completed subsets.

At the initial reporting snapshot, the four cold combinations crossing split
seeds 42/43 with training seeds 42/43 were complete:

| Split seed | Training seed | Row NLL | Scaled coordinate MAE | Macro-cell NLL | Macro-cell target hit rate |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | 42 | 0.546 | 0.606 | 0.652 | 8.79% |
| 42 | 43 | 0.528 | 0.589 | 0.657 | 9.46% |
| 43 | 42 | 0.928 | 0.636 | 0.934 | 16.22% |
| 43 | 43 | 0.190 | 0.675 | 0.380 | 12.24% |

Within split 43, the test rows and train-fitted transforms are identical.
Changing training randomness changes NLL by 0.738 while MAE changes by 0.038.
For the same 55,080 observed coordinates, the 99th percentile of absolute
between-fit NLL difference is 11.407, versus 0.773 for absolute difference in
per-row coordinate error; 6.21% of rows change NLL by more than five. These are
descriptive paired diagnostics, not iid significance tests. The run with worse
NLL has better target hit rate in this comparison, reinforcing that the endpoints
are not interchangeable. Two seeds alone do not establish a population variance.

Completed off-diagonal fits from the preliminary execution are reused through
`runs/assessment-references-v1.1`; no trained weights are discarded based on
test performance. The preliminary execution's incomplete member is preserved
but not reported as completed. The corrected suite uses unchanged training
settings and adds no new biological identities. Its exact continuation command,
if the process has stopped, is:

```bash
uv run --frozen reverse-dose experiments --config configs/v2.yaml \
  --output runs/factorial-v1.1 --split-seeds 42 43 44 45 46 \
  --training-seeds 42 43 --protocols cell cell_pair \
  --reference-suite runs/assessment-references-v1.1 --resume
```

Do not run a second writer against this directory while the suite is active.
All declared combinations retain validation-only selection; no test-driven
pruning or hyperparameter changes are allowed. This is retrospective robustness
evidence on existing identities, not an untouched confirmatory test.

## Verification and what remains unresolved

`uv run --frozen pytest -q`: **113 passed**. `uv run --frozen ruff check .`: clean.
Tests cover raw-axis reconstruction with missing/QC-excluded responses, response
blindness, no oracle reselection, inclusive numerical boundaries, analytic
Gaussian/mixture calibration, cluster rather than row resampling, both factorial
axes, immutable resume design and reference artifact/settings verification.

The evaluation and reporting gaps are addressed with current data and without
changing the proposal. More repetitions are implemented and running, not yet
claimed complete at this snapshot. The poor target hit rates, limited cold-cell
support and density failures remain scientific findings, not repaired model
properties. These data do not support a claim of prospective effectiveness or
response at arbitrary unmeasured generated coordinates; the assessment makes no
such claim and introduces no requirement for new assays.
