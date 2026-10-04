# Evaluation results

The completed factorial evaluation contains twenty combinations: five partition
seeds (42–46) × two training seeds (42,43) × two protocols. The saved
[progress](../runs/five_split_two_train_seed_run/progress.json) records 20/20
completed combinations. The [summary](../runs/five_split_two_train_seed_run/summary.json)
and protocol-specific CSV files contain the full results and provenance.
These generated artifacts are local experiment outputs and are not included in
the source distribution. Run the commands in [README](../README.md) to reproduce
this design. Metrics below are taken from the saved summary.

| Protocol | Runs | Mean row NLL | Mean macro-identity NLL (µM) | Mean macro-identity MAE (µM) | Mean macro-identity target hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cell` | 10 | 0.268 | -0.907 | 0.472 | 11.74% |
| `cell_pair` | 10 | -0.006 | -1.220 | 0.462 | 9.72% |

`cell` evaluates unseen cell identities. `cell_pair` evaluates held-out pairs in
a population largely represented during training. These protocols support
different generalization claims and must be reported separately.

Target hit rate requests inhibition 0.5, snaps the model's transformed-space
mixture mean to the nearest assay coordinate without seeing responses, and
scores the selected measured inhibition within ±0.05. Missing or QC-excluded
selected coordinates remain unscorable; they are never replaced.

Across-run means and variability are descriptive. Test identities overlap across
partitions, and repeated dose-grid rows are not independent biological samples.
Identity-cluster intervals condition on the selected checkpoint and partition.
Low target hit rates and density diagnostics remain scientific limitations.
Coordinate accuracy does not establish target response at arbitrary unmeasured
doses or prospective effectiveness. See the [assessment protocol](evaluation_assessment.md)
for estimands, support counts, calibration and uncertainty definitions.
