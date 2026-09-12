# V2 implementation plan and scientific decision register

The complete `proposal.md` was read before this plan. Its §§12–16 define the
architecture, but it contains no specified training objective, numerical
hyperparameters, feature definitions, target extraction algorithm, split ratios,
metrics, or named evaluation experiments. The executable baseline below fills
those operational gaps explicitly; these are **implementation choices, not
claims that the proposal specifies them**. No new response labels, compounds,
or forward-response model will be invented.

## Architecture (required by the proposal)

1. Separately train a VAE on cell metadata, then retain only its encoder and freeze
   it throughout final model training. Use the posterior mean as its deterministic
   compressed representation (an explicit baseline choice).
2. Cell and shared Siamese drug encoders: linear → ReLU → linear. Inhibition
   encoder: linear → ReLU.
3. Implement all four independent context projections, the shared ordered-pair
   projection, and the exact two-coordinate products in §14.
4. Implement the shared deep MLP on both ordered concatenations and concatenate
   the two results. Fuse `[Z1, Z2, HA, HB]`.
5. An unrestricted MDN head produces M correlated bivariate Gaussian components,
   softmax weights, unrestricted means, softplus standard deviations, and tanh
   correlations. Numerical floors prevent singular covariance matrices.
6. The paired branches are swap equivariant; the specified unrestricted MDN head
   is not mathematically guaranteed to be. Preserve it and report a final-output
   swap diagnostic. Do not silently add symmetrization or a symmetry loss.

## Data and baseline protocol (not specified by the proposal)

- Use corrected DREAM matrices only by default. Support explicit OI inclusion,
  recording the source; do not automatically pool different screens or duplicate
  original raw matrices and corrected releases.
- Parse every matrix's internal Agent1/Agent2, units, axes, response grid and title.
  Canonicalize the ordered drug pair together with its dose coordinates. Preserve
  source file, experiment, source row/column, raw response and labels.
- The supplied Hill-fitting code fixes the zero-dose response at 100. Interpret
  matrix values as percent control viability and use `inhibition=1-response/100`.
  Before any split, keep only the inclusive inhibition range `[-2.0, 2.0]`.
  Remove rather than clip values outside it, preserving their raw response and
  computed inhibition in the exclusion table under
  `response_outside_quality_range`. Preserve nonphysical responses within the
  quality range without clipping.
- Each measured grid point supplies its observed inhibition input and dose target.
  Keep all measured alternatives and replicates; do not interpolate isoboles or
  collapse multiple solutions. Include zero-dose controls and monotherapies;
  make this configurable. The model learns a conditional distribution under the
  observed experimental sampling scheme, not a guaranteed feasible isobole.
- Convert recognized concentration units to µM. Transform doses using
  `log1p(dose_um / configured_reference_um)`, then one shared scalar mean/std
  fitted only on the training dose coordinates. This preserves zero and order.
  Gaussian support outside nonnegative doses remains explicit at inference.
- Resolve DREAM cells to CMP through unique COSMIC identifiers. Use only name
  fields supplied in cell metadata plus explicitly configured, provenance-bearing
  aliases. Quarantine KMSII and conflicting BT474/BT474C matrices unless evidence
  is configured; do not guess their identities.
- Use RDKit Morgan fingerprints from released, parseable SMILES, with primary
  chemistry precedence and supplemental fallback. Complete source inspection
  found 42 numeric PubChem-only identifiers. Resolve those exact released CIDs
  through PubChem, with response bytes, identity checks and hashes stored in an
  immutable cache. This explicit identifier-resolution step preserves coverage;
  it introduces no guessed compounds. An optional local-only baseline excludes
  numeric-only identifiers. Both modes exclude the seven known structure gaps.
- Stream CMP RNA-seq for every catalogued Broad/Sanger RNA-seq cell-line model,
  with `gene_id` as the stable key. Aggregate multiple finite, nonnegative TPM
  measurements for each model/gene by arithmetic mean and record
  multiplicities/source fields. Log1p TPM; select variable genes and fit
  imputation/standardization on the internal VAE-training models alone, excluding
  its internal metadata-validation cohort and every final evaluation identity.
- Seeded 70/15/15 cell-disjoint splits are the cold-start baseline. All replicates,
  doses and reversed pairs for a cell stay together. Provide a grouped warm
  completion alternative, grouping all replicates of a cell/unordered pair.
- The offline VAE uses the maximum released CMP cell-line RNA-seq population.
  Stable sample/patient/database IDs, normalized model names and parent
  relationships define conservative identity components. Final
  validation/test-only components are excluded before the VAE split; related
  models cannot cross boundaries. Every remaining profile enters VAE training or
  its deterministic internal validation subset. Neither final validation nor test
  metadata enter VAE fitting, internal selection or preprocessing.
- Offline objective: summed Gaussian reconstruction squared error (one-half)
  plus analytic diagonal-Gaussian KL to N(0,I), averaged over cells. Final objective:
  exact stable correlated bivariate mixture negative log likelihood (NLL). Use
  final-stage L2 weight decay `1e-3`, selected on validation only, to regularize
  the many repeated dose-grid observations against the much smaller number of
  independent training cell identities; leave offline-VAE weight decay at zero.
- Select offline and final checkpoints by their own validation objectives.
  Report test NLL, coordinate errors for the mixture-mean point summary, errors
  in original units, per-cell/pair/response-range summaries, near-50% observed
  response results with an explicit tolerance, and final-output swap diagnostics.
  Point summaries do not guarantee target achievement. No forward efficacy or
  clinical validation claim is possible from this inverse model alone.
- Extend evaluation without changing training: response-blind fixed-target
  grid recommendations followed by held-out scoring; primary macro-identity,
  fixed inhibition-stratum, near-target and worst-group reporting; intact
  identity-cluster intervals; and PIT/coverage/covariance diagnostics. Cross
  independent partition and full-pipeline training seeds, keep protocols
  separate, and preserve all declared runs. See the versioned
  [assessment protocol](evaluation_assessment.md) for estimands and limitations.

## Engineering sequence

1. Package, strict configuration, atomic artifact utilities and CLI.
2. Immutable-source inspection, complete parsing, chemistry/cell mapping,
   expression extraction and exclusion/provenance manifests.
3. Grouped splits, train-only fitted transforms and validated tensor datasets.
4. Exact model, mathematical and branch-symmetry tests.
5. Separate VAE and final trainers, validation selection, persistent event logs,
   atomic checkpoints including optimizer/scheduler/RNG/sampler position, and
   strict experiment identity checks on resume.
6. Evaluation tables, portable prediction bundles, distribution/grid/sample
   inference, user-supplied expression for unseen cells and experiment runner.
7. Synthetic end-to-end and real-source smoke validation, leakage and resumption
   tests, locked dependencies, reproducibility documentation and honest results.

## Initial environment inspection

Apple Silicon macOS, 12 logical CPUs, 36 GiB unified memory, MPS available and no
CUDA. The shell's initial Python belongs to another project; create a dedicated
local `.venv` using Python 3.12. CPU is the deterministic reference backend. Full
research training is configurable and is not conflated with smoke validation.

## Local source evidence

`Combenefit_dose_response_fitting_code.txt` lines 131–144 fixes `EC0=100.0`.
The primary methodology describes matrices as percent control:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5018366/ and the DREAM publication
describes viability measurements: https://www.nature.com/articles/s41467-019-09799-2.
`cell_info.csv` and the CMP model list establish COSMIC mapping, but do not
establish KMSII or BT474C aliases. See generated manifests for exact source hashes,
resolved labels, exclusions, and effective configuration of each execution.
