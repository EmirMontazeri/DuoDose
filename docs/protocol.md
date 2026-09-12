# V2 executable protocol and scientific decisions

## Specification boundary

[The proposal](../proposal.md) is authoritative for the model architecture. It
specifies the independently trained frozen VAE encoder, the input encoder layer
patterns, the four context projections, shared ordered drug-pair projection,
two-coordinate products, shared deep MLP, fusion, and correlated bivariate MDN
outputs. These components are implemented without an additional symmetry loss,
forward-response model, complex arithmetic or a learned cell-ID embedding.

The proposal does **not** specify the cell/drug feature definitions, VAE widths,
numerical hyperparameters, objectives, response-grid sampling, split ratios,
normalization, model-selection metric, numerical evaluation metrics or named
benchmark experiments. The following are explicit executable baseline choices.
They can be reviewed and changed through versioned code/configuration; they are
not presented as requirements quoted from the proposal. The earlier
[implementation plan](implementation_plan.md) records the design rationale.

## Source population, identifiers and observation construction

Corrected DREAM matrices are the default response source. Corrected OI matrices
can be included explicitly. The original `Raw_Data` matrices and original DREAM
expression archive remain source/reference data and are not duplicated into the
default training population.

Each readable corrected matrix supplies its internal Agent1/Agent2 identifiers,
axis concentrations/units, matrix title, experiment and measured response grid.
The parser validates grid content and converts recognized concentration units to
µM. The pair is canonicalized by case-sensitive lexicographic drug identifier,
with the corresponding dose coordinates moved together. Provenance retains the
original labels, order, source path and grid row/column. Released metadata can
establish a drug spelling synonym, such as `Gemcitibine` → `Gemcitabine`; its
evidence is recorded rather than inferred from string similarity.

The local Combenefit fitting code fixes its zero-dose response at 100. Combined
with the supplied response matrix representation, the baseline uses

\[
I = 1 - V/100,
\]

where \(V\) is measured percent-control viability and \(I\) is inhibition in
fraction units. Source preprocessing retains an observation only when
\(-2.0 \le I \le 2.0\), with inclusive boundaries. Values outside that quality
range are removed—not clipped—before dataset splitting and recorded in
`exclusions.parquet` with reason `response_outside_quality_range`, original
`response_percent`, unmodified computed `inhibition`, source file, matrix and grid
coordinates. Responses outside 0–100% remain valid when their inhibition is still
inside the quality range. Missing response points are excluded and counted under
the configured policy; unsupported units or malformed required data fail clearly.
Declared grid dimensions that disagree with a valid populated rectangular grid
are handled using the actual complete grid and recorded explicitly in source
metadata.

Every measured grid point yields one conditional-density observation:

\[
(c,d_A,d_B,I) \longmapsto (D_A,D_B).
\]

All observed alternatives and replicate measurements remain separate. The default
includes monotherapy axes and zero-dose controls; `include_zero_doses: false`
restricts the population to points with both doses positive. There is no inferred
isobole, nearest-response label, interpolated dose coordinate, response clipping or
averaging of response replicates. The fixed response-quality policy and its
inclusive bounds, action, reason and execution stage are part of the versioned
source manifest and source identity. Consequently the model estimates the dose
distribution under this experimental sampling scheme. It does not establish
that all points with high predicted density achieve an arbitrary requested
inhibition between measured observations.

DREAM cell names map to CMP models through exact, unique COSMIC identifiers.
Only released cell-name fields and configured aliases with `cell_id` and
`evidence` are eligible. `KMSII` is unresolved by default. A filename implying
`BT474` and an internal title `BT474C` is a conflict requiring explicit evidence;
the pipeline does not silently choose one label. A source matrix with an
unresolved or conflicting identity is excluded and recorded, or raises under the
configured error policy.

## Chemistry features and external identifier resolution

The baseline uses RDKit Morgan bit fingerprints: radius 2, 2,048 bits, chirality
disabled. The primary released chemistry file has precedence when its SMILES
parses, followed by the supplemental chemistry release. Semicolon-separated
alternatives are accepted only if their canonical isomeric structures are equal.
The feature definition, canonical structure, source field and software version
are recorded for each agent. No feature fitting uses response outcomes.

For 42 agents supplied as numeric PubChem identifiers, the default configuration
resolves the **exact released CID** through PubChem PUG REST. The cache preserves
the original response bytes, requested and returned CID, retrieval URL/time,
response hash and parsed canonical structure. Cache verification precedes reuse;
existing entries are not refreshed. HTTP failures exhaust bounded retries and
then fail without changing the experimental cohort. This is explicit external
identifier resolution, not a claim that all structures were contained directly
in the downloaded raw files. Reproduction of a recorded online-resolved experiment
requires its archived cache; future PubChem responses may change. Preserve
`dataset/processed/drug_features/pubchem/v1/` with the run.

`resolve_pubchem: false` defines a separate offline baseline using only parseable
structures in the supplied local files; numeric-only identifiers are then
excluded. Neither setting fabricates structures for the seven known exceptions:

| Challenge agent              | Released condition                    |
| ---------------------------- | ------------------------------------- |
| `ADAM17`                     | MEDI3622, biologic                    |
| `CD19`                       | MT-103/Blinatumomab, biologic         |
| `PDGFR_Ab`                   | MEDI-575, biologic                    |
| `TNFSF10`                    | TRAIL, protein ligand                 |
| `ATR_3`, `PIP5K1`, `SLC16A4` | No usable released structure/identity |

Affected response matrices are excluded by the default structure policy and
listed in the exclusion table. No surrogate small molecule is assigned.

## Cell expression and fitting populations

CMP `gene_id` is the stable feature key. The RNA-seq archive is streamed in full.
The offline source population is every model marked as having Broad or Sanger
**cell-line** RNA-seq in the released CMP availability catalog; organoid-only
profiles are not cell lines and are outside the proposal's input population. The
current catalog contains 1,643 such models. Models without any usable archive
measurement are excluded and recorded; every mapped response model is required
to have usable expression. For each model/gene, the raw feature is the arithmetic
mean of all finite, nonnegative `rsem_tpm` measurements across the available
datasets/sources. Extraction records the availability/model-list hashes,
source/dataset/duplicate counts, measurement counts and exclusions.

Leakage control operates on conservative CMP identity components, not model names
alone. Exact stable sample, patient, COSMIC, Broad/CCLE and RRID identifiers,
normalized model names, and declared parent relationships connect related model
records. Generic or missing identifiers never connect records. For the cold-start
protocol, response cells in one identity component are assigned as a group, so a
related cell cannot cross the final train/validation/test boundary.

The default cold-start split first sorts distinct retained cell identifiers,
permutes them with NumPy `default_rng(seed)`, and assigns approximately 15% to
validation, 15% to test and the remainder to training. Each holdout count uses
rounded group count with a minimum of one. Every dose, matrix, replicate and drug
pair belonging to a cell stays in that cell's split. Split memberships and exact
counts are saved; no observation-row splitting occurs.

The alternative `cell_pair` protocol applies the same seeded group allocation to
`(CMP identity component, unordered drug pair)`, keeping every related model,
matrix and replicate for the pair together. Cells shared with final training may
supply metadata to the VAE because the warm task explicitly observes those cells;
validation/test-only identity components cannot. The manifest records overlapping
cells. This protocol measures grouped warm completion and must be reported
separately from the cell-disjoint result.

After the response split is fixed, every expression identity component belonging
exclusively to final validation or test is removed from the complete offline VAE
population. This exclusion applies to VAE optimization, internal VAE validation,
gene selection, imputation and scaling. Every remaining external or final-training
cell-line profile is used in the offline stage. Related models stay together while
`default_rng(SeedSequence([split_seed, 1]))` assigns identity components to VAE
training or the internal metadata-only validation subset (default fraction 0.15).
At least two VAE-training identity components and one internal validation component
are required. **Only VAE-training models fit gene selection, imputation and
expression scaling.** Internal VAE-validation models never fit these transforms.
Final validation/test metadata are transformed only after fitting, exactly as an
unseen inference input would be. Manifests store all eligible, selected, validation
and evaluation-excluded model and identity lists so this boundary is auditable.

For raw TPM \(T*{cg}\), let \(L*{cg}=\log(1+T\_{cg})\). Compute each gene's mean
from observed VAE-training-model measurements and impute its missing log-expression
with that mean. Rank genes by population variance of the imputed VAE-training
matrix, breaking ties by ascending gene ID; discard unobserved and zero-variance
genes. Select up to `n_genes` (default 2,000) and standardize:

\[
X*{cg}=\frac{\widetilde L*{cg}-m*g}{s_g},\qquad
s_g^2=\frac{1}{|C*{\mathrm{VAE,train}}|}
\sum*{c\in C*{\mathrm{VAE,train}}}(\widetilde L\_{cg}-m_g)^2.
\]

The selected IDs, means, scales and exact fitting cell IDs are saved. Inference
accepts reordered genes but requires every selected identifier. Explicit missing
values use the saved mean; a cell with no measured selected genes fails. No
statistics are refitted on an unseen cell.

## Dose coordinates and density units

With a positive reference \(r\) in µM (default 1), transform each coordinate by

\[
Y_j=\frac{\log(1+D_j/r)-a}{b},\qquad
D_j=r\left[\exp(a+bY_j)-1\right].
\]

One shared scalar \(a\) and population standard deviation \(b\) are fitted over
**both coordinates of final training observations only**. This preserves the
meaning of zero and exchange of drug order. All transforms are saved and reused;
held-out dose ranges never choose scaling or default prediction-grid bounds.

The original-unit density uses the exact Jacobian:

\[
\log p*D(D_A,D_B)=\log p_Y(Y_A,Y_B)
-\sum*{j\in\{A,B\}}\bigl[\log b+\log(r+D_j)\bigr].
\]

The MDN is Gaussian in scaled log-dose coordinates, so inverse transformation
allows doses in \((-r,\infty)\). Negative candidates/samples remain explicit with
validity flags. Neither training nor inference clips them or silently truncates
the distribution. A nonnegative plotting rectangle contains only part of its
mass; density maps in µM⁻² are not renormalized to that rectangle. The reported
grid mass is numerical trapezoidal quadrature, not an exact integration.

## Architecture and objectives

The standalone VAE uses a diagonal-Gaussian posterior
\(q*\phi(z\mid X)=\mathcal N(\mu*\phi(X),\operatorname{diag}(\exp\ell\_\phi(X)))\),
a standard-normal prior and a mirrored dense decoder. Training uses one
reparameterized sample per cell and the batch-mean objective

\[
\mathcal L*{\mathrm{VAE}}=
\frac1B\sum_c\left[
\frac12\|\widehat X_c-X_c\|\_2^2
+\frac{\beta}{2}\sum_k
\left(\mu*{ck}^2+\exp\ell*{ck}-1-\ell*{ck}\right)\right],
\qquad\beta=1.
\]

This is the negative ELBO with a fixed unit-variance Gaussian reconstruction
likelihood, omitting its additive constant, when reconstruction is averaged over
posterior samples. The internal validation score uses reconstruction at the
posterior mean plus the same KL for deterministic checkpoint selection. That
selection surrogate is explicitly **not** a Monte Carlo estimate of the expected
training reconstruction term.

After selecting the best offline checkpoint, remove its decoder and freeze every
encoder parameter. Final-model compression uses the posterior mean. The trainable
cell and shared drug encoders use `Linear → ReLU → Linear`; the inhibition encoder
uses `Linear → ReLU`. With \(\oplus\) denoting concatenation and \(\odot\)
element-wise multiplication, implement the proposal's exact factorization:

\[
\begin{aligned}
U*1&=W*{C1}(E*c)\odot W*{X1}(E*I), &
U_2&=W*{C2}(E*c)\odot W*{X2}(E_I),\\
D_1&=W_D(E_A\oplus E_B), &D_2&=W_D(E_B\oplus E_A),\\
Z_1&=U_1\odot D_1+U_2\odot D_2, &
Z_2&=U_1\odot D_2+U_2\odot D_1.
\end{aligned}
\]

The same deep MLP processes `[Ec, EI, EA, EB]` and `[Ec, EI, EB, EA]` to obtain
\(H_A,H_B\). Concatenate `[Z1,Z2,HA,HB]` into the unrestricted MDN heads:

\[
p*Y(y\mid c,A,B,I)=\sum*{m=1}^M\pi*m\mathcal N_2(y;\mu_m,\Sigma_m),
\qquad
\Sigma_m=\begin{bmatrix}
\sigma*{mA}^2&\rho*m\sigma*{mA}\sigma*{mB}\\
\rho_m\sigma*{mA}\sigma*{mB}&\sigma*{mB}^2
\end{bmatrix}.
\]

Weights use softmax; means remain unrestricted; scales use softplus plus a
positive floor; correlations use tanh multiplied by a small interior margin to
avoid singular covariance. These are numerical safeguards, not alternate output
representations. The final training objective is exact bivariate mixture NLL:

\[
\mathcal L*{\mathrm{final}}=-\frac1B\sum_i
\log\sum_m\pi*{im}\mathcal N*2(y_i;\mu*{im},\Sigma\_{im}).
\]

Component log densities are combined with `logsumexp`; correlation arithmetic
uses a stable quadratic form. Mixture standard deviations and variances are
distinct quantities. There is no auxiliary efficacy, synergy, symmetry or
point-regression loss.

Both stages use Adam, gradient clipping, validation-driven learning-rate
reduction and early stopping. The final stage uses L2 weight decay of `1e-3`;
the offline VAE uses none. This is parameter regularization, not an auxiliary
prediction loss: the final data-fit term remains the exact MDN NLL above. Every
epoch records mean log scale, mean covariance log determinant, mixture entropy,
and mixture-mean RMSE alongside NLL so covariance collapse is auditable. Defaults
and all effective values are serialized in the configuration and checkpoints.
Select VAE weights by its internal validation surrogate and final-model weights
by final validation NLL. Test observations have no role in fitting, scheduling,
stopping, hyperparameter choice or selection.

## Drug ordering and evaluation

Exchanging drugs exchanges \(Z_1\leftrightarrow Z_2\) and
\(H_A\leftrightarrow H_B\) exactly. The proposal's unrestricted final linear MDN
heads do **not** mathematically guarantee an equivariant distribution. The
implementation preserves those heads and measures raw final-output swap
differences. It does not silently replace the head or enforce equality through a
new loss.

The public prediction interface uses the same canonical drug order as training
and restores coordinates to the caller's requested A/B order. This makes the
interface consistent for reversed requests; it does not make the raw MDN head
equivariant. Evaluation diagnostics call the raw model in both orders before
canonicalization to expose that distinction.

The checkpoint-selection metric remains validation mean NLL in scaled log-dose
coordinates. It is not the sole primary reporting result: evaluation promotes
macro-identity, inhibition-stratified, near-target, worst-group and retrospective
target-achievement summaries alongside it. Outputs include original-unit NLL
with the Jacobian; per-coordinate and pooled
MAE/RMSE in scaled and µM units; negative point-prediction fraction; and raw-head
swap diagnostics. The point summary is

\[
\widehat D=T^{-1}\!\left(\sum_m\pi_m\mu_m\right).
\]

This is the inverse of the mixture mean in transformed coordinates. It is not
the expectation in µM, is not generally a distribution mode, and need not lie on
a target-achieving isobole. Candidate coordinates are inverse-transformed
component means with the same limitations.

Per-cell, canonical drug-pair, matrix and inhibition-range tables preserve group
counts and metrics. Group macro NLL equally weights groups; overall NLL equally
weights observations. Pooled coordinate MAE averages the two absolute coordinate
errors per observation; pooled RMSE is the square root of the mean squared error
over observations and both coordinates. Swap metrics are mean coordinate
absolute difference after restoring the swapped output and absolute difference
in log density at correspondingly swapped observed targets.

The near-target summary selects **observed** responses satisfying
\(|I-0.5|\le0.05\) by default, with target/tolerance saved in the report. An empty
subset records zero count and null metrics. It does not relabel observations as
exactly 50% inhibition. Multi-seed and warm/cold experiments retain separate
per-run results and declared protocol/seed settings; no benchmark numbers are
implied by the proposal.

The versioned [evaluation assessment protocol](evaluation_assessment.md) defines
the additional estimands and artifacts. Recommendations at fixed inhibition 0.5
are frozen before revealing responses, snapped to the full original assay grid
using dose distance only, then scored without response-based reselection. A
missing or quality-excluded selected response remains unscorable. This tests the
discrete recommendation policy, not the biological response at an unmeasured
continuous dose. Original workbook hashes and the prepared split are verified.

Uncertainty resamples whole CMP identity components, retaining all their nested
pairs, matrices and rows. It does not treat dose rows as independent biological
samples. Intervals are conditional on the fixed checkpoint and split; cold-cell
intervals based on only 12 identities carry an explicit small-cluster warning.
The experimental design crosses independent split and training seeds and reports
their descriptive variability separately within each protocol. Neither those
repetitions nor warm-completion identities constitute additional cold-cell
biological replication. No test metric changes checkpoint selection, training
weights, the model architecture or the output distribution.

## Reproducibility, artifacts and continuation

Source and extracted-expression tables contain no learned preprocessing state.
Prepared bundles contain the source/expression lineage, exact split memberships,
fitting cell IDs, selected genes, transform parameters, feature definitions and
hashes. Every consumed raw/externally resolved source is recorded by the source
stage. Prepared bundles verify their manifest fingerprint and all artifact hashes
on load and remain portable when copied without the raw directories.

Source and prepared caches compare effective scientific configuration, consumed
source hashes, implementation/dependency identity and output hashes; expression
caches verify the raw archive, core mapped cohort and expression artifact hash.
An incompatible or incomplete cache fails clearly. No existing raw file is
rewritten. Preserve the configuration, `uv.lock`, code revision, downloaded raw
sources and any exact-CID response cache to reproduce the experiment.

Continuation checkpoints preserve model weights, optimizer and scheduler states,
epoch/step counters, active minibatch permutation and next offset, partial-epoch
totals, selection/early-stopping state, best weights, sampler generator state and
Python/NumPy/Torch RNG states, including available accelerator state. Atomic
replacement prevents partial checkpoint files from becoming authoritative.
Signals request saving at a completed minibatch boundary. Resume validates
dataset, configuration, encoder values, source code and runtime identity before
restoring an experiment; `best.pt` is deliberately selection-only.

CPU with the recorded runtime is the deterministic reference. CUDA/MPS are
configurable when available, with platform/operator determinism limitations;
cross-device bitwise equality is not promised. Persistent terminal logs and
structured events record starts, minibatches, training/validation epochs,
learning rates, gradients, interruptions, restoration and checkpoint selection.

The delivered tests check source axis/unit/identifier handling, train-only
transform fitting by perturbing held-out values, portable hash verification,
Gaussian density mathematics, branch exchange behavior, frozen weights,
inference and exact CPU interrupted-versus-uninterrupted training. A real-source
smoke run establishes integration, not research-level predictive performance.
