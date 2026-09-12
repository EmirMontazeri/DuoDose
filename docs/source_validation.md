# Complete source validation

Both corrected-matrix releases were fully parsed with source schema
`source-v2.4`; validation was not limited to the original audit's sampled files.
The runs used the default source policies, exact released PubChem identifier
resolution, zero-dose inclusion, and no configured cell aliases. DREAM and OI
were processed independently. These are ingestion results, not trained-model
performance measurements.

## Generated cohorts

| Measurement                                                       |   DREAM |      OI |
| ----------------------------------------------------------------- | ------: | ------: |
| Corrected XLSX matrices parsed                                    |  11,608 |  20,489 |
| Matrices with retained measurements                               |   9,931 |  11,185 |
| Retained dose-response observations                               | 357,462 | 402,427 |
| DREAM metadata cells mapped by exact COSMIC ID                    |      85 |      85 |
| Cells represented in retained observations                        |      80 |      80 |
| Drugs represented in retained observations                        |     110 |     110 |
| Released agents with supported fingerprints                       |     112 |     112 |
| Quality outliers removed before splitting                         |      54 |     233 |
| Retained observations outside biological interval `[0,1]`         |  49,645 |  69,114 |
| Workbooks with incorrect `A1` declared dimensions read completely |  11,608 |  20,489 |

The default DREAM artifacts are in
`dataset/processed/dose_response/v2_2a3cd80276be/`, with identity
`e44354cd782a96b7113f52d4c029a86ce5bedda898b73ff5e11bef0339260d24`.
The separately configured OI artifacts are generated in
`dataset/processed/dose_response/v2_18a34f622cc1/`, with identity
`776999b0ac164291b6431cec65226979972ee4c63fcd0516002cda1af7df8eb2`.
Exact source identities are recorded in each generated manifest and include
cached CID retrieval provenance.
These paths are computed from the source configuration and schema version; use
the CLI to obtain the correct path after changing a source policy.

Reproduce source preparation through the documented CLI:

```bash
.venv/bin/reverse-dose preprocess --config configs/v2.yaml
.venv/bin/reverse-dose preprocess --config configs/oi.yaml
```

Standalone preparation writes persistent logs to
`dataset/manifests/v2_logs/pipeline.log`.

## Exclusions and identifier evidence

| Exclusion reason                                     | DREAM rows | OI rows |
| ---------------------------------------------------- | ---------: | ------: |
| Unsupported drug structure, per matrix               |      1,102 |   1,925 |
| Unresolved cell label, per matrix                    |        634 |   8,151 |
| Conflicting filename/internal cell label, per matrix |        615 |     966 |
| Missing response, per grid point                     |          0 |     252 |
| Entire grid has no measured responses, per matrix    |          0 |       7 |
| Inhibition outside inclusive `[-2.0,2.0]`, per point |         54 |     233 |

Reasons overlap: one matrix may have an unsupported drug and an unresolved cell,
or both an unresolved label and a label mismatch. Consequently, summing reason
counts does not give the number of distinct excluded matrices. Every reason is
recorded with its source filename in `exclusions.parquet`; point exclusions also
carry the original worksheet coordinates. Rows excluded by inhibition quality
control additionally retain the original `response_percent` and the unmodified
computed `inhibition`. The retained inhibition range is `[-2.0,1.873]` in both
cohorts; excluded ranges are `[-2.505,41.965]` for DREAM and
`[-6.629,44.513]` for OI.

The complete DREAM cell-label discrepancies are:

| Internal title | Filename cell label | Matrices |
| -------------- | ------------------- | -------: |
| BT474C         | BT474               |      435 |
| MDAMB175V11    | MDAMB175VII         |       88 |
| H747           | NCIH747             |       47 |
| PTEN22RV1      | 22RV1               |       26 |
| KMSII          | KMSII               |       19 |
| MCF7NORMAL     | MCF7                |       14 |
| PTENVCAP       | VCAP                |        5 |

No identities are inferred from these strings. A configured alias requires an
existing canonical `cell_id` and explicit evidence; its provenance enters the
source identity. OI contains 76 distinct unresolved internal titles, including
additional models and experimental derivatives. The current OI baseline includes
only cells mapped through the released DREAM metadata and its exact COSMIC link
to Cell Model Passports. It does not infer mappings for those additional OI
models or merge derivative labels into parental cells. Inspect `exclusions.parquet`
for the complete OI list.

The exact mapped DREAM cells absent from both retained source cohorts are
`22RV1`, `BT-474`, `KMS-11`, `M14`, and `MDA-MB-175-VII`. Their RNA-seq mappings
remain available in `cells.parquet`, but they do not supply response examples.
The first, second, third and fifth cells have unresolved response labels listed
above; neither release has a corrected matrix whose filename cell is `M14`.
Do not interpret these omissions as expression-data failures.

One drug spelling discrepancy has direct source evidence:
`Gemcitibine` maps to `Gemcitabine` because the supplemental
`Drug_Chemistry_Data` sheet explicitly lists
`Gemcitabine, Gemcitibine, Gemzar` in its `Drug name` field. The exact released
token is checked before this alias is used, and `drug_alias_evidence` records the
source sheet, column and value. DREAM retains 6,696 observations carrying this
original internal spelling. Generic comma-separated drug names are not treated
as aliases because some fields describe cocktail constituents.

## Completely missing OI grids

Seven OI workbooks contain valid axes, units and metadata but all 36 response
cells are blank:

```text
CHEK1&BCL2_BCL2L1_BCL2L2&ZR751&Exp1.xlsx
CHEK1&MET&ZR751&Exp1.xlsx
CHEK1&Proteasome&ZR751&Exp1.xlsx
ERBB&FNTA_FNTB&CAMA1&Exp1.xlsx
ERBB&HDAC1_HDAC3&CAMA1&Exp1.xlsx
ERBB&HDAC_4&CAMA1&Exp1.xlsx
ERBB&PI3KCD&CAMA1&Exp1.xlsx
```

They are under `dataset/raw/az_dream/DREAM_OI_matrices_corrected/`. With
`missing_response_policy: exclude`, preprocessing records all 252 missing point
coordinates and seven `no_measured_responses` matrix exclusions. It does not
construct response labels or impute dose-response measurements. With
`missing_response_policy: error`, the same sources produce an explicit failure.
Malformed numeric values, units and grid schemas continue to fail regardless of
the missing-response policy.

## Chemistry and offline reproducibility

The released chemistry tables contain 119 agents. Seventy have usable local
SMILES after primary/supplemental precedence: 62 single representations and eight
semicolon-delimited alternatives whose canonical isomeric SMILES are verified
equal. Another 42 entries have exact released numeric PubChem identifiers.
Their responses were retrieved through the PubChem exact-CID property endpoint;
the requested and returned CIDs, RDKit-valid SMILES, response bytes, retrieval URL
and SHA256 are preserved under
`dataset/processed/drug_features/pubchem/v1/<CID>/`. The resolver does not search
names or select similar compounds. The
[official PUG REST specification](https://pubchem.ncbi.nlm.nih.gov/pcfe/docs/markdown/pug-rest.md)
documents this exact-CID property interface.

The seven known gaps remain excluded: biologics `ADAM17`, `CD19`, `PDGFR_Ab`,
`TNFSF10`, and unreleased challenge structures `ATR_3`, `PIP5K1`, `SLC16A4`.
`IAP_2` and `MAP2K_2` have fingerprints but occur in neither retained response
cohort. Setting `resolve_pubchem: false` selects an explicitly reduced cohort
using locally released SMILES only; this is a different source configuration.

Verified CID cache entries are never silently refreshed. Preserve them with
experiment artifacts to reproduce the exact resolved structures offline. A
fresh uncached resolution depends on the external database response at that
retrieval time; the recorded cache makes that dependency explicit. A changed or
corrupted cache fails validation rather than silently changing the cohort.

## Integrity and regression checks

The source manifest records SHA256 and size for every consumed raw file,
separately records the consumed PubChem response/manifest hashes, and includes
the complete effective source configuration, source inventories, preprocessing
code hash and dependency versions. Every generated table and fingerprint array
has an output SHA256. The manifest itself has a reproducible
`source_manifest.sha256` companion file.

Cache reuse verifies the manifest checksum, recomputes the source identity,
compares every identity field, requires the complete expected five-artifact hash
inventory, and verifies every output hash. Incomplete or mismatched artifacts
are rejected. The atomic directory commit prevents successful-looking partial
outputs. Generation timestamps are excluded from the scientific manifest so
identical source/cache/code/config inputs regenerate identical manifest bytes.
PubChem retrieval timestamps remain in the immutable cached retrieval provenance.

Source schema v2.4 makes the inclusive bounds, remove-without-clipping action,
exclusion reason and pre-split execution stage part of the hashed source identity.
It also records the observation and exclusion schemas, exact reason counts, raw
excluded responses and computed inhibitions. Downstream preparation verifies both
the declared policy and every retained inhibition value before assigning a split;
a pre-v2.4 or internally inconsistent source artifact is rejected.

The preprocessing test module covers axis/drug orientation,
units, malformed and missing numeric values, completely empty OI responses,
aliases, immutable raw data, exact CID validation, offline cache reuse,
manifest/output corruption, inclusive quality boundaries, raw-value-preserving
quality exclusions, exclusions and byte-identical regeneration. Ruff passes for
the preprocessing module and its tests.
