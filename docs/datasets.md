# Independent dataset workflows

Three independent response datasets are supported. No command pools response
tables across them. Each has its own source artifacts, splits, fitted expression
and dose transforms, VAE, final model, checkpoints and evaluation results.

| Dataset | Training configuration | Integration configuration |
| --- | --- | --- |
| AstraZeneca–DREAM | `configs/astra_zeneca.yaml` | `configs/astra_zeneca_smoke.yaml` |
| NCI-ALMANAC | `configs/nci_almanac.yaml` | `configs/nci_almanac_smoke.yaml` |
| ONeil-Merck | `configs/oneil.yaml` | `configs/oneil_smoke.yaml` |

`sources.sources` selects one native format: `[nci_almanac]`, `[oneil]`, or
`[dream]`/`[oi]` for the two releases within AstraZeneca–DREAM. Mixing ALMANAC,
O'Neil or DREAM sources in one configuration is rejected. Training also checks
that the configuration agrees with the prepared dataset's name and sources.

## Official raw files

Acquisition is an explicit step; preprocessing and training never download or
modify raw data:

```bash
uv run reverse-dose fetch-official --config configs/nci_almanac.yaml
uv run reverse-dose fetch-official --config configs/oneil.yaml
```

Files are stored under `dataset/raw/<dataset_name>/`. Existing files are never
replaced. `official_downloads.json` records each URL, SHA256, size, and publisher
MD5 where available. Repeat acquisition verifies the inventory offline. Corrupt
or incomplete references fail rather than silently changing the scientific cohort.

ALMANAC's original NCI release provides `ComboDrugGrowth_Nov2017.csv`,
`ComboCompoundSet.sdf`, `ComboCompoundNames_small.txt` and `ALMANAC_DataFields.txt`.
If the response CSV is missing, acquisition downloads the original response ZIP
and extracts its named CSV. An existing user-supplied CSV is retained; each
preprocessing run pins its actual bytes with SHA256.

O'Neil acquisition uses version 1 of the AACR Figshare records for combination
responses, single-agent responses, response legends and the cell-line table. It
verifies both the original paper's DOI in the publisher metadata and each file's
publisher checksum. The published combination file has an `.xls` suffix but is
an OOXML workbook; it is read by content through a binary file handle.

The O'Neil response supplement does not contain structures. Official NIH PubChem
exact-name responses, exact-CID structure records and validating synonyms are
archived under its raw `pubchem/` directory. Multiple hits are accepted only when
they have the identical organic stereochemical structure and charge, differing
solely in recognized inorganic counterions, and exactly one hit is the isolated
parent. This handles ABT-888's parent/dihydrochloride records without choosing
between different stereoisomers or organic compounds. All candidate records are
preserved. Other ambiguous hits fail. ALMANAC uses NCI's released SDF first; its
cisplatin entry fails strict RDKit valence checks, so a verified NIH record is
archived for the exact NCI-released cisplatin name. NCI's exact name list also
establishes the shared vemurafenib identity of NSC753082 and NSC761431.

Official sources:

- [NCI-ALMANAC release and documentation](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC)
- [AACR combination responses](https://aacr.figshare.com/articles/dataset/Supplementary_Data-Combination_Response_from_An_Unbiased_Oncology_Compound_Screen_to_Identify_Novel_Combination_Strategies/22503133/1)
- [AACR single-agent responses](https://aacr.figshare.com/articles/dataset/Supplementary_Data-Single_Agent_Response_from_An_Unbiased_Oncology_Compound_Screen_to_Identify_Novel_Combination_Strategies/22503127/1)
- [AACR response legends](https://aacr.figshare.com/articles/dataset/Supplementary_Data-Response_Legends_from_An_Unbiased_Oncology_Compound_Screen_to_Identify_Novel_Combination_Strategies/22503130/1)
- [AACR cell-line table](https://aacr.figshare.com/articles/figure/Supplemental_Table_1_Thirty-nine_cell_lines_used_in_combination_screen_from_An_Unbiased_Oncology_Compound_Screen_to_Identify_Novel_Combination_Strategies/22503139/1)
- [NIH PubChem](https://pubchem.ncbi.nlm.nih.gov/)

## Expression reference and exclusions

The new configurations explicitly set
`cell_reference_dataset: AstraZeneca–DREAM`. This points to the existing official
CMP snapshot under that directory's `cell_model_passports/` folder. It shares a
reference resource, not another dataset's drug-response measurements or fitted
features. Extracted raw expression is stored independently in each dataset's
processed directory. VAE fitting, gene selection, imputation and scaling are
performed independently for every declared split and run.

Cells are matched to CMP using exact normalized released model names, followed
by released synonyms, then unambiguous CCLE name prefixes. Canonical model names
take precedence over derivative records sharing a CCLE identifier. Related
identities still stay together during splitting through the existing CMP
identity-component rules. There is no fuzzy matching or automatic parent-cell
substitution. Explicit `sources.aliases` may map a source label to a CMP
`model_id` using the existing `{cell_id, evidence}` schema. Conflicting aliases
are rejected. Missing RNA-seq is excluded or raises according to
`unresolved_cell_policy`; it is never replaced by an unrelated profile.

The checked reference currently supports 54 ALMANAC cells and 34 O'Neil cells:

| Dataset | Excluded labels under the supplied configurations |
| --- | --- |
| ALMANAC: identity unresolved | `HL-60(TB)`, `MDA-MB-231/ATCC` |
| ALMANAC: no catalogued expression | `MDA-MB-435`, `NCI/ADR-RES`, `PC-3`, `SNB-19` |
| O'Neil: identity unresolved | `COLO320DM`, `LNCAP`, `MSTO`, `UWB1289BRCA1` |
| O'Neil: no catalogued expression | `EFM192B` |

These exclusions describe the pinned local reference, not an assertion that
expression can never be obtained elsewhere. Parent and derivative profiles are
not interchangeable. For example, [HL-60(TB)](https://www.cellosaurus.org/CVCL_A794)
and [UWB1.289+BRCA1](https://www.cellosaurus.org/CVCL_B078) have their own identities.
Coverage changes require new evidenced mappings or compatible official expression
records, followed by new source artifacts.

## Measured responses and experimental groups

O'Neil's `viability1`–`viability4` are fractional viability technical replicates.
Each finite measurement contributes `inhibition = 1 - viability`; replicates are
not averaged. Each batch/cell/drug-pair/technical-replicate combination is its own
matrix. All such matrices stay together under cell or cell/pair splitting.
`oneil_batches: [1, 2, 3]` preserves all released batches; the original legend
identifies batch 3 as a partial validation repeat. A primary-screen-only
sensitivity analysis can explicitly select `[1, 2]`. Literal `NULL` values are
excluded and counted; other malformed text fails. `mu/muMax` and `X/X0` are derived
growth endpoints and are not substituted for measured viability.

ALMANAC matrix identity includes screener, study, test date, the composite
`PANELNBR`/`CELLNBR` cell identifier, drug identifiers and sample identifiers.
Plate is not a matrix boundary: some 5×3 grids span several plates. Actual
concentrations define coordinates, rather than assuming concentration indices
always start at one. Molar doses are converted to µM. Record ID, source row,
original drug labels, original units and response origin are preserved.

The supplied ALMANAC configuration declares:

```yaml
sources:
  sources: [nci_almanac]
  nci_response: percent_control
  nci_missing_no_tz: reconstruct
```

It uses reported `PERCENTGROWTHNOTZ`. Where that field is missing, it explicitly
reconstructs percent-control response as `100 * TESTVALUE / CONTROLVALUE`, only
with finite test/control values and a positive control. It records the origin
for every row and never replaces it with time-zero-corrected `PERCENTGROWTH`.
Reconstruction inherits the precision and normalization of the released raw
measurements; it is not claimed to reproduce a missing reported value exactly.
`nci_missing_no_tz: exclude` defines a reduced cohort that drops the affected
laboratory; `error` refuses missing reported responses. A separately declared
`nci_response: percent_growth` experiment uses the original time-zero-corrected
growth endpoint. External evaluation rejects comparisons between that endpoint
and percent-control viability.

Both adapters retain the inclusive inhibition quality range `[-2, 2]`, without
clipping. Missing/QC-excluded points retain provenance in `exclusions.parquet`.
The supplied new configurations use measured combination points only. Standalone
single-agent data are retained as reference, not copied into every pair's grid.
There are no invented pair-specific monotherapy axes or zero-dose controls.
`include_zero_doses` applies only to actually released pair coordinates.

`matrix_design.parquet` preserves complete original axes before response QC or
missing-value exclusions, for every matrix with a retained measurement. It is
hashed, copied into prepared/exported bundles, and used for response-blind target
recommendations. A recommendation at a missing/excluded response is reported as
unscorable; assessment never selects an alternative after revealing the response.

## Inspect, prepare and train independently

```bash
uv run reverse-dose inspect --config configs/nci_almanac.yaml
uv run reverse-dose inspect --config configs/oneil.yaml
uv run reverse-dose prepare --config configs/nci_almanac.yaml
uv run reverse-dose prepare --config configs/oneil.yaml

uv run reverse-dose run --config configs/nci_almanac.yaml --run-dir runs/nci_almanac
uv run reverse-dose run --config configs/oneil.yaml --run-dir runs/oneil
```

`inspect` verifies/generates source artifacts and writes coverage, exclusions,
mapping evidence and effective configuration to the selected dataset's
`manifests/inspection.json`. The standalone `audit_dataset.py` remains the
DREAM-specific source auditor. Separate-stage training, resume, export,
prediction and factorial experiments use the same commands as DREAM, with the
appropriate configuration and prepared directory.

Source paths include configuration/schema identities and, for the new adapters,
all parsing/chemistry helper hashes. An implementation change selects fresh
default source artifacts. Explicit occupied output paths are never overwritten.
Prepared paths separate split seeds and feature settings. Every fitting state and
checkpoint retains its own dataset fingerprint.

## Cross-dataset evaluation without pooled training

After independent training, use a selected source checkpoint on another dataset:

```bash
NCI_DATA=$(uv run python -c 'import json; print(json.load(open("runs/nci_almanac/run.json"))["data_dir"])')
uv run reverse-dose evaluate-external \
  --training-data-dir "$NCI_DATA" \
  --checkpoint runs/nci_almanac/model/best.pt \
  --target-config configs/oneil.yaml \
  --output runs/nci_almanac/external/oneil
```

Repeat with the desired source checkpoint and target configuration for other
directions, including DREAM. The target's observations remain a separate table.
No target VAE, gene selection, imputation, standardization, dose scaling or
checkpoint selection is fitted. Target genes must match the archived expression
reference, and Morgan fingerprint definitions must agree. New target drug IDs
are supported through their validated fingerprints. Target configuration model
settings do not replace the source checkpoint's architecture or transforms.

The report and prediction rows identify cell exposure to source supervised
training, source VAE training and source checkpoint selection. Dataset separation
alone does not establish cell cold-start: the shared CMP pretraining population
can contain target cell identities. `--cell-scope all` reports the complete
supported external cohort. `unseen_supervised` excludes identities used for
supervised training; `unseen_fitting` also excludes identities used for VAE
training or either checkpoint-selection cohort. If none remain, evaluation fails
clearly rather than reporting an empty cohort as a result. These scopes do not
retroactively change the source training experiment.

## Validation of the current downloads

Full preprocessing and independent 2,000-gene, cell-cold preparation were checked:

| Dataset | Retained measurements | Cells | Drugs | Matrices |
| --- | ---: | ---: | ---: | ---: |
| NCI-ALMANAC | 2,584,890 | 54 | 104 | 280,300 |
| ONeil-Merck | 1,280,859 | 34 | 38 | 80,107 |

The ALMANAC retained cohort contains 155,475 explicitly reconstructed no-TZ
responses; O'Neil counts individual technical-replicate measurements. These
counts are ingestion checks, not trained-model performance claims. Use the smoke
configurations for short end-to-end execution checks; use the full configurations
for scientific experiments.
