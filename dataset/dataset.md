# Datasets

Dataset storage is grouped first by artifact type, then by dataset name:

```text
dataset/
├── dataset.md
├── raw/
│   ├── AstraZeneca–DREAM/
│   │   ├── az_dream/
│   │   ├── az_dream_molecular/
│   │   └── cell_model_passports/
│   ├── NCI-ALMANAC/
│   └── ONeil-Merck/
├── processed/
│   ├── AstraZeneca–DREAM/
│   │   ├── dose_response/
│   │   ├── cell_expression/
│   │   └── drug_features/
│   ├── NCI-ALMANAC/
│   └── ONeil-Merck/
├── splits/
│   ├── AstraZeneca–DREAM/
│   ├── NCI-ALMANAC/
│   └── ONeil-Merck/
└── manifests/
    ├── AstraZeneca–DREAM/
    ├── NCI-ALMANAC/
    └── ONeil-Merck/
```

`dataset_root: dataset` selects the shared storage root. `dataset_name:
AstraZeneca–DREAM` selects the subdirectory in each artifact type; the name uses
an en dash (`–`). Additional datasets should use sibling directories under all
four artifact types and their own configurations. Source readers currently
support DREAM/OI, NCI-ALMANAC and O'Neil–Merck release formats. The latter two use
separate configurations and never pool response measurements with DREAM or each
other. See [independent workflows](../docs/datasets.md) for official downloads,
coverage, endpoint definitions, exclusions and cross-dataset evaluation.

## AstraZeneca–DREAM

This dataset supports training and evaluation of inverse cancer drug-combination
dose prediction.

It includes:

- AZ-Sanger DREAM drug-combination dose-response matrices and drug chemistry.
- AZ-Sanger cell-line metadata and molecular data.
- Cell Model Passports RNA-seq and model metadata for offline cell-metadata VAE pretraining.

The offline VAE extraction uses every model marked with Broad or Sanger
cell-line RNA-seq in the released availability catalog. Dataset preparation
groups related CMP identities and excludes all final-validation/test-only
identity components before VAE preprocessing, training, or internal selection.
The response validation/test expression vectors are transformed only with state
fitted on VAE-training models.

Dose-response source preprocessing computes
`inhibition = 1 - response_percent / 100` and, before constructing any dataset
split, retains only observations in the inclusive range `[-2.0, 2.0]`. Values
outside that range are removed without clipping. Their original response and
computed inhibition remain in the processed `exclusions.parquet` table with
reason `response_outside_quality_range` and source matrix/grid provenance.

## Sources

- **AstraZeneca-Sanger DREAM Challenge:** downloaded from Synapse, project `syn4231880`.
- **Cell Model Passports:** downloaded from the Wellcome Sanger Institute at `cellmodelpassports.sanger.ac.uk/downloads`.

Files under `raw/AstraZeneca–DREAM/` are preserved in their original downloaded
form. Derived features and dose-response tables are stored under
`processed/AstraZeneca–DREAM/`, splits under `splits/AstraZeneca–DREAM/`, and audit,
inspection and preprocessing logs under `manifests/AstraZeneca–DREAM/`.

Existing artifacts were moved without editing their contents. Older hashed
manifests can therefore retain the previous source paths; retrospective matrix
assessment resolves those DREAM paths under the new raw directory and still
verifies the original file hashes. New preprocessing records the new paths.
Use newly generated default artifact directories for new experiments: changed
code and provenance intentionally invalidate reuse of old preprocessing caches
and checkpoint resumption.
