# Proposal V2 Dataset

This dataset contains the data required to train and evaluate Proposal V2 for inverse cancer drug-combination dose prediction.

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

Files under `raw/` are preserved in their original downloaded form. Derived V2-ready features, dose-response tables, and dataset splits are stored under `processed/` and `splits/`.
