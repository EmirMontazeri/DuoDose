# Proposal V2 Dataset

This dataset contains the data required to train and evaluate Proposal V2 for inverse cancer drug-combination dose prediction.

It includes:

- AZ-Sanger DREAM drug-combination dose-response matrices and drug chemistry.
- AZ-Sanger cell-line metadata and molecular data.
- Cell Model Passports RNA-seq and model metadata for offline cell-metadata VAE pretraining.

## Sources

- **AstraZeneca-Sanger DREAM Challenge:** downloaded from Synapse, project `syn4231880`.
- **Cell Model Passports:** downloaded from the Wellcome Sanger Institute at `cellmodelpassports.sanger.ac.uk/downloads`.

Files under `raw/` are preserved in their original downloaded form. Derived V2-ready features, dose-response tables, and dataset splits are stored under `processed/` and `splits/`.
