"""Target-only evaluation using the selected source model and frozen transforms."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, config_from_dict
from .data import DataBundle, DoseDataset, extract_expression
from .evaluation import evaluate, load_trained_model
from .preprocess import preprocess_sources, response_quality_control
from .utils import (
    atomic_json,
    environment_info,
    fingerprint,
    guard_output,
    load_checkpoint,
    read_json,
    sha256_file,
)


def _verified(directory: Path, name: str, key: str, required: set[str]) -> dict[str, Any]:
    manifest = read_json(directory / name)
    hashes = manifest.get(key)
    if not isinstance(hashes, dict) or not required <= set(hashes):
        raise ValueError(f"Missing artifact inventory in {name}")
    for relative, digest in hashes.items():
        path = directory / relative
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Target artifact integrity mismatch: {path}")
    return manifest


class ExternalBundle:
    """A target-only tensor view. No statistic is fitted to target measurements."""

    def __init__(
        self,
        training: DataBundle,
        source: Path,
        expression: Path,
        checkpoint: Path,
        cell_scope: str = "all",
    ) -> None:
        native = _verified(
            source,
            "source_manifest.json",
            "output_hashes",
            {"observations.parquet", "cells.parquet", "drugs.npz"},
        )
        expr = _verified(expression, "expression_manifest.json", "artifacts", {"expression.npz"})
        if native.get("response_quality_control") != response_quality_control():
            raise ValueError("Target source lacks the required response quality policy")
        original = read_json(training.path / "source_manifest.json")

        def endpoint(manifest: dict[str, Any]) -> str:
            settings = manifest.get("config", {})
            return (
                "time_zero_growth"
                if settings.get("sources") == ["nci_almanac"]
                and settings.get("nci_response") == "percent_growth"
                else "percent_control_viability"
            )

        if endpoint(original) != endpoint(native):
            raise ValueError(
                "Training and target response endpoints differ; use percent_control for cross-dataset comparisons"
            )
        for name in ("algorithm", "bits", "radius", "includeChirality"):
            if original["fingerprint_definition"].get(name) != native["fingerprint_definition"].get(
                name
            ):
                raise ValueError(f"Target and training drug fingerprint definitions differ: {name}")
        original_expr = read_json(training.path / "expression_manifest.json")
        # Provider-specific gene IDs require a shared reference, not just equal strings.
        for name in ("gene_key", "source_sha256", "catalog_hashes", "population_definition"):
            if name not in original_expr or original_expr[name] != expr.get(name):
                raise ValueError(
                    f"Target expression reference is not compatible with training: {name}"
                )
        self.path = source
        self.expression_transform, self.dose_transform = (
            training.expression_transform,
            training.dose_transform,
        )
        self.observations = pd.read_parquet(source / "observations.parquet").reset_index(drop=True)
        cells = pd.read_parquet(source / "cells.parquet")
        if cells.cell_id.duplicated().any() or cells.model_id.duplicated().any():
            raise ValueError("Target cells must map uniquely to CMP models")
        for column in ("cell_id", "drug_a", "drug_b", "matrix_id", "observation_id"):
            self.observations[column] = self.observations[column].astype(str)
        if self.observations.observation_id.duplicated().any() or not set(
            self.observations.cell_id
        ) <= set(cells.cell_id):
            raise ValueError("Target observation identifiers/cell mappings are invalid")
        numeric = self.observations[["dose_a_um", "dose_b_um", "inhibition"]].to_numpy(dtype=float)
        if (
            not np.isfinite(numeric).all()
            or (numeric[:, :2] < 0).any()
            or (numeric[:, 2] < -2).any()
            or (numeric[:, 2] > 2).any()
        ):
            raise ValueError("Target observations violate dose/response quality requirements")
        with np.load(expression / "expression.npz", allow_pickle=False) as a:
            models, genes, groups = (
                a["model_ids"].tolist(),
                a["gene_ids"].tolist(),
                a["identity_group_ids"].tolist(),
            )
            if (
                len(set(models)) != len(models)
                or len(set(genes)) != len(genes)
                or len(groups) != len(models)
            ):
                raise ValueError("Target expression IDs must be unique and aligned")
            lookup = {model: i for i, model in enumerate(models)}
            if not set(cells.model_id) <= set(lookup):
                raise ValueError("Target cells lack required expression measurements")
            cells["identity_group"] = [groups[lookup[model]] for model in cells.model_id]
            raw = a["values"][[lookup[model] for model in cells.model_id]]
        cell_identity = dict(zip(cells.cell_id, cells.identity_group, strict=True))
        source_identity = dict(
            zip(training.cell_ids, training.cell_identity_group_ids, strict=True)
        )
        supervised = {
            source_identity[c]
            for c in training.observations.loc[training.observations.split.eq("train"), "cell_id"]
        }
        selection = {
            source_identity[c]
            for c in training.observations.loc[
                training.observations.split.eq("validation"), "cell_id"
            ]
        }
        vae_train = {training.vae_identity_group_ids[i] for i in training.vae_train_indices}
        vae_select = {training.vae_identity_group_ids[i] for i in training.vae_validation_indices}
        identity = self.observations.cell_id.map(cell_identity)
        self.observations["seen_in_supervised_training"] = identity.isin(supervised)
        self.observations["seen_in_vae_training"] = identity.isin(vae_train)
        self.observations["seen_in_checkpoint_selection"] = identity.isin(selection | vae_select)
        if cell_scope == "unseen_supervised":
            keep = ~identity.isin(supervised)
        elif cell_scope == "unseen_fitting":
            keep = ~identity.isin(supervised | selection | vae_train | vae_select)
        elif cell_scope == "all":
            keep = pd.Series(True, index=self.observations.index)
        else:
            raise ValueError("Unknown external cell scope")
        excluded_rows = int((~keep).sum())
        self.observations = self.observations.loc[keep].reset_index(drop=True)
        if self.observations.empty:
            raise ValueError(
                "No target observations remain under the declared identity-exposure scope"
            )
        self.observations["split"] = "test"
        transformed = self.expression_transform.transform(raw, genes)
        self.cell_ids = sorted(set(self.observations.cell_id))
        indices = {cell: i for i, cell in enumerate(cells.cell_id)}
        self.cell_features = transformed[[indices[c] for c in self.cell_ids]].astype(np.float32)
        self.cells = cells.set_index("cell_id").loc[self.cell_ids].reset_index()
        self.cell_identity_group_ids = [cell_identity[c] for c in self.cell_ids]
        with np.load(source / "drugs.npz", allow_pickle=False) as a:
            self.drug_ids, self.drug_features = a["ids"].tolist(), a["features"].astype(np.float32)
        if (
            len(set(self.drug_ids)) != len(self.drug_ids)
            or self.drug_features.shape != (len(self.drug_ids), training.drug_features.shape[1])
            or not np.isfinite(self.drug_features).all()
        ):
            raise ValueError(
                "Target drug features do not match the trained model's feature contract"
            )
        cell_lookup, drug_lookup = {c: i for i, c in enumerate(self.cell_ids)}, {
            d: i for i, d in enumerate(self.drug_ids)
        }
        if not (set(self.observations.drug_a) | set(self.observations.drug_b)) <= set(drug_lookup):
            raise ValueError("Target observations reference missing drug features")
        self.targets = self.dose_transform.transform(
            self.observations[["dose_a_um", "dose_b_um"]].to_numpy()
        ).astype(np.float32)
        self._cell_indices = self.observations.cell_id.map(cell_lookup).to_numpy(dtype=np.int64)
        self._drug_a_indices = self.observations.drug_a.map(drug_lookup).to_numpy(dtype=np.int64)
        self._drug_b_indices = self.observations.drug_b.map(drug_lookup).to_numpy(dtype=np.int64)
        self._inhibition = self.observations.inhibition.to_numpy(dtype=np.float32).reshape(-1, 1)
        exposure = self.observations.drop_duplicates("cell_id")
        self.protocol = {
            "response_endpoint": endpoint(native),
            "training_dataset_fingerprint": training.manifest["fingerprint"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "target_source_manifest_sha256": sha256_file(source / "source_manifest.json"),
            "target_expression_manifest_sha256": sha256_file(
                expression / "expression_manifest.json"
            ),
            "cell_scope": cell_scope,
            "excluded_rows_by_scope": excluded_rows,
            "target_cells": len(self.cell_ids),
            "target_cells_seen_in_supervised_training": int(
                exposure.seen_in_supervised_training.sum()
            ),
            "target_cells_seen_in_vae_training": int(exposure.seen_in_vae_training.sum()),
            "target_cells_seen_in_checkpoint_selection": int(
                exposure.seen_in_checkpoint_selection.sum()
            ),
            "feature_policy": "Reuse selected source model, VAE, genes, imputation, expression scaling, Morgan definition and dose transform; no target fitting or checkpoint selection",
        }
        self.manifest = {
            "fingerprint": fingerprint(self.protocol),
            "external_protocol": self.protocol,
            "split": {"configuration": {"protocol": "external"}},
            "artifacts": (
                {"matrix_design.parquet": native["output_hashes"]["matrix_design.parquet"]}
                if "matrix_design.parquet" in native["output_hashes"]
                else {}
            ),
        }

    def dataset(self, split: str) -> DoseDataset:
        return DoseDataset(self, split)


def evaluate_external(
    training_data: Path,
    checkpoint: Path,
    target_config: ExperimentConfig,
    output: Path,
    *,
    source_dir: Path | None = None,
    expression_dir: Path | None = None,
    cell_scope: str = "all",
    device: str | None = None,
) -> dict[str, Any]:
    """Prepare target measurements independently and evaluate a fixed source model."""
    from .cli import artifact_paths

    saved = load_checkpoint(checkpoint)
    if not saved.get("selection_only", False):
        raise ValueError("External evaluation requires a selected best.pt model checkpoint")
    config = config_from_dict(saved["config"])
    guard_output(output, Path(config.dataset_root))
    guard_output(output, Path(target_config.dataset_root))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"External evaluation output already exists: {output}")
    training = DataBundle(training_data)
    load_trained_model(training, config, checkpoint, device or config.device)
    defaults = artifact_paths(target_config)
    source_dir, expression_dir = source_dir or defaults[0], expression_dir or defaults[1]
    preprocess_sources(
        Path(target_config.dataset_root),
        asdict(target_config.sources),
        source_dir,
        dataset_name=target_config.dataset_name,
        reference_dataset_name=target_config.cell_reference_dataset,
    )
    extract_expression(
        Path(target_config.dataset_root),
        pd.read_parquet(source_dir / "cells.parquet"),
        expression_dir,
        dataset_name=target_config.dataset_name,
        reference_dataset_name=target_config.cell_reference_dataset,
    )
    target = ExternalBundle(training, source_dir, expression_dir, checkpoint, cell_scope)
    config.device = device or config.device
    config.dataset_root, config.dataset_name = (
        target_config.dataset_root,
        target_config.dataset_name,
    )
    report = evaluate(target, config, checkpoint, output, "test", reference_bundle=training)
    report["external_protocol"]["environment"] = environment_info()
    atomic_json(output / "metrics.json", report)
    return report
