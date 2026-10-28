"""Held-out distribution evaluation, grouped diagnostics and reproducible tables.

Validation NLL selects checkpoints; primary reporting also covers biological
groups, inhibition strata, target achievement and calibration. Original-unit
densities include the Jacobian. Coordinate errors alone are not target evidence.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .assessment import target_window_mask
from .config import ExperimentConfig
from .data import DataBundle
from .layout import DatasetPaths
from .models import mixture_log_prob
from .training import make_model
from .utils import (
    atomic_json,
    guard_output,
    load_checkpoint,
    resolve_device,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)


def load_trained_model(
    bundle: DataBundle,
    config: ExperimentConfig,
    checkpoint_path: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    """Restore an inference model only after verifying feature/architecture identity."""
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "model":
        raise ValueError("Evaluation requires a final-model checkpoint")
    if checkpoint["dataset_fingerprint"] != bundle.manifest["fingerprint"]:
        raise ValueError("Checkpoint and prepared feature/split fingerprint differ")
    from .utils import fingerprint

    for key in ("vae", "model"):
        if fingerprint(checkpoint["config"][key]) != fingerprint(config.to_dict()[key]):
            raise ValueError(f"Checkpoint and {key} configuration differ")
    model = make_model(bundle, config, "model")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(resolve_device(device)).eval()


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize per-observation metrics with an explicit empty-cohort result."""
    if frame.empty:
        return {
            "count": 0,
            "nll": None,
            "nll_um": None,
            "mae_scaled": None,
            "rmse_scaled": None,
            "mae_um": None,
            "rmse_um": None,
            "invalid_point_fraction": None,
            "swap_mean_l1_scaled": None,
            "swap_abs_log_density_difference": None,
        }
    result: dict[str, Any] = {
        "count": len(frame),
        "cells": frame.cell_id.nunique(),
        "pairs": frame[["drug_a", "drug_b"]].drop_duplicates().shape[0],
    }
    for name in (
        "nll",
        "nll_um",
        "swap_mean_l1_scaled",
        "swap_abs_log_density_difference",
        "invalid_point_fraction",
    ):
        result[name] = float(frame[name].mean())
    for domain in ("scaled", "um"):
        result[f"mae_{domain}"] = float(frame[f"absolute_error_{domain}"].mean())
        result[f"rmse_{domain}"] = math.sqrt(float(frame[f"squared_error_{domain}"].mean()))
        for coordinate in ("a", "b"):
            result[f"mae_{coordinate}_{domain}"] = float(
                frame[f"absolute_error_{coordinate}_{domain}"].mean()
            )
            result[f"rmse_{coordinate}_{domain}"] = math.sqrt(
                float(frame[f"squared_error_{coordinate}_{domain}"].mean())
            )
    return result


@torch.no_grad()
def evaluate(
    bundle: DataBundle,
    config: ExperimentConfig,
    checkpoint: Path,
    output_dir: Path,
    split: str = "test",
    *,
    reference_bundle: DataBundle | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed checkpoint and persist individual and grouped results."""
    if split not in {"validation", "test", "train"}:
        raise ValueError("Evaluation split must be train, validation, or test")
    if (
        reference_bundle is None
        and bundle.manifest.get("dataset_name") is not None
        and bundle.manifest["dataset_name"] != config.dataset_name
    ):
        raise ValueError("Evaluation configuration and prepared dataset names differ")
    guard_output(output_dir, Path(config.dataset_root))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    if reference_bundle is not None:
        if (
            bundle.manifest.get("external_protocol", {}).get("training_dataset_fingerprint")
            != reference_bundle.manifest["fingerprint"]
        ):
            raise ValueError("External evaluation does not declare the verified training dataset")
    model = load_trained_model(reference_bundle or bundle, config, checkpoint, config.device)
    dataset = bundle.dataset(split)
    if len(dataset) == 0:
        raise ValueError(f"Empty evaluation split: {split}")
    loader = DataLoader(
        dataset, batch_size=config.evaluation.batch_size, shuffle=False, num_workers=0
    )
    frames = []
    parameter_rows: dict[str, list[np.ndarray]] = {
        key: [] for key in ("weights", "means", "scales", "correlations", "indices")
    }
    LOGGER.info("Evaluating %d %s observations using %s", len(dataset), split, checkpoint)
    for number, raw in enumerate(loader):
        indices = raw["index"].numpy()
        batch = {key: value.to(device) for key, value in raw.items()}
        parameters = model(batch["cell"], batch["inhibition"], batch["drug_a"], batch["drug_b"])
        swapped = model(batch["cell"], batch["inhibition"], batch["drug_b"], batch["drug_a"])
        log_prob = mixture_log_prob(parameters, batch["target"])
        swap_log_prob = mixture_log_prob(swapped, batch["target"].flip(-1))
        mean = parameters.mixture_mean.cpu().numpy().astype(np.float64)
        target = batch["target"].cpu().numpy().astype(np.float64)
        point_um = bundle.dose_transform.inverse(mean)
        frame = bundle.observations.iloc[indices].copy()
        actual_um = frame[["dose_a_um", "dose_b_um"]].to_numpy(dtype=np.float64)
        jacobian = bundle.dose_transform.log_abs_det_jacobian(actual_um)
        if np.ndim(jacobian) == 2:
            jacobian = jacobian.sum(axis=-1)
        frame["nll"] = -log_prob.cpu().numpy()
        frame["nll_um"] = frame["nll"].to_numpy() - jacobian
        frame["swap_mean_l1_scaled"] = (
            (parameters.mixture_mean - swapped.mixture_mean.flip(-1)).abs().mean(-1).cpu().numpy()
        )
        frame["swap_abs_log_density_difference"] = (log_prob - swap_log_prob).abs().cpu().numpy()
        frame["invalid_point_fraction"] = (point_um < 0).any(axis=1).astype(float)
        for domain, predicted, measured in (
            ("scaled", mean, target),
            ("um", point_um, actual_um),
        ):
            difference = predicted - measured
            frame[f"absolute_error_{domain}"] = np.abs(difference).mean(axis=1)
            frame[f"squared_error_{domain}"] = np.square(difference).mean(axis=1)
            for j, coordinate in enumerate(("a", "b")):
                frame[f"predicted_{coordinate}_{domain}"] = predicted[:, j]
                frame[f"absolute_error_{coordinate}_{domain}"] = np.abs(difference[:, j])
                frame[f"squared_error_{coordinate}_{domain}"] = np.square(difference[:, j])
        if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
            raise FloatingPointError(
                "Non-finite evaluation metric; inspect covariance/dose scaling"
            )
        frames.append(frame)
        for key in ("weights", "means", "scales", "correlations"):
            parameter_rows[key].append(getattr(parameters, key).cpu().numpy())
        parameter_rows["indices"].append(indices)
        if number % 25 == 0:
            LOGGER.info("Evaluation batch %d/%d", number + 1, len(loader))
    results = pd.concat(frames, ignore_index=True)
    arrays = {key: np.concatenate(value) for key, value in parameter_rows.items()}
    results.to_parquet(output_dir / "predictions.parquet", index=False)
    np.savez_compressed(
        output_dir / "mixture_parameters.npz",
        **arrays,
    )
    near_target = results.loc[target_window_mask(results.inhibition, config.evaluation)]
    groups = {}
    for label, columns in {
        "cell": ["cell_id"],
        "drug_pair": ["drug_a", "drug_b"],
        "matrix": ["matrix_id"],
    }.items():
        records = []
        for key, frame in results.groupby(columns, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            records.append({**dict(zip(columns, values)), **metric_summary(frame)})
        table = pd.DataFrame(records)
        table.to_csv(output_dir / f"by_{label}.csv", index=False)
        groups[f"macro_{label}_nll"] = float(table.nll.mean())
    bins = [-np.inf, *config.evaluation.inhibition_bin_edges, np.inf]
    results["inhibition_bin"] = pd.cut(results.inhibition, bins, right=False).astype(str)
    bin_records = [
        {"inhibition_bin": key, **metric_summary(frame)}
        for key, frame in results.groupby("inhibition_bin", sort=True)
    ]
    pd.DataFrame(bin_records).to_csv(output_dir / "by_inhibition.csv", index=False)
    report = {
        "split": split,
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_fingerprint": bundle.manifest["fingerprint"],
        "evaluation_settings": config.to_dict()["evaluation"],
        "overall": metric_summary(results),
        "group_macro": groups,
        "near_target": metric_summary(near_target),
        "point_definition": "inverse_transform(weighted_mean_of_scaled_component_means)",
        "density_units": {
            "nll": "standardized log1p dose",
            "nll_um": "per square micromolar",
        },
        "limitations": [
            "Point error does not establish inhibition achievement.",
            "An unrestricted MDN does not guarantee final drug-order equivariance.",
            "Near-target metrics use measured responses within the recorded tolerance.",
            "Gaussian predictions may have negative dose coordinates; no clipping.",
        ],
    }
    from .assessment import enhanced_assessment

    if reference_bundle is not None:
        report["external_protocol"] = dict(bundle.manifest["external_protocol"])

    raw_available = "matrix_design.parquet" in bundle.manifest.get("artifacts", {}) or (
        "source_file" in results
        and all(
            DatasetPaths(Path(config.dataset_root), config.dataset_name).source_path(name).is_file()
            for name in results.source_file.unique()
        )
    )
    assessment = enhanced_assessment(
        bundle,
        config,
        model,
        results.drop(columns="inhibition_bin"),
        arrays,
        output_dir / "assessment",
        split,
        include_target=raw_available,
    )
    report["primary"] = assessment["primary"]
    report["target_achievement"] = assessment["target_achievement"]
    report["generalization"] = assessment["generalization"]
    report["biological_units"] = assessment["biological_units"]
    report["assessment_artifacts"] = "assessment/assessment.json"
    atomic_json(output_dir / "metrics.json", report)
    LOGGER.info(
        "%s NLL=%.6f coordinate MAE=%.6f; artifacts: %s",
        split,
        report["overall"]["nll"],
        report["overall"]["mae_scaled"],
        output_dir,
    )
    return report
