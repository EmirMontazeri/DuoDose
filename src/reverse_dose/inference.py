"""Portable inverse-dose distributions, candidate coordinates, samples and maps."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, config_from_dict
from .data import DataBundle
from .evaluation import load_trained_model
from .models import MixtureParameters, mixture_log_prob, sample_mixture
from .utils import (
    atomic_json,
    event,
    guard_output,
    load_checkpoint,
    read_json,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)


class Predictor:
    """Inference with the same feature definitions and dose transform as training.

    Drug identifiers are canonicalized exactly as during source preparation;
    returned coordinates are restored to the caller's requested A/B order.
    The distribution remains the proposal's original M-component MDN.
    """

    def __init__(self, data_dir: Path, checkpoint: Path, device: str = "cpu"):
        self.bundle = DataBundle(data_dir)
        self.checkpoint_path = checkpoint
        self.config: ExperimentConfig = config_from_dict(
            load_checkpoint(checkpoint)["config"]
        )
        self.config.device = device
        self.model = load_trained_model(self.bundle, self.config, checkpoint, device)
        self.device = next(self.model.parameters()).device
        self.cell_lookup = {str(name): i for i, name in enumerate(self.bundle.cell_ids)}
        self.drug_lookup = {str(name): i for i, name in enumerate(self.bundle.drug_ids)}

    @classmethod
    def from_export(cls, directory: Path, device: str = "cpu") -> "Predictor":
        """Load and hash-check a self-contained exported prediction directory."""
        manifest = read_json(directory / "export.json")
        for name, expected in manifest["files"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Export contains an invalid path")
            if sha256_file(path) != expected:
                raise ValueError(f"Export integrity failure: {name}")
        return cls(directory / "data", directory / "model.pt", device)

    def expression_from_csv(self, path: Path) -> np.ndarray:
        """Transform a new cell's measured TPM table (gene_id,rsem_tpm).

        Identifiers and measurement values are required; duplicate gene rows use
        the same explicitly documented arithmetic mean as source extraction.
        Training gene definitions and imputation are reused, never refitted.
        """
        table = pd.read_csv(path, dtype={"gene_id": str})
        if not {"gene_id", "rsem_tpm"} <= set(table.columns):
            raise ValueError("New-cell expression CSV requires gene_id and rsem_tpm")
        table["rsem_tpm"] = pd.to_numeric(table.rsem_tpm, errors="raise")
        if (
            table.gene_id.isna().any()
            or table.gene_id.str.strip().eq("").any()
            or np.isinf(table.rsem_tpm).any()
            or (table.rsem_tpm < 0).any()
        ):
            raise ValueError(
                "New-cell genes and nonnegative TPM or explicit missing values are required"
            )
        table = table.groupby("gene_id", sort=True, as_index=False).rsem_tpm.mean()
        return self.bundle.expression_transform.transform(
            table.rsem_tpm.to_numpy()[None, :], table.gene_id.to_numpy()
        )[0]

    @torch.no_grad()
    def distribution(
        self,
        cell_id: str,
        drug_a: str,
        drug_b: str,
        inhibition: float,
        expression: np.ndarray | None = None,
    ) -> MixtureParameters:
        """Return one bivariate mixture in scaled coordinates and requested drug order.

        ``expression`` is an already transformed vector from expression_from_csv;
        cell metadata may describe a cell never used in training. Inhibition uses
        fraction units; measured out-of-range inputs are supported and surfaced.
        """
        if not np.isfinite(inhibition):
            raise ValueError("Target inhibition must be finite (fraction units)")
        missing = [drug for drug in (drug_a, drug_b) if drug not in self.drug_lookup]
        if missing:
            raise ValueError(
                f"No released feature representation for drug(s): {missing}"
            )
        if expression is None:
            if cell_id not in self.cell_lookup:
                raise ValueError(
                    f"Unknown cell {cell_id}; provide a gene_id,rsem_tpm expression CSV"
                )
            expression = self.bundle.cell_features[self.cell_lookup[cell_id]]
        if np.shape(expression) != (self.bundle.cell_features.shape[1],) or not (
            np.isfinite(expression).all()
        ):
            raise ValueError(
                "Cell feature vector has wrong dimension or non-finite values"
            )
        first, second = sorted((drug_a, drug_b))

        def tensor(value: Any) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        parameters = self.model(
            tensor(expression)[None],
            tensor([[inhibition]]),
            tensor(self.bundle.drug_features[self.drug_lookup[first]])[None],
            tensor(self.bundle.drug_features[self.drug_lookup[second]])[None],
        )
        if first != drug_a:
            parameters = MixtureParameters(
                parameters.logits,
                parameters.means.flip(-1),
                parameters.scales.flip(-1),
                parameters.correlations,
            )
        return parameters

    def _training_bounds(self, drug: str) -> tuple[tuple[float, float], str]:
        rows = self.bundle.observations
        rows = rows.loc[rows.split == "train"]
        values = np.concatenate(
            (
                rows.loc[rows.drug_a == drug, "dose_a_um"].to_numpy(),
                rows.loc[rows.drug_b == drug, "dose_b_um"].to_numpy(),
            )
        )
        source = "drug-specific training observations"
        if not len(values):
            values = rows[["dose_a_um", "dose_b_um"]].to_numpy().ravel()
            source = "pooled training observations; drug absent from training"
        return (0.0, max(float(np.max(values)), np.finfo(float).eps)), source

    @torch.no_grad()
    def predict(
        self,
        cell_id: str,
        drug_a: str,
        drug_b: str,
        inhibition: float,
        *,
        expression: np.ndarray | None = None,
        samples: int = 0,
        seed: int = 42,
        grid_size: int = 0,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Produce unmodified mixture parameters, candidates and optional density map.

        Negative dose candidates/samples remain visible with validity masks. Grid
        densities use the original-unit Jacobian and are never renormalized to the
        plotted rectangle. Grid bounds come only from training observations unless
        supplied. Returned candidate locations are transformed Gaussian means,
        not claimed modes of the density after nonlinear transformation.
        """
        if samples < 0 or seed < 0 or (grid_size != 0 and grid_size < 2):
            raise ValueError(
                "samples/seed must be nonnegative and grid_size either 0 or >=2"
            )
        params = self.distribution(cell_id, drug_a, drug_b, inhibition, expression)
        # CPU sampling uses a local RNG and does not perturb training/global state.
        params = MixtureParameters(
            *(
                getattr(params, key).cpu()
                for key in ("logits", "means", "scales", "correlations")
            )
        )
        means = params.means.numpy()[0].astype(np.float64)
        candidates = self.bundle.dose_transform.inverse(means)
        point = self.bundle.dose_transform.inverse(
            params.mixture_mean.numpy().astype(np.float64)
        )[0]
        train = self.bundle.observations.loc[self.bundle.observations.split == "train"]
        train_range = (float(train.inhibition.min()), float(train.inhibition.max()))
        result = {
            "cell_id": cell_id,
            "drug_a": drug_a,
            "drug_b": drug_b,
            "target_inhibition_fraction": inhibition,
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "dataset_fingerprint": self.bundle.manifest["fingerprint"],
            "drug_order_policy": "canonical input, output restored to requested order",
            "training_inhibition_range": train_range,
            "target_outside_training_range": not train_range[0]
            <= inhibition
            <= train_range[1],
            "weights": params.weights.numpy()[0],
            "means_scaled": means,
            "standard_deviations_scaled": params.scales.numpy()[0],
            "correlations": params.correlations.numpy()[0],
            "candidate_coordinates_um": candidates,
            "candidate_nonnegative": (candidates >= 0).all(axis=-1),
            "point_coordinates_um": point,
            "point_nonnegative": bool((point >= 0).all()),
            "point_definition": "inverse_transform(weighted_mean_of_scaled_component_means)",
            "limitations": [
                "Coordinates are predictions, not validated target-achieving doses.",
                "The Gaussian distribution has support outside nonnegative doses.",
                "Component locations are not necessarily density modes in micromolar units.",
            ],
        }
        arrays = {}
        if samples:
            draws = sample_mixture(
                params, samples, torch.Generator().manual_seed(seed)
            ).numpy()[0]
            doses = self.bundle.dose_transform.inverse(draws.astype(np.float64))
            arrays["samples_um"] = doses
            arrays["samples_nonnegative"] = (doses >= 0).all(axis=-1)
            result["sampling"] = {
                "count": samples,
                "seed": seed,
                "nonnegative_fraction": float(arrays["samples_nonnegative"].mean()),
            }
        if grid_size:
            if bounds is None:
                first, first_source = self._training_bounds(drug_a)
                second, second_source = self._training_bounds(drug_b)
                bounds = (first, second)
                bound_sources = [first_source, second_source]
            else:
                bound_sources = ["caller-supplied", "caller-supplied"]
            if any(
                not np.isfinite([lo, hi]).all() or lo < 0 or hi <= lo
                for lo, hi in bounds
            ):
                raise ValueError("Grid bounds require finite 0 <= lower < upper")
            axes = [np.linspace(lo, hi, grid_size) for lo, hi in bounds]
            aa, bb = np.meshgrid(*axes, indexing="xy")
            doses = np.stack((aa.ravel(), bb.ravel()), axis=-1)
            scaled = self.bundle.dose_transform.transform(doses)
            density_parts = []
            for start in range(0, len(doses), 4096):
                count = min(4096, len(doses) - start)
                expanded = MixtureParameters(
                    *(
                        getattr(params, key).expand(
                            count, *getattr(params, key).shape[1:]
                        )
                        for key in ("logits", "means", "scales", "correlations")
                    )
                )
                logp = mixture_log_prob(
                    expanded,
                    torch.tensor(scaled[start : start + count], dtype=torch.float32),
                )
                jacobian = self.bundle.dose_transform.log_abs_det_jacobian(
                    doses[start : start + count]
                )
                if np.ndim(jacobian) == 2:
                    jacobian = jacobian.sum(axis=-1)
                density_parts.append(np.exp(logp.numpy().astype(np.float64) + jacobian))
            density = np.concatenate(density_parts).reshape(grid_size, grid_size)
            mass = np.trapezoid(np.trapezoid(density, axes[0], axis=1), axes[1])
            arrays.update(dose_a_um=axes[0], dose_b_um=axes[1], density_per_um2=density)
            result["grid"] = {
                "bounds_um": bounds,
                "size": grid_size,
                "bounds_sources": bound_sources,
                "approximate_unconditional_mass_in_rectangle": float(mass),
                "quadrature": "trapezoidal; finite grid is an approximation",
            }
        return result, arrays


def write_prediction(
    output_dir: Path,
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
    dataset_root: Path = Path("dataset"),
) -> None:
    """Save JSON, optional NPZ samples/map and a standalone scientific PNG figure."""
    guard_output(output_dir, dataset_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Prediction output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "prediction.json", result)
    if arrays:
        np.savez_compressed(output_dir / "distribution.npz", **arrays)
    if "density_per_um2" in arrays:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        mesh = axis.pcolormesh(
            arrays["dose_a_um"],
            arrays["dose_b_um"],
            arrays["density_per_um2"],
            shading="auto",
            cmap="viridis",
        )
        axis.set(
            xlabel=f"{result['drug_a']} dose (µM)",
            ylabel=f"{result['drug_b']} dose (µM)",
            title=f"{result['cell_id']}; target inhibition {result['target_inhibition_fraction']:.3g}",
        )
        fig.colorbar(mesh, ax=axis, label="Density (µM⁻²)")
        fig.savefig(output_dir / "probability_map.png", dpi=180)
        plt.close(fig)
    event(
        output_dir / "events.jsonl",
        {
            "event": "prediction_saved",
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "cell_id": result["cell_id"],
            "drug_a": result["drug_a"],
            "drug_b": result["drug_b"],
            "target_inhibition_fraction": result["target_inhibition_fraction"],
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        },
    )
    LOGGER.info("Prediction artifacts written to %s", output_dir)


def export_model(
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    dataset_root: Path = Path("dataset"),
) -> None:
    """Export selected weights and all feature definitions without any raw sources."""
    guard_output(output_dir, dataset_root)
    if output_dir.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("Export destination must be outside its source data directory")
    # Constructing a predictor validates the checkpoint and the entire data bundle.
    Predictor(data_dir, checkpoint)
    if output_dir.exists():
        raise FileExistsError(f"Export path exists: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copytree(data_dir, output_dir / "data")
    shutil.copy2(checkpoint, output_dir / "model.pt")
    event(
        output_dir / "events.jsonl",
        {"event": "model_exported", "checkpoint_sha256": sha256_file(checkpoint)},
    )
    files = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    atomic_json(
        output_dir / "export.json",
        {
            "format_version": 1,
            "files": files,
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    LOGGER.info("Portable prediction bundle exported to %s", output_dir)
