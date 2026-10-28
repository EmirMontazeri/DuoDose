"""Leakage-controlled expression, grouped splits and portable model datasets.

Expression source values are arithmetic-mean RSEM TPM, keyed by CMP ``gene_id``.
The offline VAE population contains every CMP cell-line model with released
cell-line RNA-seq, except identity groups reserved exclusively for final
validation or test. All gene selection, imputation and scaling state is fitted
exclusively on the offline VAE training models. Dose scaling is fitted on
final-model training rows. Neither transform learns from final validation/test
observations or metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .layout import DEFAULT_DATASET_NAME, DatasetPaths
from .preprocess import (
    INHIBITION_QUALITY_MAX,
    INHIBITION_QUALITY_MIN,
    response_quality_control,
)

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 2
EXPRESSION_POPULATION = "all CMP models with Broad or Sanger cell-line RNA-seq"
SPLITS = ("train", "validation", "test")


def _digest(path: Path) -> str:
    """Hash file contents without loading the complete file into memory."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")


def _ids(values: Sequence[Any], name: str) -> list[str]:
    if any(pd.isna(value) or not str(value).strip() for value in values):
        raise ValueError(f"{name} contains missing or empty identifiers")
    result = [str(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate identifiers")
    return result


def _guard_output(output_dir: Path, dataset_root: Path | None = None) -> None:
    resolved = output_dir.resolve()
    if dataset_root is not None and resolved.is_relative_to((dataset_root / "raw").resolve()):
        raise ValueError("Generated artifacts must never be written under dataset/raw")
    # Also protects CLI calls that supply a derived source directory rather than
    # dataset_root. The repository contract reserves the dataset/raw subtree.
    for parent in [resolved, *resolved.parents]:
        if parent.name == "raw" and parent.parent.name == "dataset":
            raise ValueError("Generated artifacts must never be written under dataset/raw")


def _verify_files(directory: Path, hashes: dict[str, str], label: str) -> None:
    """Verify a producer's relative artifact hashes before consuming its data."""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{label} manifest has no artifact hashes")
    for name, expected in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"{label} artifact missing or invalid path: {name}")
        if _digest(path) != expected:
            raise ValueError(f"{label} artifact integrity mismatch: {name}")


def _one_catalog(directory: Path, pattern: str, label: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(paths)}")
    return paths[0]


def _boolean_series(values: pd.Series, name: str) -> pd.Series:
    """Parse source availability flags without treating the string 'False' as true."""
    normalized = values.map(
        lambda value: (
            False
            if pd.isna(value)
            else (value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower())
        )
    )
    mapping = {True: True, False: False, "true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(set(normalized[invalid].astype(str)))[:5]
        raise ValueError(f"{name} contains invalid Boolean flags: {examples}")
    return normalized.map(mapping).astype(bool)


def _identity_groups(model_list: pd.DataFrame, model_ids: Sequence[str]) -> list[str]:
    """Return conservative CMP identity components for leakage-safe splitting.

    Components join exact stable sample/patient/database identifiers, normalized
    model names, and declared parent relationships. Generic/missing values never
    create edges. The component's lexicographically first model ID is its stable
    label.
    """
    required = {"model_id", "model_name"}
    _require_columns(model_list, required, "CMP model list")
    if model_list.model_id.isna().any() or model_list.model_id.duplicated().any():
        raise ValueError("CMP model list requires unique, nonmissing model_id")
    requested = set(_ids(model_ids, "expression model IDs"))
    available = set(model_list.model_id.astype(str))
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Expression models are absent from CMP model list: {missing[:10]}")

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: str, second: str) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    generic = {"", "-", "na", "n/a", "nan", "none", "null", "unknown", "not available"}

    def token(column: str, value: Any) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if text.lower() in generic:
            return None
        if column == "model_name":
            text = "".join(character for character in text.upper() if character.isalnum())
            if not text:
                return None
        return f"{column}:{text}"

    strong_columns = (
        "sample_id",
        "patient_id",
        "COSMIC_ID",
        "BROAD_ID",
        "CCLE_ID",
        "RRID",
        "model_name",
    )
    rows = model_list.copy()
    rows["model_id"] = rows.model_id.astype(str)
    for _, row in rows.iterrows():
        model = f"model_id:{row.model_id}"
        find(model)
        for column in strong_columns:
            if column in rows:
                value = token(column, row[column])
                if value is not None:
                    union(model, value)
        if "parent_id" in rows:
            value = token("parent_id", row["parent_id"])
            if value is not None:
                union(model, value)
                parent_model = str(row["parent_id"]).strip()
                if parent_model in available:
                    union(model, f"model_id:{parent_model}")

    components: dict[str, list[str]] = {}
    for model in rows.model_id:
        components.setdefault(find(f"model_id:{model}"), []).append(model)
    labels = {model: min(components[find(f"model_id:{model}")]) for model in requested}
    return [labels[model] for model in model_ids]


def extract_expression(
    dataset_root: Path,
    cells: pd.DataFrame,
    output_dir: Path,
    chunk_size: int = 250_000,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    reference_dataset_name: str | None = None,
) -> Path:
    """Stream and aggregate all released CMP cell-line RNA-seq model/gene values.

    Writes ``expression.npz`` (Unicode model/gene IDs, float64 TPM means and
    uint32 measurement counts, plus identity-component IDs) and a provenance
    manifest. ``cells`` identifies response models that must be present, but does
    not restrict the extracted population. Absent measurements remain NaN.
    Multiple source/dataset measurements are averaged without weighting.
    """
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    paths = DatasetPaths(dataset_root, reference_dataset_name or dataset_name)
    _guard_output(output_dir, dataset_root)
    _require_columns(cells, {"cell_id", "model_id"}, "cells")
    _ids(cells["cell_id"].tolist(), "cell_id")
    required_models = set(_ids(cells["model_id"].tolist(), "model_id"))
    if (
        not required_models
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("Expression extraction requires mapped cells and a positive chunk_size")
    archives = sorted((paths.raw / "cell_model_passports").glob("rnaseq_all_*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Expected exactly one rnaseq_all_*.zip, found {len(archives)}")
    archive = archives[0]
    source_sha256 = _digest(archive)
    catalog_dir = paths.raw / "cell_model_passports"
    availability_path = _one_catalog(
        catalog_dir,
        "model_dataset_availability_*.csv",
        "model_dataset_availability_*.csv catalog",
    )
    model_list_path = _one_catalog(catalog_dir, "model_list_*.csv", "model_list_*.csv catalog")
    availability = pd.read_csv(availability_path)
    rna_columns = ["RNASeq Broad Cell Lines", "RNASeq Sanger Cell Lines"]
    _require_columns(availability, {"model_id", *rna_columns}, "CMP availability catalog")
    if availability.model_id.isna().any() or availability.model_id.duplicated().any():
        raise ValueError("CMP availability catalog requires unique, nonmissing model_id")
    available = np.zeros(len(availability), dtype=bool)
    for column in rna_columns:
        available |= _boolean_series(availability[column], column).to_numpy()
    models = sorted(availability.loc[available, "model_id"].astype(str))
    missing_catalog_models = sorted(required_models - set(models))
    if missing_catalog_models:
        raise ValueError(
            "Mapped response models lack released cell-line RNA-seq availability: "
            f"{missing_catalog_models}"
        )
    model_list = pd.read_csv(model_list_path, dtype=str)
    catalog_hashes = {
        availability_path.relative_to(dataset_root).as_posix(): _digest(availability_path),
        model_list_path.relative_to(dataset_root).as_posix(): _digest(model_list_path),
    }
    path = output_dir / "expression.npz"
    manifest_path = output_dir / "expression_manifest.json"
    if manifest_path.exists() or path.exists():
        if not manifest_path.is_file() or not path.is_file():
            raise FileExistsError(
                f"Incomplete expression artifacts; use a new output directory: {output_dir}"
            )
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("schema_version") != SCHEMA_VERSION
            or cached.get("source_sha256") != source_sha256
            or cached.get("source_path") != archive.relative_to(dataset_root).as_posix()
            or cached.get("population_definition") != EXPRESSION_POPULATION
            or cached.get("catalog_hashes") != catalog_hashes
        ):
            raise ValueError(
                f"Existing expression artifacts differ from raw sources or extraction policy; use a new output directory: {output_dir}"
            )
        if set(cached.get("artifacts", {})) != {"expression.npz"}:
            raise ValueError(
                "Expression manifest does not declare the required expression.npz artifact"
            )
        _verify_files(output_dir, cached["artifacts"], "Cached expression")
        with np.load(path, allow_pickle=False) as arrays:
            cached_models = set(_ids(arrays["model_ids"].tolist(), "cached expression model IDs"))
            if "identity_group_ids" not in arrays.files:
                raise ValueError("Cached expression lacks identity_group_ids")
        missing_required = sorted(required_models - cached_models)
        if missing_required:
            raise ValueError(
                f"Mapped response models have no usable expression: {missing_required}"
            )
        LOGGER.info("Reusing verified expression cache %s (%s)", output_dir, source_sha256[:12])
        return path
    model_lookup = {model: index for index, model in enumerate(models)}
    gene_lookup: dict[str, int] = {}
    capacity = 65536
    sums = np.zeros((len(models), capacity), dtype=np.float64)
    counts = np.zeros((len(models), capacity), dtype=np.uint32)
    record_counts: Counter[str] = Counter()
    source_counts: Counter[tuple[str, str, str, str]] = Counter()
    LOGGER.info(
        "Extracting expression for all %d catalogued cell-line RNA-seq models from %s",
        len(models),
        archive,
    )
    with zipfile.ZipFile(archive) as zfile:
        members = [item for item in zfile.infolist() if item.filename.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(
                "RNA-seq archive must contain exactly one CSV; choose the source explicitly"
            )
        member = members[0]
        with zfile.open(member) as handle:
            columns = pd.read_csv(handle, nrows=0).columns
        required = {"model_id", "gene_id", "rsem_tpm"}
        if not required.issubset(columns):
            raise ValueError(f"RNA-seq source lacks columns {sorted(required - set(columns))}")
        provenance_columns = [
            name for name in ("dataset_id", "data_source", "duplicate") if name in columns
        ]
        usecols = sorted(required | set(provenance_columns))
        with zfile.open(member) as handle:
            reader = pd.read_csv(handle, usecols=usecols, dtype=str, chunksize=chunk_size)
            for chunk_index, chunk in enumerate(reader):
                record_counts["archive_rows"] += len(chunk)
                subset = chunk.loc[chunk["model_id"].isin(model_lookup)].copy()
                record_counts["population_rows"] += len(subset)
                if subset.empty:
                    continue
                if subset["gene_id"].isna().any() or subset["gene_id"].str.strip().eq("").any():
                    raise ValueError(
                        "Mapped RNA-seq rows contain missing gene_id; cannot assign biological features"
                    )
                for column in ("dataset_id", "data_source", "duplicate"):
                    if column not in subset:
                        subset[column] = "<not supplied>"
                grouped_sources = (
                    subset.fillna("<missing>")
                    .groupby(
                        ["model_id", "dataset_id", "data_source", "duplicate"],
                        sort=True,
                    )
                    .size()
                )
                source_counts.update(
                    {tuple(key): int(value) for key, value in grouped_sources.items()}
                )
                for gene in sorted(set(subset["gene_id"]) - set(gene_lookup)):
                    gene_lookup[gene] = len(gene_lookup)
                while len(gene_lookup) > capacity:
                    sums = np.pad(sums, ((0, 0), (0, capacity)))
                    counts = np.pad(counts, ((0, 0), (0, capacity)))
                    capacity *= 2
                numeric = pd.to_numeric(subset["rsem_tpm"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                finite = np.isfinite(numeric)
                record_counts["missing_or_nonnumeric_tpm"] += int(np.isnan(numeric).sum())
                record_counts["infinite_tpm"] += int(np.isinf(numeric).sum())
                record_counts["negative_tpm"] += int((finite & (numeric < 0)).sum())
                valid = finite & (numeric >= 0)
                record_counts["usable_measurements"] += int(valid.sum())
                model_indices = (
                    subset.loc[valid, "model_id"].map(model_lookup).to_numpy(dtype=np.int64)
                )
                gene_indices = (
                    subset.loc[valid, "gene_id"].map(gene_lookup).to_numpy(dtype=np.int64)
                )
                np.add.at(sums, (model_indices, gene_indices), numeric[valid])
                np.add.at(counts, (model_indices, gene_indices), 1)
                if chunk_index % max(1, 2_000_000 // chunk_size) == 0:
                    LOGGER.info(
                        "RNA-seq rows=%s population=%s usable=%s genes=%s",
                        record_counts["archive_rows"],
                        record_counts["population_rows"],
                        record_counts["usable_measurements"],
                        len(gene_lookup),
                    )
    genes = sorted(gene_lookup)
    order = [gene_lookup[gene] for gene in genes]
    counts = counts[:, order]
    sums = sums[:, order]
    values = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=values, where=counts > 0)
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("RNA-seq aggregation overflowed; TPM means must be finite")
    unusable_models = [model for model, row in zip(models, counts, strict=True) if not np.any(row)]
    missing_required = sorted(required_models & set(unusable_models))
    if missing_required:
        raise ValueError(
            f"Mapped response models have no finite nonnegative TPM measurements: {missing_required}"
        )
    usable = np.asarray([model not in set(unusable_models) for model in models])
    models = [model for model, keep in zip(models, usable, strict=True) if keep]
    values = values[usable]
    counts = counts[usable]
    identity_group_ids = _identity_groups(model_list, models)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        path,
        model_ids=np.asarray(models),
        gene_ids=np.asarray(genes),
        values=values,
        measurement_counts=counts,
        identity_group_ids=np.asarray(identity_group_ids),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "population_definition": EXPRESSION_POPULATION,
        "source_path": archive.relative_to(dataset_root).as_posix(),
        "source_sha256": source_sha256,
        "catalog_hashes": catalog_hashes,
        "archive_member": member.filename,
        "archive_member_crc32": member.CRC,
        "aggregation": "arithmetic mean of all finite nonnegative RSEM TPM per model_id/gene_id across datasets/sources",
        "missing_policy": "invalid TPM excluded and counted; missing model/gene values stored as NaN",
        "gene_key": "CMP gene_id",
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__},
        "record_counts": dict(record_counts),
        "n_models": len(models),
        "catalogued_models": int(available.sum()),
        "models_without_usable_expression": unusable_models,
        "n_genes": len(genes),
        "missing_model_gene_values": int((counts == 0).sum()),
        "multiple_measurement_model_gene_values": int((counts > 1).sum()),
        "maximum_measurements_per_model_gene": int(counts.max()),
        "identity_definition": "connected CMP stable sample/patient/database identifiers, normalized model_name, and parent relationships",
        "source_measurement_counts": [
            dict(
                zip(
                    ("model_id", "dataset_id", "data_source", "duplicate"),
                    key,
                    strict=True,
                ),
                rows=count,
            )
            for key, count in sorted(source_counts.items())
        ],
        "artifacts": {"expression.npz": _digest(path)},
    }
    _atomic_json(manifest_path, manifest)
    LOGGER.info(
        "Saved raw expression %s: %d cell-line models × %d genes; %d missing model/gene values",
        path,
        len(models),
        len(genes),
        int((counts == 0).sum()),
    )
    return path


@dataclass(frozen=True)
class DoseTransform:
    """Shared scalar log-dose transform preserving both coordinate identities.

    ``z=(log1p(dose_um/reference_um)-mean)/scale``. Inversion deliberately permits
    negative model predictions; clipping would alter the specified Gaussian MDN.
    """

    reference_um: float
    mean: float
    scale: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.reference_um, self.mean, self.scale]).all():
            raise ValueError("Dose transform parameters must be finite")
        if self.reference_um <= 0 or self.scale <= 0:
            raise ValueError("Dose reference and scale must be positive")

    @classmethod
    def fit(cls, doses_um: np.ndarray, reference_um: float = 1.0) -> DoseTransform:
        """Fit one population mean/std over both coordinates of training doses."""
        values = np.asarray(doses_um, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
            raise ValueError("Training doses must have nonempty shape [N, 2]")
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isfinite(reference_um)
            or reference_um <= 0
        ):
            raise ValueError("Doses must be finite and nonnegative; reference must be positive")
        logs = np.log1p(values / reference_um)
        if not np.isfinite(logs).all():
            raise ValueError("Dose logarithm overflowed; check reference units")
        scale = float(logs.std())
        if scale <= np.finfo(np.float64).eps:
            raise ValueError("Training dose coordinates have no variation")
        return cls(float(reference_um), float(logs.mean()), scale)

    def transform(self, doses_um: np.ndarray) -> np.ndarray:
        """Map finite, nonnegative doses in µM to scaled log coordinates."""
        values = np.asarray(doses_um, dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Input doses must be finite and nonnegative")
        with np.errstate(over="raise", invalid="raise"):
            return (np.log1p(values / self.reference_um) - self.mean) / self.scale

    def inverse(self, coordinates: np.ndarray) -> np.ndarray:
        """Map scaled coordinates to µM; fail visibly if exponentiation overflows."""
        values = np.asarray(coordinates, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Predicted dose coordinates must be finite")
        with np.errstate(over="raise", invalid="raise"):
            return self.reference_um * np.expm1(values * self.scale + self.mean)

    def log_abs_det_jacobian(self, doses_um: np.ndarray) -> np.ndarray:
        """Return log|dz/dd| summed over the final two-dose dimension."""
        values = np.asarray(doses_um, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 2:
            raise ValueError("Jacobian input must have final dimension 2")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Jacobian input must be finite and nonnegative")
        return (-np.log(values + self.reference_um) - np.log(self.scale)).sum(axis=-1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_um": self.reference_um,
            "mean": self.mean,
            "scale": self.scale,
            "definition": "(log1p(dose_um/reference_um)-mean)/scale",
            "std_ddof": 0,
        }


@dataclass(frozen=True)
class ExpressionTransform:
    """Selected genes with VAE-training-model log1p-TPM imputation and scaling.

    Input gene order may differ but identifiers must be unique and include every
    selected gene. Explicit NaNs are imputed; an entirely missing selected vector
    fails. No unseen-cell statistics are fitted or updated during transformation.
    """

    gene_ids: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        _ids(self.gene_ids, "selected gene_ids")
        if (
            not self.gene_ids
            or self.mean.shape != (len(self.gene_ids),)
            or self.scale.shape != self.mean.shape
        ):
            raise ValueError("Expression transform dimensions do not match selected gene IDs")
        if (
            not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError(
                "Expression transform requires finite means and positive finite scales"
            )

    @classmethod
    def fit(cls, values: np.ndarray, gene_ids: Sequence[str], n_genes: int) -> ExpressionTransform:
        """Select highest-variance genes after training-mean imputation (ddof=0)."""
        genes = _ids(gene_ids, "gene_ids")
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(genes) or raw.shape[0] < 2:
            raise ValueError("Expression fitting requires [at least 2 training cells, genes]")
        if isinstance(n_genes, bool) or not isinstance(n_genes, (int, np.integer)) or n_genes < 1:
            raise ValueError("n_genes must be a positive integer")
        if np.isinf(raw).any() or (raw < 0).any():
            raise ValueError("TPM must be nonnegative and finite, or NaN for missing measurements")
        logs = np.log1p(raw)
        observed = np.isfinite(logs)
        counts = observed.sum(axis=0)
        mean = np.divide(
            np.nansum(logs, axis=0), counts, out=np.zeros(len(genes)), where=counts > 0
        )
        imputed = np.where(observed, logs, mean)
        variance = np.mean((imputed - mean) ** 2, axis=0)
        valid = np.flatnonzero((counts > 0) & (variance > np.finfo(np.float64).eps))
        if not len(valid):
            raise ValueError("No variable genes are observed among the VAE-training cells")
        selected = sorted(valid, key=lambda index: (-variance[index], genes[index]))[:n_genes]
        return cls(
            tuple(genes[index] for index in selected),
            mean[selected],
            np.sqrt(variance[selected]),
        )

    def transform(self, values: np.ndarray, gene_ids: Sequence[str]) -> np.ndarray:
        """Transform one vector [G] or a matrix [N,G] using supplied stable IDs."""
        genes = _ids(gene_ids, "input gene_ids")
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim not in (1, 2) or raw.shape[-1] != len(genes):
            raise ValueError("Expression values must have shape [genes] or [cells, genes]")
        if np.isinf(raw).any() or (raw < 0).any():
            raise ValueError("TPM must be nonnegative and finite, or explicitly NaN")
        lookup = {gene: index for index, gene in enumerate(genes)}
        missing = sorted(set(self.gene_ids) - set(lookup))
        if missing:
            raise ValueError(
                f"Input omits {len(missing)} required gene IDs; supply measured TPM or explicit NaN: {missing[:10]}"
            )
        selected = raw[..., [lookup[gene] for gene in self.gene_ids]]
        if (~np.isfinite(selected)).all(axis=-1).any():
            raise ValueError("A cell has no measured expression among selected genes")
        logs = np.log1p(selected)
        transformed = (np.where(np.isfinite(logs), logs, self.mean) - self.mean) / self.scale
        result = transformed.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Expression feature transform overflowed float32")
        return result

    def save(self, path: Path) -> None:
        _atomic_npz(
            Path(path),
            gene_ids=np.asarray(self.gene_ids),
            mean=self.mean,
            scale=self.scale,
        )

    @classmethod
    def load(cls, path: Path) -> ExpressionTransform:
        with np.load(path, allow_pickle=False) as values:
            return cls(tuple(values["gene_ids"].tolist()), values["mean"], values["scale"])


def _split_observations(
    observations: pd.DataFrame,
    config: dict[str, Any],
    cell_identity_groups: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol = config.get("protocol", "cell")
    seed = config.get("seed", 42)
    fractions = [
        config.get("validation_fraction", 0.15),
        config.get("test_fraction", 0.15),
    ]
    if protocol not in {"cell", "cell_pair"}:
        raise ValueError("Split protocol must be 'cell' or 'cell_pair'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    if not all(np.isfinite(value) and 0 < value < 1 for value in fractions) or sum(fractions) >= 1:
        raise ValueError("Validation/test fractions must be positive with sum below one")
    rows = observations.copy().reset_index(drop=True)
    missing_groups = sorted(set(rows.cell_id.astype(str)) - set(cell_identity_groups))
    if missing_groups:
        raise ValueError(f"Response cells lack expression identity groups: {missing_groups}")
    pairs = [tuple(sorted((str(a), str(b)))) for a, b in zip(rows.drug_a, rows.drug_b, strict=True)]
    group_keys = [
        (
            (cell_identity_groups[str(cell)],)
            if protocol == "cell"
            else (cell_identity_groups[str(cell)], *pair)
        )
        for cell, pair in zip(rows.cell_id, pairs, strict=True)
    ]
    groups = sorted(set(group_keys))
    n_validation, n_test = [max(1, int(round(len(groups) * fraction))) for fraction in fractions]
    if len(groups) - n_validation - n_test < 1:
        raise ValueError(f"Too few {protocol} groups ({len(groups)}) for three nonempty splits")
    permutation = np.random.default_rng(seed).permutation(len(groups))
    assignments = {
        groups[int(index)]: split
        for split, indices in (
            ("validation", permutation[:n_validation]),
            ("test", permutation[n_validation : n_validation + n_test]),
            ("train", permutation[n_validation + n_test :]),
        )
        for index in indices
    }
    rows["split"] = [assignments[key] for key in group_keys]
    # Entire matrices must remain together even in warm completion.
    if rows.groupby("matrix_id")["split"].nunique().max() > 1:
        raise ValueError("A matrix contains multiple cell/pair groups crossing split boundaries")
    cells_by_split = {
        split: sorted(set(rows.loc[rows.split.eq(split), "cell_id"])) for split in SPLITS
    }
    if len(cells_by_split["train"]) < 3:
        raise ValueError(
            "At least three final-training cells are required for VAE training and internal validation"
        )
    manifest = {
        "configuration": {
            "protocol": protocol,
            "seed": seed,
            "validation_fraction": fractions[0],
            "test_fraction": fractions[1],
            "vae_validation_fraction": config.get("vae_validation_fraction", 0.15),
        },
        "group_definition": (
            "CMP expression identity group"
            if protocol == "cell"
            else "CMP expression identity group + unordered drug pair (all matrices and replicates)"
        ),
        "assignment_algorithm": "sorted group keys, NumPy default_rng(seed) permutation, rounded holdout counts with minimum one",
        "group_counts": dict(Counter(assignments.values())),
        "observation_counts": {split: int(rows.split.eq(split).sum()) for split in SPLITS},
        "cells": cells_by_split,
        "cell_overlap_with_train": {
            split: sorted(set(cells_by_split["train"]) & set(cells_by_split[split]))
            for split in ("validation", "test")
        },
        "interpretation": (
            "cell cold start"
            if protocol == "cell"
            else "warm grouped completion; training-cell metadata may overlap evaluation cells"
        ),
    }
    return rows, manifest


def prepare_dataset(
    source_dir: Path,
    expression_dir: Path,
    output_dir: Path,
    split_config: dict[str, Any],
    feature_config: dict[str, Any],
) -> dict[str, Any]:
    """Construct a self-contained, fingerprinted training/evaluation bundle.

    Inputs are source ``cells.parquet``, ``observations.parquet``, ``drugs.npz``
    and ``source_manifest.json`` plus extracted expression artifacts. All fitting
    populations and split memberships are saved in human-readable JSON/parquet.
    """
    source_dir, expression_dir, output_dir = map(Path, (source_dir, expression_dir, output_dir))
    _guard_output(output_dir)
    if output_dir.resolve() in {source_dir.resolve(), expression_dir.resolve()}:
        raise ValueError("Prepared output must differ from source/expression artifact directories")
    unknown = set(split_config) - {
        "protocol",
        "seed",
        "validation_fraction",
        "test_fraction",
        "vae_validation_fraction",
    }
    unknown_features = set(feature_config) - {"n_genes", "dose_reference_um"}
    if unknown or unknown_features:
        raise ValueError(
            f"Unrecognized preprocessing configuration: {sorted(unknown | unknown_features)}"
        )
    source_files = [
        "cells.parquet",
        "observations.parquet",
        "drugs.npz",
        "source_manifest.json",
    ]
    expression_files = ["expression.npz", "expression_manifest.json"]
    source_manifest = json.loads((source_dir / "source_manifest.json").read_text())
    has_design = "matrix_design.parquet" in source_manifest.get("output_hashes", {})
    if has_design:
        source_files.append("matrix_design.parquet")
    expression_manifest = json.loads((expression_dir / "expression_manifest.json").read_text())
    if source_manifest.get("response_quality_control") != response_quality_control():
        raise ValueError(
            "Source manifest lacks the required pre-split inhibition quality-control policy"
        )
    source_hashes = source_manifest.get("output_hashes", {})
    expression_hashes = expression_manifest.get("artifacts", {})
    if (
        not (set(source_files) - {"source_manifest.json"}) <= set(source_hashes)
        or "expression.npz" not in expression_hashes
    ):
        raise ValueError(
            "Input manifests must declare hashes for all required source/expression artifacts"
        )
    _verify_files(source_dir, source_hashes, "Source")
    _verify_files(expression_dir, expression_hashes, "Expression")
    lineage = {
        "source_files": {name: _digest(source_dir / name) for name in source_files},
        "expression_files": {name: _digest(expression_dir / name) for name in expression_files},
    }
    preparation_identity = {
        "schema_version": SCHEMA_VERSION,
        "lineage": lineage,
        "split_configuration": {
            "protocol": "cell",
            "seed": 42,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "vae_validation_fraction": 0.15,
            **split_config,
        },
        "feature_configuration": {
            "n_genes": 2000,
            "dose_reference_um": 1.0,
            **feature_config,
        },
        "implementation": {
            "module_sha256": _digest(Path(__file__)),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        cached = DataBundle(output_dir)
        if cached.manifest.get("preparation_identity") != preparation_identity:
            raise ValueError(
                f"Existing prepared artifacts differ from inputs/configuration/code; use a new output directory: {output_dir}"
            )
        LOGGER.info("Reusing verified prepared cache %s", output_dir)
        return cached.manifest
    if output_dir.exists() and any(path.suffix != ".log" for path in output_dir.iterdir()):
        raise FileExistsError(
            f"Output contains incomplete/unrecognized artifacts; use a new output directory: {output_dir}"
        )
    cells = pd.read_parquet(source_dir / "cells.parquet")
    _require_columns(cells, {"cell_id", "model_id", "cosmic_id"}, "cells")
    _ids(cells.cell_id.tolist(), "cell_id")
    _ids(cells.model_id.tolist(), "model_id")
    rows = pd.read_parquet(source_dir / "observations.parquet")
    _require_columns(
        rows,
        {
            "observation_id",
            "matrix_id",
            "cell_id",
            "drug_a",
            "drug_b",
            "dose_a_um",
            "dose_b_um",
            "inhibition",
        },
        "observations",
    )
    _ids(rows.observation_id.tolist(), "observation_id")
    for column in ("matrix_id", "cell_id", "drug_a", "drug_b"):
        if rows[column].isna().any() or rows[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Observations contain missing {column}")
        rows[column] = rows[column].astype(str)
    numeric = rows[["dose_a_um", "dose_b_um", "inhibition"]].to_numpy(dtype=np.float64)
    if (
        not len(rows)
        or not np.isfinite(numeric).all()
        or (numeric[:, :2] < 0).any()
        or (numeric[:, 2] < INHIBITION_QUALITY_MIN).any()
        or (numeric[:, 2] > INHIBITION_QUALITY_MAX).any()
    ):
        raise ValueError(
            "Observations must have finite nonnegative doses and inhibition in inclusive "
            f"quality range [{INHIBITION_QUALITY_MIN}, {INHIBITION_QUALITY_MAX}]"
        )
    unknown_cells = set(rows.cell_id) - set(cells.cell_id)
    if unknown_cells:
        raise ValueError(f"Response cells have no validated model mapping: {sorted(unknown_cells)}")
    with np.load(source_dir / "drugs.npz", allow_pickle=False) as archive:
        drug_ids = _ids(archive["ids"].tolist(), "drug ids")
        drug_features = archive["features"].astype(np.float32)
    if (
        drug_features.ndim != 2
        or drug_features.shape[0] != len(drug_ids)
        or not drug_features.shape[1]
        or not np.isfinite(drug_features).all()
    ):
        raise ValueError(
            "Drug feature matrix is incompatible with its IDs or contains invalid values"
        )
    unknown_drugs = (set(rows.drug_a) | set(rows.drug_b)) - set(drug_ids)
    if unknown_drugs:
        raise ValueError(f"Observed drugs have no features: {sorted(unknown_drugs)}")
    with np.load(expression_dir / "expression.npz", allow_pickle=False) as archive:
        model_ids = _ids(archive["model_ids"].tolist(), "expression model_ids")
        genes = _ids(archive["gene_ids"].tolist(), "expression gene_ids")
        expression = archive["values"].astype(np.float64)
        identity_group_ids = archive["identity_group_ids"].tolist()
    if expression.shape != (len(model_ids), len(genes)):
        raise ValueError("Raw expression matrix dimensions do not match model/gene IDs")
    if len(identity_group_ids) != len(model_ids) or any(
        pd.isna(group) or not str(group).strip() for group in identity_group_ids
    ):
        raise ValueError("Expression identity groups must align with model IDs")
    identity_group_ids = [str(group) for group in identity_group_ids]
    missing_models = set(cells.model_id) - set(model_ids)
    if missing_models:
        raise ValueError(f"Response models are absent from expression: {sorted(missing_models)}")
    model_lookup = {model: index for index, model in enumerate(model_ids)}
    model_identity = dict(zip(model_ids, identity_group_ids, strict=True))
    response_identity = {
        str(row.cell_id): model_identity[str(row.model_id)] for row in cells.itertuples(index=False)
    }
    rows, split_manifest = _split_observations(rows, split_config, response_identity)
    cell_ids = sorted(set(rows.cell_id))
    cells = cells.set_index("cell_id").loc[cell_ids].reset_index()
    response_model_ids = cells.model_id.astype(str).tolist()
    response_raw = expression[[model_lookup[model] for model in response_model_ids]]
    cell_identity_group_ids = [model_identity[model] for model in response_model_ids]

    train_cells = set(split_manifest["cells"]["train"])
    evaluation_cells = set(split_manifest["cells"]["validation"]) | set(
        split_manifest["cells"]["test"]
    )
    evaluation_only_cells = sorted(evaluation_cells - train_cells)
    cell_model = dict(zip(cells.cell_id.astype(str), response_model_ids, strict=True))
    excluded_evaluation_groups = {
        model_identity[cell_model[cell]] for cell in evaluation_only_cells
    }
    eligible_indices = np.asarray(
        [
            index
            for index, group in enumerate(identity_group_ids)
            if group not in excluded_evaluation_groups
        ],
        dtype=np.int64,
    )
    eligible_model_ids = [model_ids[index] for index in eligible_indices]
    eligible_group_ids = [identity_group_ids[index] for index in eligible_indices]
    if len(eligible_model_ids) < 3:
        raise ValueError("At least three leakage-safe expression models are required")
    vae_fraction = split_config.get("vae_validation_fraction", 0.15)
    if not np.isfinite(vae_fraction) or not 0 < vae_fraction < 1:
        raise ValueError("vae_validation_fraction must be between zero and one")
    vae_groups = sorted(set(eligible_group_ids))
    n_vae_validation = max(1, round(len(vae_groups) * vae_fraction))
    if len(vae_groups) - n_vae_validation < 2:
        raise ValueError(
            "VAE split must retain at least two training identity groups and one validation group"
        )
    permutation = np.random.default_rng(
        np.random.SeedSequence([split_config.get("seed", 42), 1])
    ).permutation(len(vae_groups))
    vae_validation_groups = {vae_groups[int(index)] for index in permutation[:n_vae_validation]}
    vae_validation_models = sorted(
        model
        for model, group in zip(eligible_model_ids, eligible_group_ids, strict=True)
        if group in vae_validation_groups
    )
    vae_training_models = sorted(set(eligible_model_ids) - set(vae_validation_models))
    vae_model_ids = vae_training_models + vae_validation_models
    vae_raw_indices = np.asarray([model_lookup[model] for model in vae_model_ids])
    vae_raw = expression[vae_raw_indices]
    vae_lookup = {model: index for index, model in enumerate(vae_model_ids)}
    vae_train_indices = np.asarray(
        [vae_lookup[model] for model in vae_training_models], dtype=np.int64
    )
    vae_validation_indices = np.asarray(
        [vae_lookup[model] for model in vae_validation_models], dtype=np.int64
    )
    n_genes = feature_config.get("n_genes", 2000)
    expression_transform = ExpressionTransform.fit(vae_raw[vae_train_indices], genes, n_genes)
    vae_features = expression_transform.transform(vae_raw, genes)
    cell_features = expression_transform.transform(response_raw, genes)
    dose_transform = DoseTransform.fit(
        numeric[rows.split.eq("train"), :2],
        feature_config.get("dose_reference_um", 1.0),
    )
    targets = dose_transform.transform(numeric[:, :2]).astype(np.float32)
    excluded_evaluation_models = sorted(
        model
        for model, group in zip(model_ids, identity_group_ids, strict=True)
        if group in excluded_evaluation_groups
    )
    split_manifest["vae_population_definition"] = EXPRESSION_POPULATION
    split_manifest["vae_candidate_models"] = len(model_ids)
    split_manifest["vae_eligible_models"] = len(vae_model_ids)
    split_manifest["vae_training_models"] = vae_training_models
    split_manifest["vae_validation_models"] = vae_validation_models
    split_manifest["vae_excluded_evaluation_cells"] = evaluation_only_cells
    split_manifest["vae_excluded_evaluation_models"] = excluded_evaluation_models
    split_manifest["vae_excluded_evaluation_identity_groups"] = sorted(excluded_evaluation_groups)
    split_manifest["vae_assignment_algorithm"] = (
        "all non-evaluation-only CMP cell-line RNA-seq identity groups; sorted groups; "
        "NumPy default_rng(SeedSequence([seed,1])) permutation; related models stay together"
    )
    train_groups = {model_identity[model] for model in vae_training_models}
    validation_groups = {model_identity[model] for model in vae_validation_models}
    if (
        train_groups & validation_groups
        or train_groups & excluded_evaluation_groups
        or validation_groups & excluded_evaluation_groups
    ):
        raise AssertionError("VAE identity groups cross a training/validation/test boundary")
    LOGGER.info(
        "Split %s: %s; VAE models training=%d validation=%d excluded evaluation identities=%d",
        split_config.get("protocol", "cell"),
        split_manifest["observation_counts"],
        len(vae_training_models),
        len(vae_validation_models),
        len(excluded_evaluation_models),
    )
    LOGGER.info(
        "Fitted %d/%d requested expression features using only %d VAE-training models",
        cell_features.shape[1],
        n_genes,
        len(vae_training_models),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output_dir / "observations.parquet", rows)
    _atomic_parquet(output_dir / "cells.parquet", cells)
    _atomic_parquet(
        output_dir / "splits.parquet",
        rows[["observation_id", "matrix_id", "cell_id", "drug_a", "drug_b", "split"]],
    )
    _atomic_npz(
        output_dir / "features.npz",
        cell_ids=np.asarray(cell_ids),
        cell_features=cell_features,
        cell_identity_group_ids=np.asarray(cell_identity_group_ids),
        vae_model_ids=np.asarray(vae_model_ids),
        vae_identity_group_ids=np.asarray([model_identity[model] for model in vae_model_ids]),
        vae_features=vae_features,
        drug_ids=np.asarray(drug_ids),
        drug_features=drug_features,
        targets=targets,
        vae_train_indices=vae_train_indices,
        vae_validation_indices=vae_validation_indices,
    )
    expression_transform.save(output_dir / "expression_transform.npz")
    _atomic_json(output_dir / "dose_transform.json", dose_transform.to_dict())
    _atomic_json(output_dir / "split_manifest.json", split_manifest)
    _atomic_bytes(
        output_dir / "source_manifest.json",
        (source_dir / "source_manifest.json").read_bytes(),
    )
    _atomic_bytes(
        output_dir / "expression_manifest.json",
        (expression_dir / "expression_manifest.json").read_bytes(),
    )
    artifact_names = [
        "observations.parquet",
        "cells.parquet",
        "splits.parquet",
        "features.npz",
        "expression_transform.npz",
        "dose_transform.json",
        "split_manifest.json",
        "source_manifest.json",
        "expression_manifest.json",
    ]
    if has_design:
        _atomic_bytes(
            output_dir / "matrix_design.parquet",
            (source_dir / "matrix_design.parquet").read_bytes(),
        )
        artifact_names.append("matrix_design.parquet")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "preparation_identity": preparation_identity,
        "dataset_name": source_manifest.get("dataset_name"),
        "sources": source_manifest.get("config", {}).get("sources"),
        "lineage": lineage,
        "split": split_manifest,
        "features": {
            "n_genes_requested": n_genes,
            "n_genes": cell_features.shape[1],
            "n_drug_features": drug_features.shape[1],
            "expression_fit_models": vae_training_models,
            "expression_gene_key": "CMP gene_id",
            "expression_definition": "log1p TPM; mean imputation; select highest population variance; population standardization",
            "expression_missing_policy": "require all selected identifiers; explicit NaN imputed with training mean; reject entirely missing selected vector",
            "expression_selection_ties": "ascending gene_id",
            "excluded_genes": "zero variance or wholly unobserved among VAE-training models",
            "dose_fit_rows": "train",
            "dose_transform": dose_transform.to_dict(),
            "inhibition_definition": source_manifest.get(
                "response_definition",
                (
                    "1 - percent_control_viability / 100; no clipping; retain only inclusive "
                    f"quality range [{INHIBITION_QUALITY_MIN}, {INHIBITION_QUALITY_MAX}]"
                ),
            ),
            "response_quality_control": response_quality_control(),
        },
        "n_cells": len(cell_ids),
        "n_drugs": len(drug_ids),
        "n_observations": len(rows),
        "artifacts": {name: _digest(output_dir / name) for name in artifact_names},
    }
    manifest["fingerprint"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    _atomic_json(manifest_path, manifest)
    LOGGER.info("Saved prepared dataset %s fingerprint=%s", output_dir, manifest["fingerprint"])
    return manifest


class DoseDataset(Dataset[dict[str, torch.Tensor]]):
    """Observation tensors with metadata features and two scaled dose targets.

    Each item contains cell [Nc], inhibition [1], drug_a/drug_b [Nd], target [2]
    and scalar ``index`` identifying its positional row in bundle.observations.
    """

    def __init__(self, bundle: DataBundle, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected {SPLITS}")
        self.bundle = bundle
        self.indices = np.flatnonzero(bundle.observations.split.eq(split).to_numpy())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        bundle = self.bundle
        return {
            "cell": torch.from_numpy(bundle.cell_features[bundle._cell_indices[row]]),
            "inhibition": torch.from_numpy(bundle._inhibition[row]),
            "drug_a": torch.from_numpy(bundle.drug_features[bundle._drug_a_indices[row]]),
            "drug_b": torch.from_numpy(bundle.drug_features[bundle._drug_b_indices[row]]),
            "target": torch.from_numpy(bundle.targets[row]),
            "index": torch.tensor(row, dtype=torch.long),
        }


class DataBundle:
    """Validated portable artifacts for training, evaluation and inference.

    Loading verifies every artifact hash and the manifest fingerprint. Paths are
    relative to the supplied directory, so copying a bundle preserves identity.
    """

    def __init__(self, output_dir: Path) -> None:
        self.path = Path(output_dir)
        self.manifest: dict[str, Any] = json.loads((self.path / "manifest.json").read_text())
        expected = self.manifest.get("fingerprint")
        content = {key: value for key, value in self.manifest.items() if key != "fingerprint"}
        if hashlib.sha256(_json_bytes(content)).hexdigest() != expected:
            raise ValueError("Prepared dataset manifest fingerprint mismatch")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported prepared dataset schema version")
        for name, digest in self.manifest["artifacts"].items():
            path = self.path / name
            if not path.resolve().is_relative_to(self.path.resolve()) or _digest(path) != digest:
                raise ValueError(f"Prepared dataset artifact integrity mismatch: {name}")
        self.observations = pd.read_parquet(self.path / "observations.parquet").reset_index(
            drop=True
        )
        self.cells = pd.read_parquet(self.path / "cells.parquet")
        with np.load(self.path / "features.npz", allow_pickle=False) as arrays:
            self.cell_ids = _ids(arrays["cell_ids"].tolist(), "cell IDs")
            self.cell_identity_group_ids = [
                str(value) for value in arrays["cell_identity_group_ids"].tolist()
            ]
            self.vae_model_ids = _ids(arrays["vae_model_ids"].tolist(), "VAE model IDs")
            self.vae_identity_group_ids = [
                str(value) for value in arrays["vae_identity_group_ids"].tolist()
            ]
            self.drug_ids = _ids(arrays["drug_ids"].tolist(), "drug IDs")
            self.cell_features = arrays["cell_features"]
            self.vae_features = arrays["vae_features"]
            self.drug_features = arrays["drug_features"]
            self.targets = arrays["targets"]
            self.vae_train_indices = arrays["vae_train_indices"]
            self.vae_validation_indices = arrays["vae_validation_indices"]
        self.expression_transform = ExpressionTransform.load(self.path / "expression_transform.npz")
        dose_parameters = json.loads((self.path / "dose_transform.json").read_text())
        self.dose_transform = DoseTransform(
            **{key: dose_parameters[key] for key in ("reference_um", "mean", "scale")}
        )
        expected_shapes = (
            (
                self.cell_features,
                (len(self.cell_ids), len(self.expression_transform.gene_ids)),
            ),
            (
                self.vae_features,
                (len(self.vae_model_ids), len(self.expression_transform.gene_ids)),
            ),
            (
                self.drug_features,
                (len(self.drug_ids), self.manifest["features"]["n_drug_features"]),
            ),
            (self.targets, (len(self.observations), 2)),
        )
        if any(
            array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all()
            for array, shape in expected_shapes
        ):
            raise ValueError(
                "Prepared feature/target arrays have invalid dimensions, dtype or values"
            )
        if not set(self.observations.split) == set(SPLITS):
            raise ValueError(
                "Prepared dataset must contain precisely train, validation and test splits"
            )
        inhibition = self.observations["inhibition"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(inhibition).all()
            or (inhibition < INHIBITION_QUALITY_MIN).any()
            or (inhibition > INHIBITION_QUALITY_MAX).any()
            or self.manifest["features"].get("response_quality_control")
            != response_quality_control()
        ):
            raise ValueError("Prepared dataset violates inhibition quality control")
        if set(self.vae_train_indices) & set(self.vae_validation_indices):
            raise ValueError("VAE training and validation models overlap")
        if (
            len(self.cell_identity_group_ids) != len(self.cell_ids)
            or len(self.vae_identity_group_ids) != len(self.vae_model_ids)
            or any(not value for value in self.cell_identity_group_ids)
            or any(not value for value in self.vae_identity_group_ids)
        ):
            raise ValueError("Expression identity groups do not align with feature rows")
        for indices in (self.vae_train_indices, self.vae_validation_indices):
            if (
                indices.ndim != 1
                or not len(indices)
                or (indices < 0).any()
                or (indices >= len(self.vae_model_ids)).any()
            ):
                raise ValueError("VAE model indices are invalid")
        covered = set(self.vae_train_indices.tolist()) | set(self.vae_validation_indices.tolist())
        if covered != set(range(len(self.vae_model_ids))):
            raise ValueError("VAE train/validation indices must cover every eligible model")
        train_groups = {self.vae_identity_group_ids[index] for index in self.vae_train_indices}
        validation_groups = {
            self.vae_identity_group_ids[index] for index in self.vae_validation_indices
        }
        excluded_groups = set(self.manifest["split"]["vae_excluded_evaluation_identity_groups"])
        if (
            train_groups & validation_groups
            or train_groups & excluded_groups
            or validation_groups & excluded_groups
        ):
            raise ValueError("VAE identity groups cross a protected split boundary")
        cell_lookup = {cell: index for index, cell in enumerate(self.cell_ids)}
        drug_lookup = {drug: index for index, drug in enumerate(self.drug_ids)}
        for series, lookup in (
            (self.observations.cell_id, cell_lookup),
            (self.observations.drug_a, drug_lookup),
            (self.observations.drug_b, drug_lookup),
        ):
            if not set(series).issubset(lookup):
                raise ValueError("Observation references an unknown feature identifier")
        split = self.manifest["split"]
        train_models = {self.vae_model_ids[index] for index in self.vae_train_indices}
        validation_models = {self.vae_model_ids[index] for index in self.vae_validation_indices}
        if train_models != set(split["vae_training_models"]) or validation_models != set(
            split["vae_validation_models"]
        ):
            raise ValueError("VAE model arrays disagree with the audited split manifest")
        if split["vae_eligible_models"] != len(self.vae_model_ids):
            raise ValueError("VAE eligible-model count disagrees with feature arrays")
        cell_groups = dict(zip(self.cell_ids, self.cell_identity_group_ids, strict=True))
        train_cells = set(self.observations.loc[self.observations.split.eq("train"), "cell_id"])
        evaluation_cells = set(
            self.observations.loc[self.observations.split.isin(("validation", "test")), "cell_id"]
        )
        evaluation_only_cells = evaluation_cells - train_cells
        required_excluded_groups = {cell_groups[cell] for cell in evaluation_only_cells}
        if required_excluded_groups != excluded_groups:
            raise ValueError(
                "Protected VAE identity groups disagree with final evaluation-only cells"
            )
        if set(self.vae_identity_group_ids) & required_excluded_groups:
            raise ValueError("Final evaluation identity is present in offline VAE data")
        if split["configuration"]["protocol"] == "cell":
            identity_splits = pd.DataFrame(
                {
                    "identity": self.observations.cell_id.map(cell_groups),
                    "split": self.observations.split,
                }
            )
            if identity_splits.groupby("identity").split.nunique().max() > 1:
                raise ValueError("A related response identity crosses cold-start splits")
        self._cell_indices = self.observations.cell_id.map(cell_lookup).to_numpy(dtype=np.int64)
        self._drug_a_indices = self.observations.drug_a.map(drug_lookup).to_numpy(dtype=np.int64)
        self._drug_b_indices = self.observations.drug_b.map(drug_lookup).to_numpy(dtype=np.int64)
        self._inhibition = self.observations.inhibition.to_numpy(dtype=np.float32).reshape(-1, 1)
        LOGGER.info(
            "Loaded dataset %s: %d observations, %d cells, %d drugs, fingerprint=%s",
            self.path,
            len(self.observations),
            len(self.cell_ids),
            len(self.drug_ids),
            expected,
        )

    def dataset(self, split: str) -> DoseDataset:
        """Construct an indexed split without copying expression/drug features."""
        return DoseDataset(self, split)
