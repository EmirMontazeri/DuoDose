"""Dataset adapters producing the same portable training contract independently."""

from __future__ import annotations

from collections import Counter
import importlib.metadata
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .layout import DatasetPaths
from .preprocess import (
    INHIBITION_QUALITY_MAX,
    INHIBITION_QUALITY_MIN,
    INHIBITION_QUALITY_REASON,
    _safe_output,
    response_quality_control,
)
from .screen_metadata import map_screen_cells, screen_chemistry
from .screen_readers import one_file, read_almanac, read_oneil
from .utils import atomic_json, fingerprint, read_json, sha256_file

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "screen-source-v1"
DESIGN_COLUMNS = [
    "matrix_id",
    "cell_id",
    "drug_a",
    "drug_b",
    "dose_a_um",
    "dose_b_um",
    "grid_row",
    "grid_col",
    "observation_id",
]
OUTPUTS = {
    "cells.parquet",
    "drugs.parquet",
    "drugs.npz",
    "observations.parquet",
    "exclusions.parquet",
    "matrix_design.parquet",
}


def implementation_identity() -> dict[str, str]:
    """Pin every local helper that can change native parsing or chemistry selection."""
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "screens.py",
            "screen_metadata.py",
            "screen_readers.py",
            "downloads.py",
            "preprocess.py",
            "layout.py",
            "utils.py",
        )
    }


def _identity(
    root: Path,
    paths: list[Path],
    config: dict[str, Any],
    dataset_name: str,
    reference_name: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "cell_reference_dataset": reference_name,
        "config": config,
        "response_quality_control": response_quality_control(),
        "raw_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(set(paths))
        ],
        "source_inventory": inventory,
        "source_code_sha256": implementation_identity(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "openpyxl", "rdkit")
        },
    }


def _cached(output: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not output.exists():
        return None
    path = output / "source_manifest.json"
    checksum = output / "source_manifest.sha256"
    if (
        not path.is_file()
        or not checksum.is_file()
        or checksum.read_text().strip() != sha256_file(path)
    ):
        raise ValueError(f"Incomplete or corrupt source manifest: {output}")
    manifest = read_json(path)
    if manifest.get("identity_hash") != fingerprint(identity) or any(
        manifest.get(k) != v for k, v in identity.items()
    ):
        raise ValueError(
            f"Existing source artifacts differ from raw/configuration/code; use a new output directory: {output}"
        )
    if set(manifest.get("output_hashes", {})) != OUTPUTS:
        raise ValueError("Incomplete source output hash inventory")
    for name, digest in manifest["output_hashes"].items():
        if not (output / name).is_file() or sha256_file(output / name) != digest:
            raise ValueError(f"Source artifact integrity mismatch: {name}")
    LOGGER.info("Reusing verified %s source cache %s", identity["dataset_name"], output)
    return manifest


def preprocess_screen(
    root: Path,
    config: dict[str, Any],
    output: Path,
    *,
    dataset_name: str,
    reference_dataset_name: str | None = None,
) -> dict[str, Any]:
    """Write one screen atomically, keeping measurements, exclusions and full axes."""
    root, output = root.resolve(), output.resolve()
    _safe_output(root, output)
    layout = DatasetPaths(root, dataset_name)
    reference = DatasetPaths(root, reference_dataset_name or dataset_name)
    source = config["sources"][0]
    path = one_file(
        layout.raw, "ComboDrugGrowth_Nov2017.csv" if source == "nci_almanac" else "*w2lrww.xls"
    )
    relative = path.relative_to(root).as_posix()
    if output.exists():
        manifest_path = output / "source_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"Output exists without a complete source manifest: {output}")
        previous = read_json(manifest_path)
        consumed = [root / entry["path"] for entry in previous.get("raw_files", [])]
        if path not in consumed or any(
            not p.resolve().is_relative_to((root / "raw").resolve()) for p in consumed
        ):
            raise ValueError("Cached source inventory does not match the selected raw dataset")
        identity = _identity(
            root, consumed, config, dataset_name, reference.name, previous["source_inventory"]
        )
        return _cached(output, identity)
    LOGGER.info("Reading independent %s response source %s", source, path)
    native, inventory = (read_almanac if source == "nci_almanac" else read_oneil)(
        path, relative, config
    )
    all_matrices = sorted(native.matrix_id.unique())
    selected = all_matrices
    maximum = config["max_matrices"]
    if maximum is not None and len(all_matrices) > maximum:
        indices = np.linspace(0, len(all_matrices) - 1, maximum, dtype=int)
        selected = [all_matrices[i] for i in indices]
        native = native.loc[native.matrix_id.isin(selected)].reset_index(drop=True)
        LOGGER.warning(
            "Engineering subset: %d/%d replicate-specific matrices",
            len(selected),
            len(all_matrices),
        )
    cells, cell_names, cell_mapping, references = map_screen_cells(
        native.cell_label.unique().tolist(),
        reference.raw,
        config["aliases"],
        config["unresolved_cell_policy"],
    )
    drug_labels = sorted(set(native.drug1) | set(native.drug2))
    drugs, drug_mapping, drug_ids, features, chemistry_sources = screen_chemistry(
        layout.raw,
        source,
        drug_labels,
        config["fingerprint_bits"],
        config["fingerprint_radius"],
        config["resolve_pubchem"],
    )
    identity = _identity(
        root,
        [path, *references, *chemistry_sources],
        config,
        dataset_name,
        reference.name,
        inventory,
    )
    cached = _cached(output, identity)
    if cached is not None:
        return cached
    frames = pd.DataFrame(index=native.index)
    frames["observation_id"], frames["matrix_id"] = native.observation_id, native.matrix_id
    frames["source"], frames["source_file"] = source, relative
    frames["experiment"], frames["cell_id"] = native.experiment, native.cell_label.map(cell_names)
    agent1, agent2 = native.drug1.map(drug_mapping), native.drug2.map(drug_mapping)
    swapped = agent1.gt(agent2)
    frames["drug_a"], frames["drug_b"] = agent1.where(~swapped, agent2), agent2.where(
        ~swapped, agent1
    )
    frames["dose_a_um"] = native.dose1_um.where(~swapped, native.dose2_um)
    frames["dose_b_um"] = native.dose2_um.where(~swapped, native.dose1_um)
    frames["response_percent"] = native.response_percent
    frames["inhibition"] = 1 - native.response_percent / 100
    frames["grid_row"], frames["grid_col"] = native.grid_row, native.grid_col
    frames["matrix_cell_label"], frames["filename_cell_label"] = (
        native.cell_label,
        native.cell_label,
    )
    frames["agent1"], frames["agent2"] = native.drug1, native.drug2
    frames["unit1"] = native["unit1"] if "unit1" in native else "uM"
    frames["unit2"] = native["unit2"] if "unit2" in native else "uM"
    frames["canonical_order_swapped"] = swapped
    for name in ("response_origin", "source_row", "replicate"):
        frames[name] = native[name]
    if frames.observation_id.duplicated().any():
        raise ValueError("Source observation identifiers must be unique")
    reason = pd.Series("", index=frames.index)
    for label, issue in cell_mapping["excluded_labels"].items():
        reason.loc[native.cell_label.eq(label)] = issue["reason"]
    unsupported = ~agent1.isin(drug_ids) | ~agent2.isin(drug_ids)
    if unsupported.any() and config["unsupported_drug_policy"] == "error":
        raise ValueError(
            f"Unsupported drug structures: {sorted(set(agent1[unsupported]) | set(agent2[unsupported]))}"
        )
    reason.loc[reason.eq("") & unsupported] = "unsupported_drug_structure"
    if "valid_source" in native:
        reason.loc[reason.eq("") & ~native.valid_source] = "invalid_source_record"
    missing = ~np.isfinite(frames.response_percent)
    if missing.any() and config["missing_response_policy"] == "error":
        raise ValueError("Missing released response measurements under error policy")
    reason.loc[reason.eq("") & missing] = "missing_response"
    outside = ~frames.inhibition.between(INHIBITION_QUALITY_MIN, INHIBITION_QUALITY_MAX)
    reason.loc[reason.eq("") & outside] = INHIBITION_QUALITY_REASON
    zeros = frames.dose_a_um.eq(0) | frames.dose_b_um.eq(0)
    if not config["include_zero_doses"]:
        reason.loc[reason.eq("") & zeros] = "zero_dose_excluded_by_config"
    accepted = frames.loc[reason.eq("")].copy()
    if accepted.empty:
        raise ValueError("No observations remain after explicit source inclusion policies")
    # Complete axes survive QC and missing-response removal. Only experiments
    # with at least one accepted label enter assessment; no response influences
    # which grid coordinate can be recommended within those experiments.
    design = frames.loc[frames.matrix_id.isin(accepted.matrix_id.unique()), DESIGN_COLUMNS].copy()
    exclusions = frames.loc[
        reason.ne(""),
        [
            "source",
            "source_file",
            "matrix_id",
            "grid_row",
            "grid_col",
            "response_percent",
            "inhibition",
            "matrix_cell_label",
            "agent1",
            "agent2",
            "source_row",
            "replicate",
            "response_origin",
        ],
    ].copy()
    exclusions["reason"] = reason.loc[reason.ne("")]
    exclusions["detail"] = (
        "Released row retained for audit; no response imputation, clipping or identity substitution"
    )
    aliases = {
        native: canonical for native, canonical in drug_mapping.items() if native != canonical
    }
    manifest = {
        **identity,
        "identity_hash": fingerprint(identity),
        "cell_mapping": cell_mapping,
        "drug_alias_evidence": {
            name: {
                "drug_id": canonical,
                "source": (layout.raw / "ComboCompoundNames_small.txt")
                .relative_to(root)
                .as_posix(),
                "method": "exact official NCI released synonym shared with one SDF identifier",
            }
            for name, canonical in aliases.items()
        },
        "observation_schema": list(accepted.columns),
        "exclusion_schema": list(exclusions.columns),
        "response_definition": inventory["response_definition"],
        "response_source_evidence": inventory["response_source_evidence"],
        "dose_unit": "uM",
        "dose_order": "case-sensitive canonical drug IDs; corresponding doses moved together",
        "matrix_design_definition": "Full released dose coordinates per experiment/technical replicate, before missing-response and inhibition QC exclusions",
        "fingerprint_definition": {
            "algorithm": "RDKit Morgan bit vector",
            "bits": config["fingerprint_bits"],
            "radius": config["fingerprint_radius"],
            "includeChirality": False,
            "source": "official NCI SDF or archived exact-name/exact-CID NIH PubChem chain",
        },
        "counts": {
            "source_matrices": {source: len(all_matrices)},
            "parsed_matrices": len(selected),
            "retained_matrices": int(accepted.matrix_id.nunique()),
            "retained_observations": len(accepted),
            "mapped_cells": len(cells),
            "retained_cells": int(accepted.cell_id.nunique()),
            "supported_drugs": len(drug_ids),
            "retained_drugs": len(set(accepted.drug_a) | set(accepted.drug_b)),
            "excluded_by_reason": dict(Counter(exclusions.reason)),
            "response_origins": {
                str(k): int(v) for k, v in accepted.response_origin.value_counts().items()
            },
            "quality_control_excluded_observations": int(
                reason.eq(INHIBITION_QUALITY_REASON).sum()
            ),
            "retained_observations_outside_unit_interval": int(
                (~accepted.inhibition.between(0, 1)).sum()
            ),
        },
    }
    # Dictionary encoding repeated strings also keeps full-screen memory/disk
    # usage practical (ALMANAC has millions of observations).
    for table in (accepted, design, exclusions):
        for column in table.select_dtypes(include=["object", "str"]).columns:
            if column not in {"observation_id", "matrix_id"}:
                table[column] = table[column].astype("category")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for name, table in (
            ("cells", cells),
            ("drugs", drugs),
            ("observations", accepted),
            ("exclusions", exclusions),
            ("matrix_design", design),
        ):
            table.to_parquet(staging / f"{name}.parquet", index=False)
        np.savez_compressed(staging / "drugs.npz", ids=drug_ids, features=features)
        manifest["output_hashes"] = {name: sha256_file(staging / name) for name in sorted(OUTPUTS)}
        atomic_json(staging / "source_manifest.json", manifest)
        (staging / "source_manifest.sha256").write_text(
            sha256_file(staging / "source_manifest.json") + "\n"
        )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    LOGGER.info("Saved independent %s source: %s", dataset_name, manifest["counts"])
    return manifest
