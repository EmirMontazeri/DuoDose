"""Behavioral tests for units, preprocessing provenance and leakage isolation."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reverse_dose.data import (
    DataBundle,
    DoseTransform,
    ExpressionTransform,
    extract_expression,
    prepare_dataset,
)


def _refresh_manifest(directory: Path, manifest: str, key: str) -> None:
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.name != manifest
    }
    (directory / manifest).write_text(json.dumps({key: hashes}, sort_keys=True) + "\n")


def _synthetic_sources(tmp_path: Path) -> tuple[Path, Path]:
    source, expression = tmp_path / "source", tmp_path / "expression"
    source.mkdir(parents=True)
    expression.mkdir()
    cells = pd.DataFrame(
        {
            "cell_id": [f"c{i:02}" for i in range(12)],
            "model_id": [f"m{i:02}" for i in range(12)],
            "cosmic_id": [str(100 + i) for i in range(12)],
        }
    )
    cells.to_parquet(source / "cells.parquet", index=False)
    observations = []
    for index, cell in enumerate(cells.cell_id):
        for pair in (("d0", "d1"), ("d0", "d2"), ("d1", "d2")):
            for replicate in range(2):
                matrix_id = f"{cell}-{pair[0]}-{pair[1]}-{replicate}"
                for point, (a, b) in enumerate(((0.0, 0.0), (0.0, 0.1), (0.2, 0.0), (0.2, 0.1))):
                    observations.append(
                        {
                            "observation_id": f"{matrix_id}-{point}",
                            "matrix_id": matrix_id,
                            "cell_id": cell,
                            "drug_a": pair[0],
                            "drug_b": pair[1],
                            "dose_a_um": a * (1 + index),
                            "dose_b_um": b * (1 + index),
                            "inhibition": (index + point + replicate) / 18,
                        }
                    )
    pd.DataFrame(observations).to_parquet(source / "observations.parquet", index=False)
    np.savez_compressed(
        source / "drugs.npz",
        ids=np.array(["d0", "d1", "d2"]),
        features=np.eye(3, dtype=np.float32),
    )
    _refresh_manifest(source, "source_manifest.json", "output_hashes")
    expression_models = [*cells.model_id, "external0", "external1", "external2", "external3"]
    values = np.random.default_rng(189).lognormal(2, 1, size=(16, 8))
    values[0, 2] = np.nan
    values[:, 7] = 0  # A constant feature must never be selected.
    np.savez_compressed(
        expression / "expression.npz",
        model_ids=np.asarray(expression_models),
        identity_group_ids=np.asarray(expression_models),
        gene_ids=np.array([f"g{i}" for i in range(8)]),
        values=values,
    )
    _refresh_manifest(expression, "expression_manifest.json", "artifacts")
    return source, expression


def test_dose_transform_round_trip_swap_and_jacobian() -> None:
    doses = np.array([[0.0, 0.01], [0.1, 1.0], [10.0, 100.0]])
    transform = DoseTransform.fit(doses, reference_um=0.7)
    np.testing.assert_allclose(transform.inverse(transform.transform(doses)), doses, atol=1e-12)
    np.testing.assert_allclose(
        transform.transform(doses[:, ::-1]), transform.transform(doses)[:, ::-1]
    )
    point = np.array([[0.3, 1.2]])
    epsilon = 1e-6
    derivatives = (transform.transform(point + epsilon) - transform.transform(point - epsilon)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(
        transform.log_abs_det_jacobian(point),
        np.log(derivatives).sum(axis=-1),
        rtol=1e-8,
    )
    assert transform.inverse(np.array([-100.0]))[0] < 0  # No silent clipping of MDN support.
    with pytest.raises(ValueError, match="nonnegative"):
        transform.transform(np.array([-1.0]))
    with pytest.raises(FloatingPointError):
        transform.inverse(np.array([1e200]))
    with pytest.raises(ValueError, match="no variation"):
        DoseTransform.fit(np.zeros((3, 2)))


def test_expression_transform_order_missing_and_serialization(tmp_path: Path) -> None:
    raw = np.array([[0.0, 1.0, np.nan, 1.0], [2.0, 2.0, 0.0, 1.0], [7.0, 3.0, 2.0, 1.0]])
    genes = ["g0", "g1", "g2", "constant"]
    transform = ExpressionTransform.fit(raw, genes, n_genes=20)
    assert len(transform.gene_ids) == 3
    assert "constant" not in transform.gene_ids
    transformed = transform.transform(raw, genes)
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(transformed.std(axis=0), 1, atol=1e-7)
    np.testing.assert_array_equal(transform.transform(raw[:, ::-1], genes[::-1]), transformed)
    transform.save(tmp_path / "transform.npz")
    restored = ExpressionTransform.load(tmp_path / "transform.npz")
    np.testing.assert_array_equal(restored.transform(raw, genes), transformed)
    with pytest.raises(ValueError, match="required gene IDs"):
        transform.transform(raw[:, 1:], genes[1:])
    with pytest.raises(ValueError, match="no measured expression"):
        transform.transform(np.full(4, np.nan), genes)
    with pytest.raises(ValueError, match="duplicate"):
        transform.transform(raw, ["g0", "g0", "g2", "constant"])


def test_streamed_expression_aggregation_and_raw_immutability(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    raw_dir = dataset / "raw" / "cell_model_passports"
    raw_dir.mkdir(parents=True)
    archive = raw_dir / "rnaseq_all_test.zip"
    pd.DataFrame(
        {
            "model_id": ["m0", "m1", "other"],
            "RNASeq Broad Cell Lines": [True, False, True],
            "RNASeq Sanger Cell Lines": [False, True, False],
        }
    ).to_csv(raw_dir / "model_dataset_availability_test.csv", index=False)
    pd.DataFrame(
        {
            "model_id": ["m0", "m1", "other"],
            "model_name": ["cell-0", "cell-1", "external"],
            "sample_id": ["s0", "s1", "s2"],
            # The external profile shares a patient identity with m1 and must
            # therefore receive the same leakage-control group.
            "patient_id": ["p0", "p1", "p1"],
        }
    ).to_csv(raw_dir / "model_list_test.csv", index=False)
    csv = (
        "model_id,gene_id,rsem_tpm,dataset_id,data_source,duplicate\n"
        "m0,g0,2,1,Sanger,f\nm1,g0,5,1,Sanger,f\nm0,g0,4,2,Broad,t\n"
        "m0,g1,,1,Sanger,f\nm1,g1,0,1,Sanger,f\nm1,g2,-1,1,Sanger,f\n"
        "m1,g2,inf,1,Sanger,f\nother,g9,17,1,Sanger,f\n"
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zfile:
        zfile.writestr("rnaseq.csv", csv)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    cells = pd.DataFrame({"cell_id": ["c0", "c1"], "model_id": ["m0", "m1"]})
    path = extract_expression(dataset, cells, tmp_path / "expression1", chunk_size=2)
    path2 = extract_expression(dataset, cells, tmp_path / "expression2", chunk_size=3)
    with np.load(path) as values:
        assert values["model_ids"].tolist() == ["m0", "m1", "other"]
        assert values["gene_ids"].tolist() == ["g0", "g1", "g2", "g9"]
        np.testing.assert_allclose(
            values["values"],
            [[3, np.nan, np.nan, np.nan], [5, 0, np.nan, np.nan], [np.nan, np.nan, np.nan, 17]],
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            values["measurement_counts"],
            [[2, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
        )
        assert values["identity_group_ids"].tolist() == ["m0", "m1", "m1"]
    assert path.read_bytes() == path2.read_bytes()
    manifest = json.loads((path.parent / "expression_manifest.json").read_text())
    assert manifest["record_counts"]["negative_tpm"] == 1
    assert manifest["record_counts"]["infinite_tpm"] == 1
    assert manifest["record_counts"]["archive_rows"] == 8
    assert manifest["multiple_measurement_model_gene_values"] == 1
    assert before == hashlib.sha256(archive.read_bytes()).hexdigest()
    mtime = path.stat().st_mtime_ns
    assert extract_expression(dataset, cells, path.parent, chunk_size=1) == path
    assert path.stat().st_mtime_ns == mtime
    assert extract_expression(dataset, cells.iloc[:1], path.parent) == path
    unavailable = pd.DataFrame({"cell_id": ["missing"], "model_id": ["missing"]})
    with pytest.raises(ValueError, match="lack released cell-line RNA-seq"):
        extract_expression(dataset, unavailable, path.parent)
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        extract_expression(dataset, cells, path.parent)
    with pytest.raises(ValueError, match="dataset/raw"):
        extract_expression(dataset, cells, raw_dir / "forbidden")


def test_cell_splits_all_replicates_and_portable_bundle(tmp_path: Path) -> None:
    source, expression = _synthetic_sources(tmp_path)
    manifest = prepare_dataset(source, expression, tmp_path / "prepared", {}, {"n_genes": 4})
    again = prepare_dataset(source, expression, tmp_path / "again", {}, {"n_genes": 4})
    assert manifest["fingerprint"] == again["fingerprint"]
    cached = prepare_dataset(source, expression, tmp_path / "prepared", {}, {"n_genes": 4})
    assert cached["fingerprint"] == manifest["fingerprint"]
    with pytest.raises(ValueError, match="differ from inputs"):
        prepare_dataset(source, expression, tmp_path / "prepared", {"seed": 43}, {"n_genes": 4})
    bundle = DataBundle(tmp_path / "prepared")
    rows = bundle.observations
    assert rows.groupby("cell_id").split.nunique().max() == 1
    assert rows.groupby("matrix_id").split.nunique().max() == 1
    train_cells = set(rows.loc[rows.split.eq("train"), "cell_id"])
    fit_models = {bundle.vae_model_ids[index] for index in bundle.vae_train_indices}
    vae_val_models = {bundle.vae_model_ids[index] for index in bundle.vae_validation_indices}
    assert fit_models.isdisjoint(vae_val_models)
    assert fit_models | vae_val_models == set(bundle.vae_model_ids)
    response_cells = pd.read_parquet(source / "cells.parquet")
    heldout_models = set(response_cells.loc[~response_cells.cell_id.isin(train_cells), "model_id"])
    assert set(bundle.vae_model_ids).isdisjoint(heldout_models)
    assert {"external0", "external1", "external2", "external3"} <= set(bundle.vae_model_ids)
    assert manifest["features"]["expression_fit_models"] == sorted(fit_models)
    item = bundle.dataset("test")[0]
    assert item["cell"].shape == (4,)
    assert item["target"].shape == (2,)
    assert item["inhibition"].shape == (1,)
    assert item["drug_a"].shape == (3,)
    row = rows.iloc[int(item["index"])]
    assert row.split == "test"
    np.testing.assert_allclose(
        bundle.dose_transform.inverse(item["target"].numpy()),
        [row.dose_a_um, row.dose_b_um],
        atol=1e-6,
    )
    shutil.copytree(tmp_path / "prepared", tmp_path / "portable")
    shutil.rmtree(source)
    shutil.rmtree(expression)
    assert DataBundle(tmp_path / "portable").manifest["fingerprint"] == manifest["fingerprint"]


def test_heldout_expression_and_doses_cannot_affect_fitted_transforms(
    tmp_path: Path,
) -> None:
    source, expression = _synthetic_sources(tmp_path)
    prepare_dataset(source, expression, tmp_path / "before", {}, {"n_genes": 4})
    original = DataBundle(tmp_path / "before")
    fit_models = set(original.manifest["features"]["expression_fit_models"])
    with np.load(expression / "expression.npz") as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for index, model in enumerate(arrays["model_ids"]):
        if model not in fit_models:
            arrays["values"][index] = 1e18
    np.savez_compressed(expression / "expression.npz", **arrays)
    _refresh_manifest(expression, "expression_manifest.json", "artifacts")
    observations = pd.read_parquet(source / "observations.parquet")
    heldout = original.observations.split.ne("train").to_numpy()
    observations.loc[heldout, ["dose_a_um", "dose_b_um"]] = 1e15
    observations.to_parquet(source / "observations.parquet", index=False)
    _refresh_manifest(source, "source_manifest.json", "output_hashes")
    prepare_dataset(source, expression, tmp_path / "after", {}, {"n_genes": 4})
    changed = DataBundle(tmp_path / "after")
    assert changed.expression_transform.gene_ids == original.expression_transform.gene_ids
    np.testing.assert_array_equal(
        changed.expression_transform.mean, original.expression_transform.mean
    )
    np.testing.assert_array_equal(
        changed.expression_transform.scale, original.expression_transform.scale
    )
    assert changed.dose_transform == original.dose_transform
    assert changed.manifest["fingerprint"] != original.manifest["fingerprint"]


def test_final_holdout_identity_cannot_enter_or_change_offline_vae(
    tmp_path: Path,
) -> None:
    source, expression = _synthetic_sources(tmp_path)
    prepare_dataset(source, expression, tmp_path / "initial", {}, {"n_genes": 4})
    initial = DataBundle(tmp_path / "initial")
    cells = pd.read_parquet(source / "cells.parquet")
    evaluation_cells = set(initial.manifest["split"]["cells"]["validation"]) | set(
        initial.manifest["split"]["cells"]["test"]
    )
    heldout_models = set(cells.loc[cells.cell_id.isin(evaluation_cells), "model_id"])

    with np.load(expression / "expression.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for index, model in enumerate(arrays["model_ids"]):
        if model in heldout_models:
            arrays["values"][index] = 1e18
    np.savez_compressed(expression / "expression.npz", **arrays)
    _refresh_manifest(expression, "expression_manifest.json", "artifacts")
    prepare_dataset(source, expression, tmp_path / "perturbed", {}, {"n_genes": 4})
    perturbed = DataBundle(tmp_path / "perturbed")

    assert set(initial.vae_model_ids).isdisjoint(heldout_models)
    assert initial.vae_model_ids == perturbed.vae_model_ids
    np.testing.assert_array_equal(initial.vae_features, perturbed.vae_features)
    assert initial.expression_transform.gene_ids == perturbed.expression_transform.gene_ids
    np.testing.assert_array_equal(
        initial.expression_transform.mean, perturbed.expression_transform.mean
    )
    np.testing.assert_array_equal(
        initial.expression_transform.scale, perturbed.expression_transform.scale
    )


def test_related_external_models_are_excluded_from_offline_vae(
    tmp_path: Path,
) -> None:
    source, expression = _synthetic_sources(tmp_path)
    prepare_dataset(source, expression, tmp_path / "reference", {}, {"n_genes": 4})
    reference = DataBundle(tmp_path / "reference")
    evaluation_cell = reference.manifest["split"]["cells"]["test"][0]
    cells = pd.read_parquet(source / "cells.parquet")
    heldout_model = cells.set_index("cell_id").loc[evaluation_cell, "model_id"]

    with np.load(expression / "expression.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    model_lookup = {model: index for index, model in enumerate(arrays["model_ids"])}
    heldout_group = arrays["identity_group_ids"][model_lookup[heldout_model]]
    arrays["identity_group_ids"][model_lookup["external0"]] = heldout_group
    np.savez_compressed(expression / "expression.npz", **arrays)
    _refresh_manifest(expression, "expression_manifest.json", "artifacts")

    prepare_dataset(source, expression, tmp_path / "grouped", {}, {"n_genes": 4})
    grouped = DataBundle(tmp_path / "grouped")
    assert "external0" not in grouped.vae_model_ids
    assert heldout_model not in grouped.vae_model_ids


def test_related_response_models_stay_in_one_cold_split(tmp_path: Path) -> None:
    source, expression = _synthetic_sources(tmp_path)
    with np.load(expression / "expression.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    model_lookup = {model: index for index, model in enumerate(arrays["model_ids"])}
    arrays["identity_group_ids"][model_lookup["m01"]] = arrays["identity_group_ids"][
        model_lookup["m00"]
    ]
    np.savez_compressed(expression / "expression.npz", **arrays)
    _refresh_manifest(expression, "expression_manifest.json", "artifacts")
    prepare_dataset(source, expression, tmp_path / "grouped", {}, {"n_genes": 4})
    grouped = DataBundle(tmp_path / "grouped")
    split_by_cell = grouped.observations.groupby("cell_id").split.first()
    assert split_by_cell["c00"] == split_by_cell["c01"]


def test_warm_split_keeps_reversed_pairs_and_replicates_together(
    tmp_path: Path,
) -> None:
    source, expression = _synthetic_sources(tmp_path)
    rows = pd.read_parquet(source / "observations.parquet")
    reverse = rows.index % 2 == 0
    rows.loc[reverse, ["drug_a", "drug_b"]] = rows.loc[reverse, ["drug_b", "drug_a"]].to_numpy()
    rows.loc[reverse, ["dose_a_um", "dose_b_um"]] = rows.loc[
        reverse, ["dose_b_um", "dose_a_um"]
    ].to_numpy()
    rows.to_parquet(source / "observations.parquet", index=False)
    _refresh_manifest(source, "source_manifest.json", "output_hashes")
    prepare_dataset(
        source,
        expression,
        tmp_path / "prepared",
        {"protocol": "cell_pair"},
        {"n_genes": 4},
    )
    bundle = DataBundle(tmp_path / "prepared")
    result = bundle.observations
    result["pair"] = [
        tuple(sorted((a, b))) for a, b in zip(result.drug_a, result.drug_b, strict=True)
    ]
    assert result.groupby(["cell_id", "pair"]).split.nunique().max() == 1
    assert result.groupby("matrix_id").split.nunique().max() == 1
    assert any(bundle.manifest["split"]["cell_overlap_with_train"].values())


def test_bundle_rejects_tampered_artifact(tmp_path: Path) -> None:
    source, expression = _synthetic_sources(tmp_path)
    prepare_dataset(source, expression, tmp_path / "prepared", {}, {"n_genes": 4})
    with (tmp_path / "prepared" / "features.npz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="integrity mismatch"):
        DataBundle(tmp_path / "prepared")


def test_preparation_rejects_tampered_source_before_fitting(tmp_path: Path) -> None:
    source, expression = _synthetic_sources(tmp_path)
    rows = pd.read_parquet(source / "observations.parquet")
    rows.loc[0, "inhibition"] = 7.0
    rows.to_parquet(source / "observations.parquet", index=False)
    with pytest.raises(ValueError, match="Source artifact integrity mismatch"):
        prepare_dataset(source, expression, tmp_path / "prepared", {}, {})
