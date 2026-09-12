"""Factorial estimands, interruption semantics and immutable experiment design."""

from pathlib import Path

import numpy as np
import pytest

from reverse_dose.config import ExperimentConfig
from reverse_dose.experiments import run_factorial
from reverse_dose.reporting import summarize_records
from reverse_dose.utils import atomic_json, read_json, sha256_file


def test_factorial_variability_separates_axes_without_pooling_protocols(tmp_path: Path) -> None:
    metrics = [
        "macro_identity_nll_um",
        "macro_identity_mae_um",
        "row_nll",
        "near_target_macro_nll",
        "inhibition_macro_nll",
        "target_macro_hit",
    ]
    records = [
        {"protocol": p, "split_seed": s, "training_seed": t, **{m: offset + s + t for m in metrics}}
        for p, offset in [("cell", 0), ("cell_pair", 100)]
        for s in (0, 10)
        for t in (0, 2)
    ]
    report = summarize_records(records, tmp_path)
    assert set(report["protocols"]) == {"cell", "cell_pair"}
    for protocol in report["protocols"].values():
        assert protocol["fully_crossed"]
        result = protocol["variability"]["row_nll"]
        assert result["sd_of_partition_means"] == pytest.approx(np.sqrt(50))
        assert result["rms_training_sd_within_partition"] == pytest.approx(np.sqrt(2))
        assert result["sd_of_training_seed_means"] == pytest.approx(np.sqrt(2))
        assert result["interaction_rms"] == pytest.approx(0)
    diagonal = [r for r in records if (r["split_seed"], r["training_seed"]) in [(0, 0), (10, 2)]]
    report = summarize_records(diagonal, tmp_path)
    assert not report["protocols"]["cell"]["fully_crossed"]
    assert "sd_of_partition_means" not in report["protocols"]["cell"]["variability"]["row_nll"]


def test_factorial_resume_keeps_seeds_and_rejects_changed_design(
    tmp_path: Path, monkeypatch
) -> None:
    from reverse_dose import cli, experiments

    calls = []
    interrupted = False

    def fake_run(config, output, *, resume):
        nonlocal interrupted
        calls.append((config.split.seed, config.seed, resume))
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "run.json", {"config": config.to_dict()})
        if config.seed == 102 and not interrupted:
            interrupted = True
            return {"status": "interrupted"}
        assessment = output / "evaluation" / "test" / "assessment"
        assessment.mkdir(parents=True, exist_ok=True)
        atomic_json(assessment / "assessment.json", {})
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(experiments, "report_row", lambda report, **kwargs: kwargs)
    monkeypatch.setattr(experiments, "summarize_records", lambda records, output: {})
    config = ExperimentConfig(dataset_root=str(tmp_path / "dataset"))
    output = tmp_path / "factorial"
    result = run_factorial(config, output, [42, 43], [101, 102], ["cell"])
    assert result["status"] == "interrupted"
    assert read_json(output / "progress.json")["completed"] == 1
    with pytest.raises(ValueError, match="identical frozen design"):
        run_factorial(config, output, [42, 43, 44], [101, 102], ["cell"], resume=True)
    result = run_factorial(config, output, [42, 43], [101, 102], ["cell"], resume=True)
    assert result["status"] == "complete"
    assert calls == [
        (42, 101, False),
        (42, 102, False),
        (42, 101, True),
        (42, 102, True),
        (43, 101, False),
        (43, 102, False),
    ]
    assert read_json(output / "progress.json")["completed"] == 4


def test_reference_reuse_applies_frozen_evaluation_settings_and_checks_sources(
    tmp_path: Path, monkeypatch
) -> None:
    from reverse_dose import cli, experiments

    config = ExperimentConfig(dataset_root=str(tmp_path / "dataset"))
    reference = tmp_path / "references" / "run"
    reference.mkdir(parents=True)
    atomic_json(reference / "run.json", {"config": config.to_dict()})
    source_paths = {
        "checkpoint_sha256": reference / "model" / "best.pt",
        "source_prediction_sha256": reference / "evaluation" / "test" / "predictions.parquet",
        "source_parameters_sha256": reference / "evaluation" / "test" / "mixture_parameters.npz",
    }
    for path in source_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    config.evaluation.target_inhibition = 0.6
    calls = []

    def fake_assess(actual, output, *, evaluation_settings):
        calls.append(evaluation_settings.target_inhibition)
        report = {
            "schema_version": "assessment-v1.1",
            "evaluation_settings": vars(evaluation_settings),
            "provenance": {key: sha256_file(path) for key, path in source_paths.items()},
        }
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "assessment.json", report)
        return report

    monkeypatch.setattr(experiments, "assess_saved", fake_assess)
    monkeypatch.setattr(experiments, "report_row", lambda report, **kwargs: kwargs)
    monkeypatch.setattr(experiments, "summarize_records", lambda records, output: {})
    monkeypatch.setattr(
        cli, "run", lambda *args, **kwargs: pytest.fail("Reference should be reused")
    )
    output = tmp_path / "factorial"
    options = {"reference_suite": reference.parent}
    run_factorial(config, output, [42], [42], ["cell"], **options)
    run_factorial(config, output, [42], [42], ["cell"], resume=True, **options)
    assert calls == [0.6]
    source_paths["checkpoint_sha256"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="source artifacts changed"):
        run_factorial(config, output, [42], [42], ["cell"], resume=True, **options)
