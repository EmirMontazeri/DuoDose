"""Configuration mistakes fail before running scientific work; CLI remains executable."""

from __future__ import annotations

from pathlib import Path

import pytest

from reverse_dose.cli import artifact_paths, build_parser, main
from reverse_dose.config import config_from_dict, load_config
from reverse_dose.utils import guard_output


@pytest.mark.parametrize(
    "values",
    [
        {"traning": {}},
        {"model": {"latent": 12}},
        {"training": {"learning_rate": -1}},
        {"training": {"learning_rate": float("nan")}},
        {"threads": 0},
        {"seed": True},
        {"seed": "42"},
        {"training": {"epochs": 1.5}},
        {"sources": {"resolve_pubchem": "false"}},
        {"sources": {"aliases": {"KMSII": {"cell_id": "KMS-11", "evidence": ""}}}},
        {"split": {"validation_fraction": 0.9, "test_fraction": 0.2}},
        {"features": {"dose_reference_um": float("inf")}},
        {"model": {"deep_dim": 3}},
        {"vae": {"hidden_dims": "512,256"}},
        {"evaluation": {"inhibition_bin_edges": [1, 0]}},
    ],
)
def test_invalid_configuration_is_rejected(values: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        config_from_dict(values)


def test_effective_config_round_trip_and_distinct_split_paths() -> None:
    config = load_config(Path("configs/v2.yaml"))
    assert config_from_dict(config.to_dict()) == config
    assert config.vae_training.weight_decay == 0.0
    assert config.training.weight_decay == pytest.approx(1e-3)
    assert config.training.early_stopping_patience == 8
    original = artifact_paths(config)
    config.split.seed += 1
    changed = artifact_paths(config)
    assert original[:2] == changed[:2]
    assert original[2] != changed[2]


def test_all_cli_help_is_discoverable(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as captured:
        main(["--help"])
    assert captured.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "inspect",
        "preprocess",
        "prepare",
        "train-vae",
        "train",
        "evaluate",
        "predict",
        "export",
        "run",
        "experiments",
    ):
        assert command in output
    parsed = build_parser().parse_args(
        [
            "predict",
            "--cell",
            "new",
            "--drug-a",
            "A",
            "--drug-b",
            "B",
            "--inhibition",
            ".5",
            "--output",
            "out",
        ]
    )
    assert parsed.inhibition == 0.5


def test_raw_output_guard_resolves_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    raw = root / "raw"
    raw.mkdir(parents=True)
    alias = tmp_path / "looks_processed"
    alias.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match="immutable raw"):
        guard_output(alias / "artifact.json", root)
    guard_output(root / "processed" / "safe", root)


def test_suite_cli_passes_independent_seed_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reverse_dose import experiments

    calls = []

    def fake_run(config, output, splits, trainings, protocols, **kwargs):
        calls.append((splits, trainings, protocols, kwargs["resume"]))
        return {"status": "complete"}

    monkeypatch.setattr(experiments, "run_factorial", fake_run)
    main(
        [
            "experiments",
            "--output",
            str(tmp_path / "suite"),
            "--split-seeds",
            "42",
            "43",
            "--training-seeds",
            "101",
            "102",
            "103",
            "--resume",
        ]
    )
    assert calls == [([42, 43], [101, 102, 103], ["cell"], True)]
