"""CPU experiment continuity, lineage, and metadata-only selection tests."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from reverse_dose import training
from reverse_dose.config import ExperimentConfig, TrainingConfig
from reverse_dose.models import ModelConfig, VAEConfig, vae_loss
from reverse_dose.utils import load_checkpoint, sha256_file


class SyntheticResponses(Dataset):
    """Small measured response rows with disjoint final validation cells."""

    def __init__(
        self, cell_features: np.ndarray, drug_features: np.ndarray, split: str
    ):
        self.cells = torch.from_numpy(cell_features)
        self.drugs = torch.from_numpy(drug_features)
        self.cell_ids = list(range(6)) * 2 + [0] if split == "train" else [6] * 5

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "cell": self.cells[self.cell_ids[index]],
            "drug_a": self.drugs[index % 3],
            "drug_b": self.drugs[(index + 1) % 3],
            "inhibition": torch.tensor([0.1 + 0.05 * index], dtype=torch.float32),
            "target": torch.tensor(
                [0.1 * index, 0.05 * index + 0.2], dtype=torch.float32
            ),
        }


@dataclass
class SyntheticBundle:
    """DataBundle's public training interface, independent of source parsing."""

    cell_features: np.ndarray
    drug_features: np.ndarray
    vae_train_indices: np.ndarray
    vae_validation_indices: np.ndarray
    manifest: dict[str, str]

    def dataset(self, split: str) -> SyntheticResponses:
        assert split in {"train", "validation"}
        return SyntheticResponses(self.cell_features, self.drug_features, split)


@pytest.fixture
def bundle() -> SyntheticBundle:
    rng = np.random.default_rng(100)
    return SyntheticBundle(
        rng.normal(size=(8, 6)).astype(np.float32),
        rng.normal(size=(3, 4)).astype(np.float32),
        np.array([0, 1, 2, 3, 4]),
        np.array([5]),
        {"fingerprint": "synthetic-training-fixture-v1"},
    )


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    stage = TrainingConfig(
        epochs=6,
        batch_size=3,
        learning_rate=0.002,
        early_stopping_patience=3,
        scheduler_patience=1,
        checkpoint_every_steps=1,
        checkpoint_every_epochs=1,
        log_every_steps=1,
    )
    return ExperimentConfig(
        dataset_root=str(tmp_path / "dataset"),
        seed=29,
        device="cpu",
        threads=1,
        vae=VAEConfig(hidden_dims=(8,), latent_dim=3),
        model=ModelConfig(
            encoder_dim=4,
            encoder_hidden_dim=6,
            mf_dim=3,
            deep_hidden_dims=(8,),
            deep_dim=6,
            mixture_components=2,
        ),
        vae_training=copy.deepcopy(stage),
        training=copy.deepcopy(stage),
    )


@pytest.fixture(autouse=True)
def stable_source_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # Other agents may write source modules during the test session. Capture a
    # real snapshot once per test; explicit drift tests replace it independently.
    snapshot = training.environment_info()
    monkeypatch.setattr(training, "environment_info", lambda: copy.deepcopy(snapshot))


def assert_exact_nested(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_exact_nested(first, second)
    else:
        assert left == right


@pytest.mark.parametrize(
    ("stage", "interruption_step"), [("vae", 1), ("vae", 3), ("model", 1), ("model", 7)]
)
def test_interruption_restores_exact_experiment(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    interruption_step: int,
) -> None:
    # A constant validation sequence exercises LR decay, retaining an earlier
    # selected model, and early stopping, independently of incidental convergence.
    monkeypatch.setattr(training, "_validate", lambda *args, **kwargs: {"loss": 1.0})
    offline = None
    if stage == "model":
        offline = training.train_stage(bundle, config, tmp_path / "offline", "vae")
    reference_dir, interrupted_dir = tmp_path / "reference", tmp_path / "resumed"
    reference_best = training.train_stage(
        bundle, config, reference_dir, stage, vae_checkpoint=offline
    )
    interrupted = training.train_stage(
        bundle,
        config,
        interrupted_dir,
        stage,
        vae_checkpoint=offline,
        stop_after_steps=interruption_step,
    )
    checkpoint = load_checkpoint(interrupted)
    assert checkpoint["progress"]["global_step"] == interruption_step
    assert not checkpoint["progress"]["completed"]
    resumed_best = training.train_stage(
        bundle,
        config,
        interrupted_dir,
        stage,
        vae_checkpoint=offline,
        resume=interrupted,
    )
    reference_last = load_checkpoint(reference_dir / stage / "last.pt")
    resumed_last = load_checkpoint(interrupted_dir / stage / "last.pt")
    for field in (
        "experiment_id",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "progress",
        "rng_state",
        "sampler_generator_state",
    ):
        assert_exact_nested(reference_last[field], resumed_last[field])
    assert_exact_nested(load_checkpoint(reference_best), load_checkpoint(resumed_best))
    assert reference_last["progress"]["epoch"] == 4
    assert reference_last["progress"]["best_epoch"] == 1
    assert (
        reference_last["optimizer_state"]["param_groups"][0]["lr"]
        < config.training.learning_rate
    )
    selected = load_checkpoint(resumed_best)
    assert_exact_nested(
        selected["model_state"], resumed_last["progress"]["best_model_state"]
    )
    if stage == "model":
        pretrained = load_checkpoint(offline)["model_state"]
        for name, value in resumed_last["model_state"].items():
            if name.startswith("vae_encoder."):
                assert_exact_nested(
                    value, pretrained["encoder." + name.removeprefix("vae_encoder.")]
                )


@pytest.mark.parametrize("mismatch", ["config", "dataset", "source"])
def test_resume_rejects_changed_identity_before_writing(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    run = tmp_path / "run"
    last = training.train_stage(bundle, config, run, "vae", stop_after_steps=1)
    files_before = {path.name: sha256_file(path) for path in (run / "vae").iterdir()}
    if mismatch == "config":
        config.vae_training.learning_rate *= 2
    elif mismatch == "dataset":
        bundle.manifest["fingerprint"] = "another-dataset"
    else:
        changed = training.environment_info()
        changed["source_hashes"]["training.py"] = "modified-code"
        monkeypatch.setattr(training, "environment_info", lambda: changed)
    with pytest.raises(ValueError, match="Resume rejected"):
        training.train_stage(bundle, config, run, "vae", resume=last)
    assert files_before == {
        path.name: sha256_file(path) for path in (run / "vae").iterdir()
    }


def test_selected_checkpoint_resume_rejected_without_artifact_changes(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
) -> None:
    run = tmp_path / "run"
    selected = training.train_stage(bundle, config, run, "vae")
    files_before = {path.name: sha256_file(path) for path in (run / "vae").iterdir()}
    with pytest.raises(ValueError, match="selection/inference"):
        training.train_stage(bundle, config, run, "vae", resume=selected)
    assert files_before == {
        path.name: sha256_file(path) for path in (run / "vae").iterdir()
    }


def test_resume_cannot_overwrite_another_experiment_directory(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
) -> None:
    last = training.train_stage(
        bundle, config, tmp_path / "source", "vae", stop_after_steps=1
    )
    destination = tmp_path / "destination" / "vae"
    destination.mkdir(parents=True)
    metadata = destination / "experiment.json"
    metadata.write_text(json.dumps({"experiment_id": "different-experiment"}))
    before = metadata.read_bytes()
    with pytest.raises(ValueError, match="destination belongs"):
        training.train_stage(bundle, config, destination.parent, "vae", resume=last)
    assert metadata.read_bytes() == before
    assert list(destination.iterdir()) == [metadata]


def test_completed_resume_restores_authoritative_selected_weights(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
) -> None:
    run = tmp_path / "run"
    selected = training.train_stage(bundle, config, run, "vae")
    authoritative = load_checkpoint(run / "vae" / "last.pt")
    selected.write_bytes(b"stale-or-incomplete-selected-artifact")
    recovered = training.train_stage(
        bundle, config, run, "vae", resume=run / "vae" / "last.pt"
    )
    assert recovered == selected
    assert_exact_nested(
        load_checkpoint(recovered)["model_state"],
        authoritative["progress"]["best_model_state"],
    )


def test_vae_validation_uses_only_metadata_and_posterior_mean(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_response_access(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Offline VAE must not load response datasets")

    monkeypatch.setattr(bundle, "dataset", no_response_access)
    selected = training.train_stage(bundle, config, tmp_path / "run", "vae")
    checkpoint = load_checkpoint(selected)
    model = training.make_model(bundle, config, "vae")
    model.load_state_dict(checkpoint["model_state"])
    data = training.MetadataDataset(bundle.cell_features, bundle.vae_validation_indices)
    state_before = torch.get_rng_state().clone()
    metrics = training._validate(
        model, data, config.vae_training, torch.device("cpu"), "vae", config.vae_beta
    )
    assert torch.equal(state_before, torch.get_rng_state())
    x = torch.from_numpy(bundle.cell_features[bundle.vae_validation_indices])
    with torch.no_grad():
        mean, logvar = model.encoder(x)
        expected = vae_loss(model.decoder(mean), x, mean, logvar, beta=config.vae_beta)
    for name, value in expected.items():
        assert metrics[name] == pytest.approx(value.item())
    assert checkpoint["progress"]["best_validation"] == pytest.approx(metrics["loss"])


def test_offline_encoder_transfer_rejects_unselected_or_wrong_lineage(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
) -> None:
    selected = training.train_stage(bundle, config, tmp_path / "run", "vae")
    with pytest.raises(ValueError, match="selected best.pt"):
        training.make_model(bundle, config, "model", selected.parent / "last.pt")
    bundle.manifest["fingerprint"] = "different-preprocessing"
    with pytest.raises(ValueError, match="different stage or prepared dataset"):
        training.make_model(bundle, config, "model", selected)


@pytest.mark.parametrize("reconstruct_selection", [False, True])
def test_completed_offline_resume_preserves_final_experiment_identity(
    tmp_path: Path,
    bundle: SyntheticBundle,
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    reconstruct_selection: bool,
) -> None:
    monkeypatch.setattr(training, "_validate", lambda *args, **kwargs: {"loss": 1.0})
    run = tmp_path / "run"
    selected_vae = training.train_stage(bundle, config, run, "vae")
    original_file_hash = sha256_file(selected_vae)
    original_encoder_identity = training._encoder_fingerprint(
        load_checkpoint(selected_vae)
    )
    interrupted_model = training.train_stage(
        bundle, config, run, "model", vae_checkpoint=selected_vae, stop_after_steps=1
    )
    original_experiment = load_checkpoint(interrupted_model)["experiment_id"]
    if reconstruct_selection:
        selected_vae.unlink()
    restored_vae = training.train_stage(
        bundle, config, run, "vae", resume=run / "vae" / "last.pt"
    )
    assert (
        training._encoder_fingerprint(load_checkpoint(restored_vae))
        == original_encoder_identity
    )
    if reconstruct_selection:
        # The reconstructed artifact carries later progress/optimizer counters.
        assert sha256_file(restored_vae) != original_file_hash
    else:
        assert sha256_file(restored_vae) == original_file_hash
    resumed_model = training.train_stage(
        bundle,
        config,
        run,
        "model",
        vae_checkpoint=restored_vae,
        resume=interrupted_model,
    )
    assert load_checkpoint(resumed_model)["experiment_id"] == original_experiment
