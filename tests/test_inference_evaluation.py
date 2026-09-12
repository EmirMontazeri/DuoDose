"""Scientific inference checks and the complete portable prediction workflow."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from reverse_dose.config import ExperimentConfig, TrainingConfig
from reverse_dose.data import DataBundle, prepare_dataset
from reverse_dose.evaluation import evaluate, load_trained_model, metric_summary
from reverse_dose.inference import Predictor, export_model, write_prediction
from reverse_dose.models import MixtureParameters, ModelConfig, VAEConfig
from reverse_dose.training import train_stage
from test_data import _synthetic_sources


@dataclass
class TrainedExperiment:
    root: Path
    data_dir: Path
    bundle: DataBundle
    config: ExperimentConfig
    vae: Path
    checkpoint: Path


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> TrainedExperiment:
    root = tmp_path_factory.mktemp("trained-inference-fixture")
    source, expression = _synthetic_sources(root)
    data_dir = root / "prepared"
    prepare_dataset(source, expression, data_dir, {}, {"n_genes": 4})
    bundle = DataBundle(data_dir)
    settings = TrainingConfig(
        epochs=2,
        batch_size=32,
        learning_rate=0.002,
        checkpoint_every_steps=100,
        log_every_steps=100,
    )
    config = ExperimentConfig(
        dataset_root=str(root / "dataset"),
        seed=31,
        threads=1,
        device="cpu",
        vae=VAEConfig(hidden_dims=(8,), latent_dim=3),
        model=ModelConfig(
            encoder_dim=4,
            encoder_hidden_dim=6,
            mf_dim=3,
            deep_hidden_dims=(8,),
            deep_dim=6,
            mixture_components=2,
        ),
        vae_training=copy.deepcopy(settings),
        training=copy.deepcopy(settings),
    )
    config.evaluation.batch_size = 7  # Exercise unequal final evaluation batches.
    offline = train_stage(bundle, config, root / "run", "vae")
    checkpoint = train_stage(bundle, config, root / "run", "model", vae_checkpoint=offline)
    return TrainedExperiment(root, data_dir, bundle, config, offline, checkpoint)


def predictor_for(experiment: TrainedExperiment) -> Predictor:
    return Predictor(experiment.data_dir, experiment.checkpoint)


class ConstantGaussian(torch.nn.Module):
    """Analytically specified output used only to test physical density mapping."""

    def __init__(self, mean: tuple[float, float], scales: tuple[float, float], rho: float):
        super().__init__()
        self.mean, self.scales, self.rho = mean, scales, rho

    def forward(self, cell: torch.Tensor, *args: torch.Tensor) -> MixtureParameters:
        count = len(cell)
        return MixtureParameters(
            torch.zeros(count, 1),
            torch.tensor(self.mean).reshape(1, 1, 2).expand(count, -1, -1),
            torch.tensor(self.scales).reshape(1, 1, 2).expand(count, -1, -1),
            torch.full((count, 1), self.rho),
        )


def test_trained_evaluation_reports_observation_and_group_metrics(
    tmp_path: Path,
    experiment: TrainedExperiment,
) -> None:
    output = tmp_path / "evaluation"
    report = evaluate(experiment.bundle, experiment.config, experiment.checkpoint, output)
    predictions = pd.read_parquet(output / "predictions.parquet")
    expected_rows = experiment.bundle.observations.loc[
        experiment.bundle.observations.split.eq("test")
    ]
    assert set(predictions.observation_id) == set(expected_rows.observation_id)
    assert report["overall"]["count"] == len(expected_rows)
    assert report["overall"]["nll"] == pytest.approx(predictions.nll.mean())
    transform = experiment.bundle.dose_transform
    doses = predictions[["dose_a_um", "dose_b_um"]].to_numpy()
    independent_log_jacobian = -np.log(
        (doses[:, 0] + transform.reference_um)
        * (doses[:, 1] + transform.reference_um)
        * transform.scale**2
    )
    np.testing.assert_allclose(predictions.nll_um, predictions.nll - independent_log_jacobian)
    selected = (
        np.abs(predictions.inhibition - experiment.config.evaluation.target_inhibition)
        <= experiment.config.evaluation.target_tolerance
    )
    assert report["near_target"]["count"] == int(selected.sum())
    for filename in (
        "by_cell.csv",
        "by_drug_pair.csv",
        "by_matrix.csv",
        "by_inhibition.csv",
    ):
        grouped = pd.read_csv(output / filename)
        assert grouped["count"].sum() == len(expected_rows)
    with np.load(output / "mixture_parameters.npz") as parameters:
        assert parameters["means"].shape == (len(expected_rows), 2, 2)
        np.testing.assert_allclose(parameters["weights"].sum(-1), 1, atol=1e-6)
    assert json.loads((output / "metrics.json").read_text())["overall"]["count"] == len(
        expected_rows
    )
    with pytest.raises(FileExistsError):
        evaluate(experiment.bundle, experiment.config, experiment.checkpoint, output)
    assert metric_summary(predictions.iloc[:0])["nll"] is None


def test_canonical_request_order_restores_every_coordinate_and_grid(
    experiment: TrainedExperiment,
) -> None:
    predictor = predictor_for(experiment)
    canonical = predictor.distribution("c00", "d0", "d1", 0.5)
    reversed_order = predictor.distribution("c00", "d1", "d0", 0.5)
    for field in ("means", "scales"):
        torch.testing.assert_close(
            getattr(canonical, field),
            getattr(reversed_order, field).flip(-1),
            rtol=0,
            atol=0,
        )
    for field in ("logits", "correlations"):
        torch.testing.assert_close(
            getattr(canonical, field), getattr(reversed_order, field), rtol=0, atol=0
        )
    bounds = ((0.0, 1.0), (0.0, 2.0))
    original, original_grid = predictor.predict("c00", "d0", "d1", 0.5, grid_size=31, bounds=bounds)
    swapped, swapped_grid = predictor.predict(
        "c00", "d1", "d0", 0.5, grid_size=31, bounds=bounds[::-1]
    )
    np.testing.assert_array_equal(
        original["candidate_coordinates_um"],
        swapped["candidate_coordinates_um"][:, ::-1],
    )
    np.testing.assert_array_equal(
        original["point_coordinates_um"], swapped["point_coordinates_um"][::-1]
    )
    np.testing.assert_allclose(
        original_grid["density_per_um2"],
        swapped_grid["density_per_um2"].T,
        rtol=2e-5,
        atol=1e-8,
    )
    assert original["grid"]["approximate_unconditional_mass_in_rectangle"] == pytest.approx(
        swapped["grid"]["approximate_unconditional_mass_in_rectangle"], rel=2e-6
    )


@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_physical_density_and_integrated_mass_match_independent_gaussian(
    experiment: TrainedExperiment,
    rho: float,
) -> None:
    predictor = predictor_for(experiment)
    mean, scales = (0.2, -0.1), (0.6, 0.8)
    predictor.model = ConstantGaussian(mean, scales, rho)
    result, arrays = predictor.predict(
        "c00", "d0", "d1", 0.5, grid_size=301, bounds=((0.0, 1.7), (0.0, 2.3))
    )
    transform = experiment.bundle.dose_transform
    a, b = np.meshgrid(arrays["dose_a_um"], arrays["dose_b_um"], indexing="xy")
    doses = np.stack((a.ravel(), b.ravel()), axis=-1)
    coordinates = (np.log1p(doses / transform.reference_um) - transform.mean) / transform.scale
    covariance = [
        [scales[0] ** 2, rho * scales[0] * scales[1]],
        [rho * scales[0] * scales[1], scales[1] ** 2],
    ]
    jacobian = 1 / (
        (doses[:, 0] + transform.reference_um)
        * (doses[:, 1] + transform.reference_um)
        * transform.scale**2
    )
    expected_density = multivariate_normal(mean=mean, cov=covariance).pdf(coordinates) * jacobian
    np.testing.assert_allclose(
        arrays["density_per_um2"].ravel(), expected_density, rtol=3e-5, atol=1e-7
    )
    lower = coordinates[0]
    upper = coordinates[-1]
    conditional_scale = scales[1] * np.sqrt(1 - rho**2)

    def conditional_mass(first: float) -> float:
        second_mean = mean[1] + rho * scales[1] / scales[0] * (first - mean[0])
        probability = norm.cdf((upper[1] - second_mean) / conditional_scale) - norm.cdf(
            (lower[1] - second_mean) / conditional_scale
        )
        return norm.pdf((first - mean[0]) / scales[0]) / scales[0] * probability

    exact_mass = quad(conditional_mass, lower[0], upper[0], epsabs=1e-11)[0]
    assert result["grid"]["approximate_unconditional_mass_in_rectangle"] == pytest.approx(
        exact_mass, abs=1e-4
    )
    assert exact_mass < 0.99  # Detect unwanted normalization to the requested grid.


def test_new_cell_expression_aligns_ids_aggregates_and_imputes(
    tmp_path: Path,
    experiment: TrainedExperiment,
) -> None:
    predictor = predictor_for(experiment)
    transform = predictor.bundle.expression_transform
    genes = list(transform.gene_ids)
    values = np.array([4.0, np.nan, 8.0, 12.0])
    rows = [
        (genes[0], 2.0),
        (genes[0], 6.0),
        (genes[1], np.nan),
        (genes[2], 8.0),
        (genes[3], 12.0),
        ("unselected-gene", 7.0),
    ]
    pd.DataFrame(rows[::-1], columns=["gene_id", "rsem_tpm"]).to_csv(
        tmp_path / "new_cell.csv", index=False
    )
    transformed = predictor.expression_from_csv(tmp_path / "new_cell.csv")
    expected = transform.transform(values[None], genes)[0]
    np.testing.assert_array_equal(transformed, expected)
    assert transformed[1] == 0.0
    first = predictor.distribution("unseen-cell", "d0", "d1", 0.5, expression=transformed)
    second = predictor.distribution("another-unseen-name", "d0", "d1", 0.5, expression=transformed)
    torch.testing.assert_close(first.means, second.means, rtol=0, atol=0)
    pd.DataFrame({"gene_id": genes, "rsem_tpm": [np.nan] * len(genes)}).to_csv(
        tmp_path / "missing.csv", index=False
    )
    with pytest.raises(ValueError, match="no measured expression"):
        predictor.expression_from_csv(tmp_path / "missing.csv")
    pd.DataFrame({"gene_id": genes[:-1], "rsem_tpm": [1.0] * (len(genes) - 1)}).to_csv(
        tmp_path / "absent_gene.csv", index=False
    )
    with pytest.raises(ValueError, match="required gene IDs"):
        predictor.expression_from_csv(tmp_path / "absent_gene.csv")


def test_invalid_inference_requests_fail_clearly(
    tmp_path: Path,
    experiment: TrainedExperiment,
) -> None:
    predictor = predictor_for(experiment)
    with pytest.raises(ValueError, match="feature representation"):
        predictor.distribution("c00", "unreleased-agent", "d1", 0.5)
    with pytest.raises(ValueError, match="Unknown cell"):
        predictor.distribution("new-cell", "d0", "d1", 0.5)
    with pytest.raises(ValueError, match="inhibition must be finite"):
        predictor.distribution("c00", "d0", "d1", float("nan"))
    with pytest.raises(ValueError, match="wrong dimension"):
        predictor.distribution("new-cell", "d0", "d1", 0.5, expression=np.ones(2))
    with pytest.raises(ValueError, match="Grid bounds"):
        predictor.predict("c00", "d0", "d1", 0.5, grid_size=10, bounds=((-1.0, 2.0), (0.0, 2.0)))
    for value in (-1.0, float("inf")):
        pd.DataFrame(
            {
                "gene_id": predictor.bundle.expression_transform.gene_ids,
                "rsem_tpm": [value] * 4,
            }
        ).to_csv(tmp_path / "invalid.csv", index=False)
        with pytest.raises(ValueError, match="nonnegative TPM"):
            predictor.expression_from_csv(tmp_path / "invalid.csv")
    with pytest.raises(ValueError, match="final-model checkpoint"):
        load_trained_model(experiment.bundle, experiment.config, experiment.vae)
    config = copy.deepcopy(experiment.config)
    config.model = ModelConfig(mixture_components=7)
    with pytest.raises(ValueError, match="model configuration differ"):
        load_trained_model(experiment.bundle, config, experiment.checkpoint)


def test_samples_are_reproducible_and_invalid_support_is_visible(
    experiment: TrainedExperiment,
) -> None:
    predictor = predictor_for(experiment)
    predictor.model = ConstantGaussian((-10.0, -10.0), (0.1, 0.1), 0.2)
    rng_before = torch.get_rng_state().clone()
    first, arrays = predictor.predict("c00", "d0", "d1", 2.0, samples=100, seed=25)
    second, repeated = predictor.predict("c00", "d0", "d1", 2.0, samples=100, seed=25)
    assert torch.equal(rng_before, torch.get_rng_state())
    np.testing.assert_array_equal(arrays["samples_um"], repeated["samples_um"])
    assert not first["point_nonnegative"]
    assert not first["candidate_nonnegative"].any()
    assert first["sampling"]["nonnegative_fraction"] == 0.0
    assert (arrays["samples_um"] < 0).all()
    assert first["target_outside_training_range"]
    assert second["sampling"]["seed"] == 25


def test_portable_export_needs_no_original_sources_and_detects_tampering(
    tmp_path: Path,
    experiment: TrainedExperiment,
) -> None:
    original = predictor_for(experiment).distribution("c00", "d0", "d1", 0.5)
    exported = tmp_path / "portable"
    export_model(
        experiment.data_dir,
        experiment.checkpoint,
        exported,
        dataset_root=Path(experiment.config.dataset_root),
    )
    unavailable = experiment.root.with_name(experiment.root.name + "-unavailable")
    experiment.root.rename(unavailable)
    try:
        restored = Predictor.from_export(exported).distribution("c00", "d0", "d1", 0.5)
    finally:
        unavailable.rename(experiment.root)
    for field in ("logits", "means", "scales", "correlations"):
        torch.testing.assert_close(
            getattr(original, field), getattr(restored, field), rtol=0, atol=0
        )
    feature_file = exported / "data" / "features.npz"
    feature_file.write_bytes(feature_file.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="Export integrity failure"):
        Predictor.from_export(exported)


def test_prediction_writer_creates_standalone_artifacts(
    tmp_path: Path,
    experiment: TrainedExperiment,
) -> None:
    predictor = predictor_for(experiment)
    result, arrays = predictor.predict("c00", "d0", "d1", 0.5, grid_size=21, samples=10)
    output = tmp_path / "prediction"
    write_prediction(output, result, arrays)
    assert (output / "prediction.json").is_file()
    assert (output / "distribution.npz").is_file()
    assert (output / "probability_map.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FileExistsError):
        write_prediction(output, result, arrays)
