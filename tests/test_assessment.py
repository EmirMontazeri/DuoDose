"""Leakage boundaries and statistical estimands, independent of learned fit quality."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import ndtr

from reverse_dose.assessment import (
    cluster_interval,
    density_rows,
    matrix_design,
    score_target_grid,
    select_target_grid,
    target_window_mask,
)
from reverse_dose.config import EvaluationConfig
from reverse_dose.data import DoseTransform
from reverse_dose.experiments import factorial_design
from reverse_dose.models import MixtureParameters
from reverse_dose.utils import atomic_json, sha256_file
from test_preprocess import write_matrix


def test_cluster_bootstrap_does_not_treat_repeated_rows_as_independent() -> None:
    frame = pd.DataFrame({"identity_group": ["a", "b", "c"], "nll": [0.0, 1.0, 5.0]})
    settings = EvaluationConfig()
    original = cluster_interval(frame, "nll", settings)
    repeated = pd.concat([frame] * 1000, ignore_index=True)
    again = cluster_interval(repeated, "nll", settings)
    for name in ("estimate", "identities", "lower", "upper"):
        assert original[name] == again[name]
    assert original["identities"] == 3
    assert cluster_interval(frame.iloc[:1], "nll", settings)["lower"] is None
    assert cluster_interval(frame.iloc[:0], "nll", settings)["estimate"] is None


def test_micro_macro_cluster_estimands_differ_for_unequal_support() -> None:
    frame = pd.DataFrame({"identity_group": ["a"] * 9 + ["b"], "x": [0.0] * 9 + [10.0]})
    assert cluster_interval(frame, "x", EvaluationConfig())["estimate"] == 5
    assert cluster_interval(frame, "x", EvaluationConfig(), macro=False)["estimate"] == 1


def test_rosenblatt_transform_and_covariance_match_analytic_gaussian() -> None:
    parameters = {
        "weights": np.ones((2, 1)),
        "means": np.zeros((2, 1, 2)),
        "scales": np.ones((2, 1, 2)),
        "correlations": np.full((2, 1), 0.8),
    }
    target = np.array([[1.0, 0.8], [0.0, 0.6]])
    result = density_rows(parameters, target)
    np.testing.assert_allclose(result.pit_a, ndtr([1, 0]))
    np.testing.assert_allclose(result.pit_b_given_a, ndtr([0, 1]))
    np.testing.assert_allclose(result.minimum_covariance_eigenvalue, 0.2)
    np.testing.assert_allclose(result.weighted_log_determinant, np.log(0.36))


class TargetModel(torch.nn.Module):
    def forward(self, cell, inhibition, a, b):
        assert torch.all(inhibition == 0.5)  # Only fixed target, never measured response.
        n = len(cell)
        return MixtureParameters(
            torch.zeros(n, 1), torch.full((n, 1, 2), 0.7), torch.ones(n, 1, 2), torch.zeros(n, 1)
        )


def design_and_bundle():
    design = pd.DataFrame(
        [
            {
                "matrix_id": "m",
                "cell_id": "c",
                "drug_a": "a",
                "drug_b": "b",
                "dose_a_um": float(a),
                "dose_b_um": float(b),
                "grid_row": a + 1,
                "grid_col": b + 1,
                "observation_id": f"m:{a+1}:{b+1}",
            }
            for a in (0, 1)
            for b in (0, 1)
        ]
    )
    bundle = SimpleNamespace(
        cell_ids=["c"],
        drug_ids=["a", "b"],
        cell_features=np.zeros((1, 2), dtype=np.float32),
        drug_features=np.zeros((2, 2), dtype=np.float32),
        dose_transform=DoseTransform(1.0, 0.0, 1.0),
    )
    return design, bundle


def test_target_selection_is_response_blind_and_unscorable_is_not_reselected() -> None:
    design, bundle = design_and_bundle()
    settings = EvaluationConfig()
    selected = select_target_grid(TargetModel(), bundle, design, settings, torch.device("cpu"))
    assert selected.observation_id.tolist() == ["m:2:2"]
    selected["identity_group"] = "c"
    # The only selected coordinate is missing/QC excluded. The evaluator must
    # not silently fall back to the perfect response at another grid coordinate.
    responses = pd.DataFrame({"observation_id": ["m:1:1"], "matrix_id": ["m"], "inhibition": [0.5]})
    scored, report = score_target_grid(selected, responses, settings)
    assert report["unscorable_matrices"] == 1
    assert report["hit_rate_scorable"] is None
    assert report["all_matrix_hit_rate_bounds"] == [0.0, 1.0]
    assert scored.observation_id.tolist() == ["m:2:2"]
    for response in (-2.0, 0.0, 0.5, 2.0):
        responses["inhibition"] = response
        repeated = select_target_grid(TargetModel(), bundle, design, settings, torch.device("cpu"))
        assert repeated.observation_id.tolist() == ["m:2:2"]
    with pytest.raises(ValueError, match="response-blind"):
        select_target_grid(
            TargetModel(), bundle, design.assign(inhibition=0.5), settings, torch.device("cpu")
        )


def test_target_hit_tolerance_and_denominator_include_unattainable_matrices() -> None:
    design, bundle = design_and_bundle()
    selected = select_target_grid(
        TargetModel(), bundle, design, EvaluationConfig(), torch.device("cpu")
    )
    selected["identity_group"] = "c"
    responses = pd.DataFrame({"observation_id": ["m:2:2"], "matrix_id": ["m"], "inhibition": [0.8]})
    _, report = score_target_grid(selected, responses, EvaluationConfig())
    assert report["matrices"] == 1
    assert report["hit_rate_scorable"] == 0
    assert report["available_target_matrix_fraction"] == 0


def test_factorial_design_crosses_every_split_and_training_seed() -> None:
    design = factorial_design([42, 43, 44, 45, 46], [101, 102], ["cell", "cell_pair"])
    assert len(design) == 20
    assert {(x["split_seed"], x["training_seed"]) for x in design} == {
        (s, t) for s in range(42, 47) for t in (101, 102)
    }
    with pytest.raises(ValueError):
        factorial_design([42, 42], [1], ["cell"])


def test_original_matrix_axes_include_missing_and_qc_excluded_candidates(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    raw = root / "raw" / "matrix.xlsx"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    # Only one retained response row is present. All four ORIGINAL coordinates
    # must remain candidates, even when the last raw response is missing or QC bad.
    bundle = SimpleNamespace(
        path=prepared,
        observations=pd.DataFrame(
            [
                {
                    "matrix_id": "m",
                    "source_file": "raw/matrix.xlsx",
                    "split": "test",
                    "canonical_order_swapped": True,
                    "cell_id": "c",
                    "drug_a": "A",
                    "drug_b": "B",
                }
            ]
        ),
    )
    designs = []
    for response in (None, -9999.0, 50.0):
        write_matrix(raw, response=response)
        atomic_json(
            prepared / "source_manifest.json",
            {
                "raw_files": [{"path": "raw/matrix.xlsx", "sha256": sha256_file(raw)}],
                "config": {"include_zero_doses": True},
            },
        )
        design = matrix_design(bundle, root, "test")
        assert len(design) == 4
        assert "inhibition" not in design
        last = design.loc[design.observation_id.eq("m:2:2")].iloc[0]
        assert (last.dose_a_um, last.dose_b_um) == (2.0, 1.0)
        designs.append(design)
    pd.testing.assert_frame_equal(designs[0], designs[1])
    pd.testing.assert_frame_equal(designs[0], designs[2])
    # Changed raw content without an updated authoritative manifest is rejected.
    write_matrix(raw, response=70.0)
    with pytest.raises(ValueError, match="hash mismatch"):
        matrix_design(bundle, root, "test")


def test_mixture_rosenblatt_uses_posterior_component_weights() -> None:
    parameters = {
        "weights": np.array([[0.5, 0.5]]),
        "means": np.array([[[0.0, -2.0], [5.0, 2.0]]]),
        "scales": np.ones((1, 2, 2)),
        "correlations": np.zeros((1, 2)),
    }
    row = density_rows(parameters, np.array([[0.0, 0.0]])).iloc[0]
    posterior_first = 1 / (1 + np.exp(-12.5))
    correct = posterior_first * ndtr(2) + (1 - posterior_first) * ndtr(-2)
    assert row.pit_b_given_a == pytest.approx(correct)
    assert row.pit_b == pytest.approx(0.5)
    assert row.pit_b_given_a > 0.97


def test_target_window_is_inclusive_despite_binary_roundoff() -> None:
    values = pd.Series([0.45, 1 - 55 / 100, 0.55, 0.5, 0.45 - 1e-12, 0.55 + 1e-12, np.nan])
    assert target_window_mask(values, EvaluationConfig()).tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
