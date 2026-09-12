"""Read-only reassessment and protocol-separated summaries of fixed experiments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .assessment import enhanced_assessment
from .config import EvaluationConfig, config_from_dict
from .data import DataBundle
from .evaluation import load_trained_model
from .utils import (
    atomic_json,
    environment_info,
    fingerprint,
    guard_output,
    load_checkpoint,
    read_json,
    sha256_file,
)


def assess_saved(
    run_dir: Path,
    output: Path,
    split: str = "test",
    *,
    evaluation_settings: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Assess saved predictions with their own configuration and selected checkpoint."""
    run = read_json(run_dir / "run.json")
    config = config_from_dict(run["config"])
    if evaluation_settings is not None:
        config.evaluation = copy.deepcopy(evaluation_settings)
        config.validate()
    guard_output(output, Path(config.dataset_root))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Assessment output already exists: {output}")
    checkpoint = run_dir / "model" / "best.pt"
    saved = load_checkpoint(checkpoint)
    bundle = DataBundle(Path(run["data_dir"]))
    original = run_dir / "evaluation" / split
    metrics = read_json(original / "metrics.json")
    if (
        not saved.get("selection_only", False)
        or metrics["checkpoint_sha256"] != sha256_file(checkpoint)
        or metrics["dataset_fingerprint"] != bundle.manifest["fingerprint"]
        or fingerprint(saved["config"]) != fingerprint(run["config"])
    ):
        raise ValueError("Saved evaluation does not match the selected experiment")
    frame = pd.read_parquet(original / "predictions.parquet")
    expected = bundle.observations.loc[bundle.observations.split.eq(split)]
    if frame.observation_id.duplicated().any() or set(frame.observation_id) != set(
        expected.observation_id
    ):
        raise ValueError("Saved evaluation must cover the complete split exactly once")
    with np.load(original / "mixture_parameters.npz", allow_pickle=False) as archive:
        parameters = {k: archive[k] for k in archive.files}
    model = load_trained_model(bundle, config, checkpoint, config.device)
    report = enhanced_assessment(bundle, config, model, frame, parameters, output, split)
    report["provenance"] = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_prediction_sha256": sha256_file(original / "predictions.parquet"),
        "source_parameters_sha256": sha256_file(original / "mixture_parameters.npz"),
        "environment": environment_info(),
        "training_seed": config.seed,
        "split_seed": config.split.seed,
    }
    atomic_json(output / "assessment.json", report)
    return report


def report_row(
    report: dict[str, Any], *, training_seed: int, split_seed: int, run_dir: str
) -> dict[str, Any]:
    primary = report["primary"]
    overall = primary["identity_cluster_intervals"]["overall"]
    near = primary["identity_cluster_intervals"]["near_target"]
    target = report["target_achievement"]
    return {
        "protocol": report["protocol"],
        "generalization": report["generalization"],
        "split_seed": split_seed,
        "training_seed": training_seed,
        "run_dir": run_dir,
        **report["biological_units"],
        "macro_identity_nll": overall["nll"]["estimate"],
        "macro_identity_nll_lower": overall["nll"]["lower"],
        "macro_identity_nll_upper": overall["nll"]["upper"],
        "macro_identity_nll_um": overall["nll_um"]["estimate"],
        "macro_identity_mae_um": overall["absolute_error_um"]["estimate"],
        "row_nll": overall["row_nll"]["estimate"],
        "near_target_macro_nll": near["nll"]["estimate"],
        "near_target_identities": primary["near_target_identities"],
        "inhibition_macro_nll": primary["group_metrics"]["inhibition"]["equal_group_nll"],
        "worst_cell_nll": primary["group_metrics"]["cell"]["worst_nll"],
        "worst_cell_fraction_nll": primary["group_metrics"]["cell"]["worst_fraction_mean_nll"],
        "worst_pair_fraction_nll": primary["group_metrics"]["drug_pair"]["worst_fraction_mean_nll"],
        "nll_p99": primary["density_quantiles"]["nll"]["p99"],
        "joint90_macro_coverage": overall["coverage_joint_0.9"]["estimate"],
        "target_macro_hit": target.get("macro_identity_hit_ci", {}).get("estimate"),
        "target_macro_hit_lower": target.get("macro_identity_hit_ci", {}).get("lower"),
        "target_macro_hit_upper": target.get("macro_identity_hit_ci", {}).get("upper"),
        "target_scorable_matrices": target.get("scorable_matrices"),
    }


def summarize_records(records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Describe variability within each protocol without pooling overlapping tests."""
    table = pd.DataFrame(records)
    summary = {}
    for protocol, g in table.groupby("protocol", sort=True):
        g.to_csv(output / f"{protocol}_results.csv", index=False)
        metrics = [
            "macro_identity_nll_um",
            "macro_identity_mae_um",
            "row_nll",
            "near_target_macro_nll",
            "inhibition_macro_nll",
            "target_macro_hit",
        ]
        splits = g.split_seed.nunique()
        trainings = g.training_seed.nunique()
        balanced = (
            len(g) == splits * trainings and not g.duplicated(["split_seed", "training_seed"]).any()
        )
        entry = {
            "runs": len(g),
            "partitions": splits,
            "training_seeds": trainings,
            "fully_crossed": balanced,
            "variability": {},
        }
        for metric in metrics:
            finite = g.dropna(subset=[metric])
            v = finite[metric]
            entry["variability"][metric] = {
                "mean": float(v.mean()) if len(v) else None,
                "minimum": float(v.min()) if len(v) else None,
                "maximum": float(v.max()) if len(v) else None,
                "sd_across_runs_descriptive": float(v.std()) if len(v) > 1 else None,
            }
            if balanced and splits > 1 and trainings > 1 and len(finite) == len(g):
                pivot = g.pivot(index="split_seed", columns="training_seed", values=metric)
                entry["variability"][metric].update(
                    {
                        "sd_of_partition_means": float(pivot.mean(axis=1).std()),
                        "rms_training_sd_within_partition": float(
                            np.sqrt(pivot.var(axis=1, ddof=1).mean())
                        ),
                        "sd_of_training_seed_means": float(pivot.mean(axis=0).std()),
                        "interaction_rms": float(
                            np.sqrt(
                                (
                                    (
                                        pivot.to_numpy()
                                        - pivot.mean(axis=1).to_numpy()[:, None]
                                        - pivot.mean(axis=0).to_numpy()[None, :]
                                        + pivot.to_numpy().mean()
                                    )
                                    ** 2
                                ).mean()
                            )
                        ),
                    }
                )
        summary[protocol] = entry
    report = {
        "protocols": summary,
        "interpretation": "Cold-start and warm-completion are distinct estimands. Across-run SDs are descriptive: test identities overlap, so runs are not independent biological replicates. No pooled row CI or ranking by test scores.",
        "factorial_interpretation": "Within-partition variability includes VAE and final training randomness; partitions and their train-fitted transforms are fixed. Interaction is not separable from residual noise with one run per factorial combination. Cross-partition physical-unit NLL/MAE avoid changing standardized-coordinate units.",
    }
    atomic_json(output / "summary.json", report)
    return report


def assess_suite(suite: Path, output: Path) -> dict[str, Any]:
    """Reassess completed suite members into a fresh directory, retaining originals."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Suite assessment already exists: {output}")
    members = sorted(suite.glob("*/run.json"))
    if not members:
        raise ValueError("No completed suite members found")
    records = []
    for member in members:
        meta = read_json(member)
        report = assess_saved(member.parent, output / member.parent.name)
        records.append(
            report_row(
                report,
                training_seed=meta["config"]["seed"],
                split_seed=meta["config"]["split"]["seed"],
                run_dir=str(member.parent),
            )
        )
    return summarize_records(records, output)
