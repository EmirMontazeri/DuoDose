"""Retrospective assessment of fixed models, without fitting to held-out labels.

All resampling preserves entire biological identity clusters. Target-dose
selection and response scoring have separate interfaces; the selector cannot
access a response column. Density diagnostics assess calibration without
recalibrating or changing the model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, ndtr

from .config import EvaluationConfig
from .data import DataBundle
from .layout import DEFAULT_DATASET_NAME, DatasetPaths
from .preprocess import parse_corrected_matrix
from .utils import atomic_json, read_json, sha256_file


def target_window_mask(values: pd.Series, settings: EvaluationConfig) -> pd.Series:
    """Inclusive inhibition window, allowing only float64 round-off at its edges.

    For example, abs(0.55 - 0.5) exceeds 0.05 in binary arithmetic. The
    round-off allowance is below 2e-15 for the default target, not an assay
    tolerance expansion. NaN responses remain outside the window.
    """
    error = (values - settings.target_inhibition).abs()
    roundoff = (
        8
        * np.finfo(np.float64).eps
        * max(1.0, abs(settings.target_inhibition), settings.target_tolerance)
    )
    return error <= settings.target_tolerance + roundoff


def cluster_interval(
    frame: pd.DataFrame, value: str, settings: EvaluationConfig, *, macro: bool = True
) -> dict[str, Any]:
    """Percentile bootstrap of intact identity clusters, conditional on this split.

    No row resampling or iid row p-values. The micro variant recomputes its
    ratio of sums after cluster resampling, preserving unequal cluster sizes.
    """
    values = frame.loc[np.isfinite(frame[value]), ["identity_group", value]]
    groups = values.groupby("identity_group")[value].agg(["sum", "count", "mean"])
    n = len(groups)
    estimate = None if not n else float(groups["mean"].mean() if macro else values[value].mean())
    result = {
        "estimate": estimate,
        "identities": n,
        "rows": len(values),
        "confidence_level": settings.confidence_level,
        "lower": None,
        "upper": None,
        "method": "intact identity-cluster percentile bootstrap",
        "estimand": "equal identity mean" if macro else "row-weighted mean",
        "small_cluster_warning": n < 20,
    }
    if n < 2:
        return result
    rng = np.random.default_rng(settings.bootstrap_seed)
    draws = rng.integers(0, n, (settings.bootstrap_repetitions, n))
    if macro:
        boot = groups["mean"].to_numpy()[draws].mean(axis=1)
    else:
        boot = groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[draws].sum(
            axis=1
        )
    tail = (1 - settings.confidence_level) / 2
    result["lower"], result["upper"] = map(float, np.quantile(boot, [tail, 1 - tail]))
    return result


def density_rows(parameters: dict[str, np.ndarray], target: np.ndarray) -> pd.DataFrame:
    """Exact Gaussian-mixture marginal PIT and A-then-B Rosenblatt transform.

    Uses conditional component weights p(component | observed A), not prior
    weights, to assess the bivariate conditional distribution. Input arrays must
    be aligned to exactly the same observations by their stored indices.
    """
    w, mu, sigma, rho = (
        np.asarray(parameters[k], dtype=np.float64)
        for k in ("weights", "means", "scales", "correlations")
    )
    z = (np.asarray(target, dtype=np.float64)[:, None, :] - mu) / sigma
    logw = np.log(np.maximum(w, np.finfo(float).tiny))
    marginal_pit = (w[..., None] * ndtr(z)).sum(axis=1)
    conditional_logits = logw - np.log(sigma[..., 0]) - z[..., 0] ** 2 / 2
    conditional_weights = np.exp(
        conditional_logits - logsumexp(conditional_logits, axis=1)[:, None]
    )
    conditional_z = (z[..., 1] - rho * z[..., 0]) / np.sqrt((1 - rho) * (1 + rho))
    conditional_pit = (conditional_weights * ndtr(conditional_z)).sum(axis=1)
    variances = sigma**2
    determinant = variances.prod(axis=-1) * (1 - rho) * (1 + rho)
    largest = (
        variances.sum(axis=-1)
        + np.sqrt(
            (variances[..., 0] - variances[..., 1]) ** 2 + 4 * rho**2 * variances.prod(axis=-1)
        )
    ) / 2
    smallest = determinant / largest
    out = pd.DataFrame(
        {
            "pit_a": marginal_pit[:, 0],
            "pit_b": marginal_pit[:, 1],
            "pit_b_given_a": conditional_pit,
            "minimum_component_scale": sigma.min(axis=(1, 2)),
            "minimum_covariance_eigenvalue": smallest.min(axis=1),
            "weighted_log_determinant": (w * np.log(determinant)).sum(axis=1),
            "mixture_entropy": -(w * logw).sum(axis=1),
            "maximum_abs_correlation": np.abs(rho).max(axis=1),
        }
    )
    for level in (0.5, 0.8, 0.9, 0.95):
        tail = (1 - level) / 2
        for axis in ("a", "b"):
            out[f"coverage_{axis}_{level}"] = (
                out[f"pit_{axis}"].between(tail, 1 - tail).astype(float)
            )
        joint_tail = (1 - math.sqrt(level)) / 2
        out[f"coverage_joint_{level}"] = (
            out.pit_a.between(joint_tail, 1 - joint_tail)
            & out.pit_b_given_a.between(joint_tail, 1 - joint_tail)
        ).astype(float)
    return out


def grouped_primary(
    frame: pd.DataFrame, settings: EvaluationConfig, output: Path
) -> dict[str, Any]:
    """Promote group, response-stratum, tail, and calibration results together."""
    metrics = [
        "nll",
        "nll_um",
        "absolute_error_scaled",
        "absolute_error_um",
        "squared_error_scaled",
        "squared_error_um",
        "invalid_point_fraction",
    ]
    coverage = [name for name in frame if name.startswith("coverage_")]
    summaries: dict[str, Any] = {}
    for label, cols in {
        "cell": ["identity_group"],
        "drug_pair": ["drug_a", "drug_b"],
        "cell_pair": ["identity_group", "drug_a", "drug_b"],
        "matrix": ["matrix_id"],
        "inhibition": ["inhibition_bin"],
    }.items():
        g = frame.groupby(cols, observed=True, sort=True)
        table = g[metrics + coverage].mean()
        table["count"] = g.size()
        table["identities"] = g.identity_group.nunique()
        table["matrices"] = g.matrix_id.nunique()
        table["cell_labels"] = g.cell_id.agg(lambda x: "|".join(sorted(set(x))))
        table["rmse_scaled"] = np.sqrt(table.squared_error_scaled)
        table["rmse_um"] = np.sqrt(table.squared_error_um)
        table = table.reset_index()
        table.sort_values("nll", ascending=False).to_csv(
            output / f"primary_by_{label}.csv", index=False
        )
        count = max(1, math.ceil(len(table) * settings.worst_group_fraction))
        summaries[label] = {
            "groups": len(table),
            "equal_group_nll": float(table.nll.mean()),
            "equal_group_mae_um": float(table.absolute_error_um.mean()),
            "worst_fraction": settings.worst_group_fraction,
            "worst_nll_group_count": count,
            "worst_fraction_mean_nll": float(table.nlargest(count, "nll").nll.mean()),
            "worst_nll": float(table.nll.max()),
            "worst_mae_um": float(table.absolute_error_um.max()),
            "worst_fraction_mean_mae_um": float(
                table.nlargest(count, "absolute_error_um").absolute_error_um.mean()
            ),
            "single_identity_groups": int(table.identities.eq(1).sum()),
        }
    near = frame.loc[target_window_mask(frame.inhibition, settings)]
    intervals = {}
    for name, cohort in (("overall", frame), ("near_target", near)):
        intervals[name] = {
            metric: cluster_interval(cohort, metric, settings) for metric in metrics + coverage
        }
        intervals[name]["row_nll"] = cluster_interval(cohort, "nll", settings, macro=False)
    stratified = {}
    for name, cohort in frame.groupby("inhibition_bin", observed=True, sort=True):
        stratified[str(name)] = {
            "fraction": len(cohort) / len(frame),
            "macro_cell_nll": cluster_interval(cohort, "nll", settings),
            "macro_cell_mae_um": cluster_interval(cohort, "absolute_error_um", settings),
        }
    # Expose all fixed bins, including empty bins; no implicit zero imputation.
    fixed_bins = pd.IntervalIndex.from_breaks(
        [-np.inf, *settings.inhibition_bin_edges, np.inf], closed="left"
    )
    present = set(frame.inhibition_bin)
    quantiles = {}
    for key in (
        "nll",
        "minimum_component_scale",
        "minimum_covariance_eigenvalue",
        "weighted_log_determinant",
        "mixture_entropy",
        "maximum_abs_correlation",
        "swap_abs_log_density_difference",
    ):
        quantiles[key] = dict(
            zip(
                ("min", "p01", "p50", "p90", "p99", "max"),
                map(float, frame[key].quantile([0, 0.01, 0.5, 0.9, 0.99, 1])),
            )
        )
    calibration = {}
    for key in ("pit_a", "pit_b", "pit_b_given_a"):
        calibration[key] = {
            "mean": float(frame[key].mean()),
            "variance": float(frame[key].var(ddof=0)),
            "histogram_counts": np.histogram(frame[key], bins=np.linspace(0, 1, 11))[0].tolist(),
            "histogram_edges": np.linspace(0, 1, 11).tolist(),
        }
    return {
        "group_metrics": summaries,
        "identity_cluster_intervals": intervals,
        "inhibition_strata": stratified,
        "empty_inhibition_bins": [str(x) for x in fixed_bins if str(x) not in present],
        "near_target_rows": len(near),
        "near_target_identities": near.identity_group.nunique(),
        "density_quantiles": quantiles,
        "calibration": calibration,
        "uncertainty_scope": "Conditional on a fixed selected model and split; intact identity clusters retain all nested pairs, matrices and dose rows. No iid-row inference; not uncertainty over retraining or all biological populations.",
        "coverage_definition": "Central marginal PIT intervals and A-then-B Rosenblatt rectangles; joint side probability sqrt(nominal). Diagnostic only, no fitted calibration or iid goodness-of-fit p-values.",
    }


def matrix_design(
    bundle: DataBundle,
    dataset_root: Path,
    split: str,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> pd.DataFrame:
    """Reconstruct full original grids from hash-verified workbooks, including missing/QC rows.

    The returned table contains no response values or missingness masks. Matrix
    inclusion follows the already fixed prepared split. Original axis values,
    not surviving response coordinates, define selection candidates.
    """
    source = read_json(bundle.path / "source_manifest.json")
    paths = DatasetPaths(dataset_root, dataset_name)
    hashes = {entry["path"]: entry["sha256"] for entry in source["raw_files"]}
    rows = []
    metadata = bundle.observations.loc[bundle.observations.split.eq(split)]
    for matrix_id, g in metadata.groupby("matrix_id", sort=True):
        first = g.iloc[0]
        relative = first.source_file
        path = paths.source_path(relative).resolve()
        if not path.is_relative_to(paths.raw.resolve()):
            raise ValueError("Matrix source must be under dataset/raw")
        if sha256_file(path) != hashes[relative]:
            raise ValueError(f"Raw matrix hash mismatch: {relative}")
        matrix = parse_corrected_matrix(path)
        for i, a in enumerate(matrix["dose1_um"], start=1):
            for j, b in enumerate(matrix["dose2_um"], start=1):
                if not source["config"]["include_zero_doses"] and (a == 0 or b == 0):
                    continue
                da, db = (b, a) if first.canonical_order_swapped else (a, b)
                rows.append(
                    {
                        "matrix_id": matrix_id,
                        "cell_id": first.cell_id,
                        "drug_a": first.drug_a,
                        "drug_b": first.drug_b,
                        "dose_a_um": da,
                        "dose_b_um": db,
                        "grid_row": i,
                        "grid_col": j,
                        "observation_id": f"{matrix_id}:{i}:{j}",
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def select_target_grid(
    model: torch.nn.Module,
    bundle: DataBundle,
    design: pd.DataFrame,
    settings: EvaluationConfig,
    device: torch.device,
) -> pd.DataFrame:
    """One fixed-target query per matrix, snapped by squared scaled-log-dose distance.

    Never restrict candidates by response, attainability, or missingness. Ties
    resolve by dose A, dose B, then original coordinate. No clipping of the raw
    generated mean; report its validity and the snapping distance separately.
    """
    allowed = {
        "matrix_id",
        "cell_id",
        "drug_a",
        "drug_b",
        "dose_a_um",
        "dose_b_um",
        "grid_row",
        "grid_col",
        "observation_id",
    }
    if set(design) != allowed:
        raise ValueError(
            "Target selector requires only the whitelisted response-blind design columns"
        )
    cells = {x: i for i, x in enumerate(bundle.cell_ids)}
    drugs = {x: i for i, x in enumerate(bundle.drug_ids)}
    contexts = design[["cell_id", "drug_a", "drug_b"]].drop_duplicates()
    means = {}
    model.eval()
    for start in range(0, len(contexts), settings.batch_size):
        c = contexts.iloc[start : start + settings.batch_size]
        cell = torch.as_tensor(bundle.cell_features[[cells[x] for x in c.cell_id]], device=device)
        a = torch.as_tensor(bundle.drug_features[[drugs[x] for x in c.drug_a]], device=device)
        b = torch.as_tensor(bundle.drug_features[[drugs[x] for x in c.drug_b]], device=device)
        inhibition = torch.full((len(c), 1), settings.target_inhibition, device=device)
        p = model(cell, inhibition, a, b)
        for key, mean in zip(
            c.itertuples(index=False, name=None), p.mixture_mean.cpu().numpy(), strict=True
        ):
            means[key] = mean.astype(float)
    records = []
    for _, g in design.groupby("matrix_id", sort=True):
        g = g.sort_values(["dose_a_um", "dose_b_um", "grid_row", "grid_col"])
        first = g.iloc[0]
        mean = means[(first.cell_id, first.drug_a, first.drug_b)]
        targets = bundle.dose_transform.transform(g[["dose_a_um", "dose_b_um"]].to_numpy())
        distances = np.sum((targets - mean) ** 2, axis=1)
        k = int(np.argmin(distances))
        selected = g.iloc[k].to_dict()
        zero = float(bundle.dose_transform.transform(np.array([0.0]))[0])
        selected.update(
            {
                "requested_inhibition": settings.target_inhibition,
                "generated_a_scaled": float(mean[0]),
                "generated_b_scaled": float(mean[1]),
                "generated_mean_invalid": bool((mean < zero).any()),
                "snap_distance_scaled": float(np.sqrt(distances[k])),
                "candidate_count": len(g),
            }
        )
        records.append(selected)
    return pd.DataFrame(records)


def score_target_grid(
    selected: pd.DataFrame, responses: pd.DataFrame, settings: EvaluationConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reveal responses only after all recommendations are frozen; no oracle reselection."""
    if responses.observation_id.duplicated().any():
        raise ValueError("Target scoring requires unique observation IDs")
    scored = selected.merge(
        responses[["observation_id", "inhibition"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    scored["scorable"] = scored.inhibition.notna()
    scored["absolute_inhibition_error"] = (scored.inhibition - settings.target_inhibition).abs()
    scored["hit"] = np.where(
        scored.scorable,
        target_window_mask(scored.inhibition, settings).astype(float),
        np.nan,
    )
    # The attainable-grid oracle is reported only as an audit denominator.
    attainable = (
        responses.assign(available_hit=target_window_mask(responses.inhibition, settings))
        .groupby("matrix_id")
        .available_hit.any()
    )
    scored["available_target_hit"] = scored.matrix_id.map(attainable).fillna(False).astype(bool)
    valid = scored.loc[scored.scorable]
    pair_scores = (
        valid.groupby(["identity_group", "drug_a", "drug_b"])[["hit", "absolute_inhibition_error"]]
        .mean()
        .reset_index()
    )
    hit_count = int(valid.hit.sum())
    unknown = int((~scored.scorable).sum())
    report = {
        "policy": "nearest original grid coordinate to frozen MDN mean in training-standardized log1p dose; fixed target; lexicographic ties",
        "scope": "Response at snapped measured grid dose, not proof about the unsnapped generated dose; QC-retained matrix population only",
        "matrices": len(scored),
        "scorable_matrices": len(valid),
        "unscorable_matrices": unknown,
        "unscorable_reason": "selected response missing or excluded by fixed preprocessing; never replace with another coordinate",
        "hit_rate_scorable": float(valid.hit.mean()) if len(valid) else None,
        "all_matrix_hit_rate_bounds": [
            hit_count / len(scored),
            (hit_count + unknown) / len(scored),
        ],
        "available_target_matrix_fraction": float(scored.available_target_hit.mean()),
        "generated_mean_invalid_fraction": float(scored.generated_mean_invalid.mean()),
        "snap_distance_median": float(scored.snap_distance_scaled.median()),
        "snap_distance_p95": float(scored.snap_distance_scaled.quantile(0.95)),
        "matrix_weighted_hit_ci": cluster_interval(scored, "hit", settings, macro=False),
        "macro_identity_hit_ci": cluster_interval(scored, "hit", settings),
        "macro_identity_pair_hit_ci": cluster_interval(pair_scores, "hit", settings),
        "macro_identity_pair_error_ci": cluster_interval(
            pair_scores, "absolute_inhibition_error", settings
        ),
    }
    return scored, report


def enhanced_assessment(
    bundle: DataBundle,
    config: Any,
    model: torch.nn.Module,
    frame: pd.DataFrame,
    parameters: dict[str, np.ndarray],
    output: Path,
    split: str,
    *,
    design: pd.DataFrame | None = None,
    include_target: bool = True,
) -> dict[str, Any]:
    """Write auditable assessment artifacts for a fixed checkpoint and held-out split."""
    indices = parameters["indices"].astype(int)
    expected = bundle.observations.iloc[indices]
    if (
        not np.array_equal(frame.observation_id.to_numpy(), expected.observation_id.to_numpy())
        or not expected.split.eq(split).all()
    ):
        raise ValueError("Assessment parameter/prediction/split alignment mismatch")
    identity = dict(zip(bundle.cell_ids, bundle.cell_identity_group_ids, strict=True))
    frame = frame.reset_index(drop=True).copy()
    frame["identity_group"] = frame.cell_id.map(identity)
    diagnostics = density_rows(parameters, bundle.targets[indices])
    frame = pd.concat([frame, diagnostics], axis=1)
    frame["inhibition_bin"] = pd.cut(
        frame.inhibition, [-np.inf, *config.evaluation.inhibition_bin_edges, np.inf], right=False
    ).astype(str)
    output.mkdir(parents=True, exist_ok=True)
    primary = grouped_primary(frame, config.evaluation, output)
    frame.to_parquet(output / "diagnostic_rows.parquet", index=False)
    # Target evaluation uses full original dose axes; portable bundles lacking
    # raw workbooks must explicitly supply a response-blind design table.
    target_report = {
        "status": "unavailable",
        "reason": "Original matrix axes are unavailable; no response-derived candidate fallback",
    }
    if include_target:
        if design is None:
            design = matrix_design(
                bundle, Path(config.dataset_root), split, dataset_name=config.dataset_name
            )
        expected_matrices = set(expected.matrix_id)
        if set(design.matrix_id) != expected_matrices:
            raise ValueError("Target design must cover exactly the requested split matrices")
        selected = select_target_grid(
            model, bundle, design, config.evaluation, next(model.parameters()).device
        )
        selected["identity_group"] = selected.cell_id.map(identity)
        selected.to_parquet(output / "target_choices.parquet", index=False)
        scored, target_report = score_target_grid(selected, expected, config.evaluation)
        scored.to_parquet(output / "target_scores.parquet", index=False)
    protocol = bundle.manifest["split"]["configuration"]["protocol"]
    report = {
        "schema_version": "assessment-v1.1",
        "split": split,
        "protocol": protocol,
        "generalization": (
            "cold-start unseen identities"
            if protocol == "cell"
            else "warm-completion held-out cell/pair groups"
        ),
        "evaluation_settings": vars(config.evaluation),
        "primary": primary,
        "target_achievement": target_report,
        "biological_units": {
            "rows": len(frame),
            "identities": frame.identity_group.nunique(),
            "cell_pair_groups": frame.groupby(["identity_group", "drug_a", "drug_b"]).ngroups,
            "matrices": frame.matrix_id.nunique(),
        },
        "dataset_fingerprint": bundle.manifest["fingerprint"],
        "limitations": [
            "Reporting describes retrospective performance on measured assay splits; it is not a prospective confirmatory test.",
            "Groups with few identities are displayed with support counts, not hidden or treated as replicated biology.",
            "Cold-start and warm-completion results must not be pooled as one generalization estimate.",
            "Calibration assessed on measured, discrete assay grids; continuous PIT uniformity is a diagnostic, not a claim of continuous biological ground truth.",
        ],
    }
    atomic_json(output / "assessment.json", report)
    return report
