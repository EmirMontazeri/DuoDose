"""Predeclared crossed partition/training-seed experiments; test never selects runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .reporting import assess_saved, report_row, summarize_records
from .utils import atomic_json, fingerprint, guard_output, read_json, sha256_file


def factorial_design(
    split_seeds: list[int], training_seeds: list[int], protocols: list[str]
) -> list[dict[str, Any]]:
    """Materialize a complete Cartesian product; never zip the two seed axes."""
    for seeds in (split_seeds, training_seeds):
        if (
            not seeds
            or any(type(x) is not int or x < 0 for x in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Seeds must be distinct nonnegative integers")
    if (
        not protocols
        or len(set(protocols)) != len(protocols)
        or set(protocols) - {"cell", "cell_pair"}
    ):
        raise ValueError("Protocols must be distinct cell/cell_pair values")
    return [
        {"protocol": p, "split_seed": s, "training_seed": t}
        for p in protocols
        for s in split_seeds
        for t in training_seeds
    ]


def run_factorial(
    config: ExperimentConfig,
    output: Path,
    split_seeds: list[int],
    training_seeds: list[int],
    protocols: list[str],
    *,
    resume: bool = False,
    reference_suite: Path | None = None,
) -> dict[str, Any]:
    """Run all declared combinations; reuse only exact scientific reference matches.

    Existing legacy diagonal experiments can be imported, never relabeled as new
    independent runs. Configuration differences in evaluation-only settings do
    not change trained weights. References are revalidated before reassessment.
    """
    from .cli import run

    guard_output(output, Path(config.dataset_root))
    design = {
        "config": config.to_dict(),
        "combinations": factorial_design(split_seeds, training_seeds, protocols),
        "reference_suite": str(reference_suite.resolve()) if reference_suite else None,
        "test_policy": "Fixed architecture/hyperparameters, validation-only selection; evaluate every declared run without test-driven pruning",
        "training_randomness": "VAE initialization/sampling and final initialization/minibatch order; independent of partition seed",
        "inference_scope": "Repeated existing identities; descriptive variability, not independent biological replication",
    }
    design["fingerprint"] = fingerprint(design)
    manifest = output / "design.json"
    if output.exists() and any(output.iterdir()):
        if (
            not resume
            or not manifest.exists()
            or fingerprint(read_json(manifest)) != fingerprint(design)
        ):
            raise ValueError(
                "Nonempty factorial output requires --resume and identical frozen design"
            )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, design)
    references = [] if reference_suite is None else sorted(reference_suite.glob("*/run.json"))
    records = []
    for entry in design["combinations"]:
        effective = copy.deepcopy(config)
        effective.seed = entry["training_seed"]
        effective.split.seed = entry["split_seed"]
        effective.split.protocol = entry["protocol"]
        effective.validate()
        name = f"{entry['protocol']}_split_{entry['split_seed']}_train_{entry['training_seed']}"
        member = output / name
        actual = member
        science = effective.to_dict()
        science.pop("evaluation")
        for ref in references:
            candidate = read_json(ref)["config"]
            candidate = {k: v for k, v in candidate.items() if k != "evaluation"}
            if fingerprint(candidate) == fingerprint(science):
                actual = ref.parent
                break
        if actual == member:
            result = run(effective, member, resume=resume and (member / "run.json").exists())
            if result["status"] != "complete":
                return {"status": "interrupted", "member": name}
            assessment = read_json(
                member / "evaluation" / "test" / "assessment" / "assessment.json"
            )
        else:
            assessed = member / "reference_assessment"
            if resume and (assessed / "assessment.json").exists():
                assessment = read_json(assessed / "assessment.json")
                provenance = assessment.get("provenance", {})
                sources = {
                    "checkpoint_sha256": actual / "model" / "best.pt",
                    "source_prediction_sha256": actual
                    / "evaluation"
                    / "test"
                    / "predictions.parquet",
                    "source_parameters_sha256": actual
                    / "evaluation"
                    / "test"
                    / "mixture_parameters.npz",
                }
                if (
                    assessment.get("schema_version") != "assessment-v1.1"
                    or assessment.get("evaluation_settings") != vars(effective.evaluation)
                    or any(
                        provenance.get(key) != sha256_file(path) for key, path in sources.items()
                    )
                ):
                    raise ValueError(
                        "Reference assessment or its source artifacts changed on resume"
                    )
            else:
                assessment = assess_saved(
                    actual, assessed, evaluation_settings=effective.evaluation
                )
        records.append(
            report_row(
                assessment,
                training_seed=effective.seed,
                split_seed=effective.split.seed,
                run_dir=str(actual),
            )
        )
        atomic_json(
            output / "progress.json",
            {"completed": len(records), "planned": len(design["combinations"]), "records": records},
        )
        summarize_records(records, output)
    return {"status": "complete", **summarize_records(records, output)}
