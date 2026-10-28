"""Executable lifecycle for reproducible experiments.

Run ``reverse-dose --help`` for stages. Relative paths resolve from the working
directory. Commands refuse raw-data output paths and artifact overwrites.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig, load_config
from .data import (
    EXPRESSION_POPULATION,
    SCHEMA_VERSION as DATA_SCHEMA_VERSION,
    DataBundle,
    extract_expression,
    prepare_dataset,
)
from .evaluation import evaluate
from .inference import Predictor, export_model, write_prediction
from .layout import DatasetPaths
from .preprocess import SCHEMA_VERSION as SOURCE_SCHEMA_VERSION, map_cells, preprocess_sources
from .training import train_stage
from .utils import (
    atomic_json,
    environment_info,
    fingerprint,
    guard_output,
    load_checkpoint,
    read_json,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


def artifact_paths(config: ExperimentConfig) -> tuple[Path, Path, Path]:
    """Compute stable configuration-specific source and prepared paths."""
    paths = DatasetPaths(Path(config.dataset_root), config.dataset_name)
    source_key = fingerprint({"sources": asdict(config.sources), "schema": SOURCE_SCHEMA_VERSION})[
        :12
    ]
    source = paths.processed / "dose_response" / f"source_{source_key}"
    expression_key = fingerprint(
        {"population": EXPRESSION_POPULATION, "schema": DATA_SCHEMA_VERSION}
    )[:12]
    expression = paths.processed / "cell_expression" / f"expression_{expression_key}"
    split_key = fingerprint(
        {
            "source": source_key,
            "split": asdict(config.split),
            "features": asdict(config.features),
            "data_schema": DATA_SCHEMA_VERSION,
            "expression": expression_key,
        }
    )[:12]
    prepared = paths.splits / f"split_{split_key}"
    return source, expression, prepared


def prepare(
    config: ExperimentConfig,
    source_dir: Path | None = None,
    expression_dir: Path | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Regenerate or verify every derived source, expression, split and feature artifact."""
    default_source, default_expression, default_data = artifact_paths(config)
    source_dir, expression_dir, data_dir = (
        source_dir or default_source,
        expression_dir or default_expression,
        data_dir or default_data,
    )
    for path in (source_dir, expression_dir, data_dir):
        guard_output(path, Path(config.dataset_root))
    LOGGER.info(
        "Preparing source=%s expression=%s data=%s",
        source_dir,
        expression_dir,
        data_dir,
    )
    preprocess_sources(
        Path(config.dataset_root),
        asdict(config.sources),
        source_dir,
        dataset_name=config.dataset_name,
    )
    cells = pd.read_parquet(source_dir / "cells.parquet")
    extract_expression(
        Path(config.dataset_root), cells, expression_dir, dataset_name=config.dataset_name
    )
    prepare_dataset(
        source_dir,
        expression_dir,
        data_dir,
        asdict(config.split),
        asdict(config.features),
    )
    return data_dir


def run(
    config: ExperimentConfig,
    run_dir: Path,
    *,
    resume: bool = False,
    source_dir: Path | None = None,
    expression_dir: Path | None = None,
    data_dir: Path | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Execute preprocessing, offline training, final training, validation and test."""
    guard_output(run_dir, Path(config.dataset_root))
    if (run_dir / "run.json").exists():
        previous = read_json(run_dir / "run.json")
        if not resume:
            raise FileExistsError(f"Run exists; use --resume: {run_dir}")
        if fingerprint(previous["config"]) != fingerprint(config.to_dict()):
            raise ValueError(
                "Existing run configuration differs; resume uses the original configuration"
            )
    elif resume:
        raise FileNotFoundError(f"Cannot resume a run without run.json: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    data_dir = prepare(config, source_dir, expression_dir, data_dir)
    bundle = DataBundle(data_dir)
    run_manifest = {
        "config": config.to_dict(),
        "data_dir": str(data_dir.resolve()),
        "dataset_fingerprint": bundle.manifest["fingerprint"],
        "environment": environment_info(),
        "engineering_smoke": config.sources.max_matrices is not None,
    }
    if resume and previous["dataset_fingerprint"] != bundle.manifest["fingerprint"]:
        raise ValueError("Resume dataset fingerprint differs from original run")
    atomic_json(run_dir / "run.json", run_manifest)
    vae_last = run_dir / "vae" / "last.pt"
    vae = train_stage(
        bundle,
        config,
        run_dir,
        "vae",
        resume=vae_last if resume and vae_last.exists() else None,
    )
    if vae.name != "best.pt":
        LOGGER.warning("Offline training interrupted; resume the run to continue")
        return {"status": "interrupted", "stage": "vae"}
    model_last = run_dir / "model" / "last.pt"
    model = train_stage(
        bundle,
        config,
        run_dir,
        "model",
        vae_checkpoint=vae,
        resume=model_last if resume and model_last.exists() else None,
    )
    if model.name != "best.pt":
        LOGGER.warning("Final training interrupted; resume the run to continue")
        return {"status": "interrupted", "stage": "model"}
    result: dict[str, Any] = {
        "status": "complete",
        "data_dir": str(data_dir),
        "vae_checkpoint": str(vae),
        "model_checkpoint": str(model),
    }
    for split in ["validation", "test"] if evaluate_test else ["validation"]:
        output = run_dir / "evaluation" / split
        if resume and (output / "metrics.json").exists():
            from .utils import sha256_file

            report = read_json(output / "metrics.json")
            if report["checkpoint_sha256"] != sha256_file(model) or (
                report["dataset_fingerprint"] != bundle.manifest["fingerprint"]
            ):
                raise ValueError("Existing evaluation differs from resumed selected model")
        else:
            if resume and output.exists():
                preserved = output.with_name(f"{output.name}.interrupted.{uuid.uuid4().hex[:8]}")
                output.rename(preserved)
                LOGGER.warning(
                    "Preserved incomplete evaluation at %s; regenerating %s",
                    preserved,
                    output,
                )
            report = evaluate(bundle, config, model, output, split)
        result[split] = report["overall"]
    atomic_json(run_dir / "result.json", result)
    return result


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))


def build_parser() -> argparse.ArgumentParser:
    """Build discoverable commands without side effects."""
    parser = argparse.ArgumentParser(prog="reverse-dose", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect", help="Read audit, cell mapping and compute/runtime information"
    )
    _common_config(inspect)
    for name, help_text in (
        ("preprocess", "Parse raw corrected matrices and released drug chemistry"),
        ("prepare", "Reproduce sources, expression, splits and train-only transforms"),
    ):
        stage = sub.add_parser(name, help=help_text)
        _common_config(stage)
        stage.add_argument("--source-dir", type=Path)
        if name == "prepare":
            stage.add_argument("--expression-dir", type=Path)
            stage.add_argument("--data-dir", type=Path)
    expression = sub.add_parser(
        "extract-expression",
        help="Stream raw CMP RNA-seq for every available cell-line model",
    )
    _common_config(expression)
    expression.add_argument("--output", type=Path)
    for name, help_text in (
        ("train-vae", "Train the independent metadata VAE"),
        ("train", "Train the dose-prediction model using a selected frozen VAE encoder"),
    ):
        stage = sub.add_parser(name, help=help_text)
        _common_config(stage)
        stage.add_argument("--data-dir", type=Path, required=True)
        stage.add_argument("--run-dir", type=Path, required=True)
        stage.add_argument("--resume", type=Path, help="Explicit last.pt or epoch checkpoint")
        stage.add_argument(
            "--stop-after-steps",
            type=int,
            help="Controlled interruption for restart verification",
        )
        if name == "train":
            stage.add_argument("--vae-checkpoint", type=Path, required=True)
    evaluation = sub.add_parser("evaluate", help="Evaluate a fixed selected checkpoint")
    _common_config(evaluation)
    evaluation.add_argument("--data-dir", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--split", choices=["train", "validation", "test"], default="test")
    prediction = sub.add_parser(
        "predict", help="Predict dose distribution, candidates, samples and a 2D map"
    )
    prediction.add_argument("--data-dir", type=Path)
    prediction.add_argument("--checkpoint", type=Path)
    prediction.add_argument(
        "--bundle", type=Path, help="Portable export instead of --data-dir/--checkpoint"
    )
    prediction.add_argument(
        "--cell",
        required=True,
        help="Canonical cell ID, or a new cell label with --expression",
    )
    prediction.add_argument("--drug-a", required=True)
    prediction.add_argument("--drug-b", required=True)
    prediction.add_argument(
        "--inhibition", type=float, required=True, help="Fraction units: 0.5 means 50%%"
    )
    prediction.add_argument("--expression", type=Path, help="New cell CSV: gene_id,rsem_tpm")
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    prediction.add_argument("--samples", type=int, default=1000)
    prediction.add_argument("--seed", type=int, default=42)
    prediction.add_argument("--grid-size", type=int, default=100)
    prediction.add_argument(
        "--bounds", type=float, nargs=4, metavar=("A_MIN", "A_MAX", "B_MIN", "B_MAX")
    )
    export = sub.add_parser("export", help="Create a self-contained portable prediction directory")
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    pipeline = sub.add_parser("run", help="Run every stage from immutable raw data")
    _common_config(pipeline)
    pipeline.add_argument("--run-dir", type=Path, required=True)
    pipeline.add_argument("--source-dir", type=Path)
    pipeline.add_argument("--expression-dir", type=Path)
    pipeline.add_argument("--data-dir", type=Path)
    pipeline.add_argument("--resume", action="store_true", help="Resume the same experiment")
    pipeline.add_argument(
        "--skip-test",
        action="store_true",
        help="Train/select using validation; defer test access",
    )
    experiments = sub.add_parser(
        "experiments", help="Run a declared seed/protocol suite and save all results"
    )
    _common_config(experiments)
    experiments.add_argument("--output", type=Path, required=True)
    experiments.add_argument("--split-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    experiments.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43])
    experiments.add_argument(
        "--reference-suite",
        type=Path,
        help="Reuse exact matching completed scientific configurations",
    )
    experiments.add_argument(
        "--protocols", choices=["cell", "cell_pair"], nargs="+", default=["cell"]
    )
    experiments.add_argument("--resume", action="store_true")
    assessment = sub.add_parser(
        "assess-suite",
        help="Reassess fixed completed checkpoints with biological-unit statistics and retrospective target achievement",
    )
    assessment.add_argument("--suite", type=Path, required=True)
    assessment.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; failures are logged with an actionable traceback."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    try:
        if args.command == "assess-suite":
            from .reporting import assess_suite

            assess_suite(args.suite, args.output)
            return
        if args.command == "predict":
            if args.bundle:
                if args.data_dir or args.checkpoint:
                    raise ValueError("Use --bundle or --data-dir/--checkpoint")
                predictor = Predictor.from_export(args.bundle, args.device)
            else:
                if not args.data_dir or not args.checkpoint:
                    raise ValueError(
                        "Prediction requires --bundle or both --data-dir and --checkpoint"
                    )
                predictor = Predictor(args.data_dir, args.checkpoint, args.device)
            guard_output(args.output, Path(predictor.config.dataset_root))
            expression = predictor.expression_from_csv(args.expression) if args.expression else None
            bounds = (
                ((args.bounds[0], args.bounds[1]), (args.bounds[2], args.bounds[3]))
                if args.bounds
                else None
            )
            result, arrays = predictor.predict(
                args.cell,
                args.drug_a,
                args.drug_b,
                args.inhibition,
                expression=expression,
                samples=args.samples,
                seed=args.seed,
                grid_size=args.grid_size,
                bounds=bounds,
            )
            write_prediction(args.output, result, arrays, Path(predictor.config.dataset_root))
            LOGGER.info(
                "Point coordinates (µM): %s; nonnegative=%s",
                result["point_coordinates_um"],
                result["point_nonnegative"],
            )
            return
        if args.command == "export":
            dataset_root = Path(load_checkpoint(args.checkpoint)["config"]["dataset_root"])
            export_model(args.data_dir, args.checkpoint, args.output, dataset_root)
            return
        config = load_config(args.config)
        paths = DatasetPaths(Path(config.dataset_root), config.dataset_name)
        for name in ("output", "run_dir", "source_dir", "expression_dir", "data_dir"):
            path = getattr(args, name, None)
            if path is not None:
                guard_output(path, Path(config.dataset_root))
        if args.command == "inspect":
            root = Path(config.dataset_root)
            audit = read_json(paths.manifests / "audit_report.json")
            cells, _, mapping = map_cells(
                root, config.sources.aliases, dataset_name=config.dataset_name
            )
            payload = {
                "audit": audit,
                "environment": environment_info(),
                "mapped_cells": len(cells),
                "cell_mapping": mapping,
                "configuration": config.to_dict(),
            }
            output = paths.manifests / "inspection.json"
            atomic_json(output, payload)
            LOGGER.info(
                "Audit=%s, uniquely mapped cells=%d, runtime=%s; wrote %s",
                audit["overall_status"],
                len(cells),
                payload["environment"]["platform"],
                output,
            )
        elif args.command == "preprocess":
            source = args.source_dir or artifact_paths(config)[0]
            setup_logging(paths.manifests / "logs")
            preprocess_sources(
                Path(config.dataset_root),
                asdict(config.sources),
                source,
                dataset_name=config.dataset_name,
            )
        elif args.command == "extract-expression":
            cells, _, _ = map_cells(
                Path(config.dataset_root), config.sources.aliases, dataset_name=config.dataset_name
            )
            setup_logging(paths.manifests / "logs")
            extract_expression(
                Path(config.dataset_root),
                cells,
                args.output or artifact_paths(config)[1],
                dataset_name=config.dataset_name,
            )
        elif args.command == "prepare":
            setup_logging(paths.manifests / "logs")
            prepare(config, args.source_dir, args.expression_dir, args.data_dir)
        elif args.command in {"train", "train-vae"}:
            setup_logging(args.run_dir)
            bundle = DataBundle(args.data_dir)
            selected = train_stage(
                bundle,
                config,
                args.run_dir,
                "vae" if args.command == "train-vae" else "model",
                vae_checkpoint=getattr(args, "vae_checkpoint", None),
                resume=args.resume,
                stop_after_steps=args.stop_after_steps,
            )
            LOGGER.info("Stage checkpoint: %s", selected)
        elif args.command == "evaluate":
            setup_logging(args.output.parent / "evaluation_logs")
            evaluate(
                DataBundle(args.data_dir),
                config,
                args.checkpoint,
                args.output,
                args.split,
            )
        elif args.command == "run":
            run(
                config,
                args.run_dir,
                resume=args.resume,
                source_dir=args.source_dir,
                expression_dir=args.expression_dir,
                data_dir=args.data_dir,
                evaluate_test=not args.skip_test,
            )
        elif args.command == "experiments":
            from .experiments import run_factorial

            run_factorial(
                config,
                args.output,
                args.split_seeds,
                args.training_seeds,
                args.protocols,
                resume=args.resume,
                reference_suite=args.reference_suite,
            )
        else:
            parser.error("Unsupported command")
    except Exception:
        LOGGER.exception(
            "Pipeline failed; existing completed artifacts and raw sources are preserved"
        )
        raise
