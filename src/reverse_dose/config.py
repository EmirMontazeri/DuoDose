"""Strict, serializable experiment configuration with documented baseline defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
from types import UnionType
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .models import ModelConfig, VAEConfig
from .layout import DEFAULT_DATASET_NAME, DatasetPaths


@dataclass
class SourceConfig:
    """Raw extraction policy; aliases require an explicit source evidence string."""

    sources: list[str] = field(default_factory=lambda: ["dream"])
    fingerprint_bits: int = 2048
    fingerprint_radius: int = 2
    include_zero_doses: bool = True
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    unsupported_drug_policy: str = "exclude"
    unresolved_cell_policy: str = "exclude"
    missing_response_policy: str = "exclude"
    resolve_pubchem: bool = True
    pubchem_timeout_seconds: int = 30
    pubchem_retries: int = 3
    max_matrices: int | None = None
    workers: int = 4


@dataclass
class SplitConfig:
    """Group split policy; no matrix coordinates or replicates cross groups."""

    protocol: str = "cell"
    seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    vae_validation_fraction: float = 0.15


@dataclass
class FeatureConfig:
    """Train-only feature and dose transformations."""

    n_genes: int = 2000
    dose_reference_um: float = 1.0


@dataclass
class TrainingConfig:
    """Optimizer and restart settings for one independently trained stage."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clip: float = 5.0
    early_stopping_patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    checkpoint_every_steps: int = 500
    checkpoint_every_epochs: int = 5
    log_every_steps: int = 50


@dataclass
class EvaluationConfig:
    """Fixed reporting policy; these settings never select training checkpoints."""

    batch_size: int = 1024
    target_inhibition: float = 0.5
    target_tolerance: float = 0.05
    inhibition_bin_edges: list[float] = field(default_factory=lambda: [0, 0.25, 0.5, 0.75, 1])
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 20260912
    confidence_level: float = 0.95
    worst_group_fraction: float = 0.1


@dataclass
class ExperimentConfig:
    """Complete effective configuration persisted with all runs."""

    dataset_root: str = "dataset"
    dataset_name: str = DEFAULT_DATASET_NAME
    seed: int = 42
    device: str = "cpu"
    deterministic: bool = True
    threads: int = 4
    sources: SourceConfig = field(default_factory=SourceConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vae_training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(epochs=200, batch_size=16, learning_rate=0.001)
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vae_beta: float = 1.0
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the full effective configuration, including defaults."""
        return asdict(self)

    def validate(self) -> None:
        """Reject unsupported protocols and invalid numerical assumptions early."""

        def finite(value: Any, path: str = "config") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    finite(child, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for child in value:
                    finite(child, path)
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{path} must be finite")

        finite(self.to_dict())
        DatasetPaths(Path(self.dataset_root), self.dataset_name)
        if self.seed < 0 or self.threads < 1 or self.vae_beta <= 0:
            raise ValueError("seed >= 0, threads >= 1 and vae_beta > 0 required")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Unsupported device")
        if self.split.protocol not in {"cell", "cell_pair"}:
            raise ValueError("split.protocol must be cell or cell_pair")
        if not (
            0 < self.split.validation_fraction < 1
            and 0 < self.split.test_fraction < 1
            and self.split.validation_fraction + self.split.test_fraction < 1
            and 0 < self.split.vae_validation_fraction < 1
        ):
            raise ValueError("Invalid train/validation/test or VAE fractions")
        if self.features.n_genes < 1 or self.features.dose_reference_um <= 0:
            raise ValueError("n_genes and dose_reference_um must be positive")
        source = self.sources
        if not source.sources or set(source.sources) - {"dream", "oi"}:
            raise ValueError("sources.sources must contain dream and/or oi")
        if len(set(source.sources)) != len(source.sources):
            raise ValueError("Repeated matrix source")
        if source.workers < 1 or source.fingerprint_bits < 1 or source.fingerprint_radius < 0:
            raise ValueError("Invalid source worker/fingerprint configuration")
        if source.max_matrices is not None and source.max_matrices < 1:
            raise ValueError("max_matrices must be null or positive")
        if (
            source.unsupported_drug_policy not in {"exclude", "error"}
            or (source.unresolved_cell_policy not in {"exclude", "error"})
            or (source.missing_response_policy not in {"exclude", "error"})
        ):
            raise ValueError("Source exclusion policies must be exclude or error")
        if source.pubchem_timeout_seconds <= 0 or source.pubchem_retries < 1:
            raise ValueError("PubChem timeout and retries must be positive")
        for name, alias in source.aliases.items():
            if set(alias) != {"cell_id", "evidence"} or not all(alias.values()):
                raise ValueError(f"Alias {name} requires cell_id and evidence")
        for stage in (self.vae_training, self.training):
            for name in (
                "epochs",
                "batch_size",
                "checkpoint_every_steps",
                "checkpoint_every_epochs",
                "log_every_steps",
                "early_stopping_patience",
                "scheduler_patience",
            ):
                if getattr(stage, name) < 1:
                    raise ValueError(f"{name} must be positive")
            if stage.learning_rate <= 0 or stage.weight_decay < 0 or stage.gradient_clip <= 0:
                raise ValueError("Invalid optimizer parameters")
            if not 0 < stage.scheduler_factor < 1:
                raise ValueError("scheduler_factor must be in (0,1)")
        if self.evaluation.batch_size < 1 or self.evaluation.target_tolerance < 0:
            raise ValueError("Invalid evaluation batch size/tolerance")
        if (
            self.evaluation.bootstrap_repetitions < 100
            or self.evaluation.bootstrap_seed < 0
            or not 0 < self.evaluation.confidence_level < 1
            or not 0 < self.evaluation.worst_group_fraction <= 1
        ):
            raise ValueError("Invalid evaluation bootstrap/confidence/worst-group settings")
        edges = self.evaluation.inhibition_bin_edges
        if len(edges) < 2 or any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("inhibition_bin_edges must strictly increase")


T = TypeVar("T")


def _type_matches(value: Any, expected: Any) -> bool:
    if expected is Any:
        return True
    origin, arguments = get_origin(expected), get_args(expected)
    if origin is UnionType:
        return any(_type_matches(value, choice) for choice in arguments)
    if origin in (list, tuple):
        return isinstance(value, origin) and all(
            _type_matches(item, arguments[0]) for item in value
        )
    if origin is dict:
        return isinstance(value, dict) and all(
            _type_matches(key, arguments[0]) and _type_matches(item, arguments[1])
            for key, item in value.items()
        )
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected in (int, bool, str):
        return type(value) is expected
    return isinstance(value, expected)


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    # Normalize sequence-valued model fields for stable serialization/equality.
    values = dict(values)
    for name in ("hidden_dims", "deep_hidden_dims"):
        if name in values:
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"{cls.__name__}.{name} must be a sequence")
            values[name] = tuple(values[name])
    types = get_type_hints(cls)
    for name, value in values.items():
        if not _type_matches(value, types[name]):
            raise ValueError(f"{cls.__name__}.{name} has invalid type {type(value).__name__}")
    return cls(**values)


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    """Load an effective/sparse mapping, rejecting all misspelled keys."""
    values = dict(values)
    nested = {
        "sources": SourceConfig,
        "split": SplitConfig,
        "features": FeatureConfig,
        "vae": VAEConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "vae_training": TrainingConfig,
        "evaluation": EvaluationConfig,
    }
    for key, cls in nested.items():
        if key in values:
            values[key] = _construct(cls, values[key])
    result = _construct(ExperimentConfig, values)
    result.validate()
    return result


def load_config(path: Path) -> ExperimentConfig:
    """Load YAML; relative paths are resolved against the current working directory."""
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config_from_dict(values)
