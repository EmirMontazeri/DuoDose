"""Independent VAE/final training with exact minibatch checkpoint resumption.

Checkpoint state records the active permutation, next minibatch, partial epoch
totals, optimizer/scheduler, best validation score, early-stop counter and every
RNG. An interrupt saves before returning; partial optimizer updates are never
represented as a completed minibatch. Resume validates experiment identity.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import pickle
import signal
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, default_collate

from .config import ExperimentConfig, TrainingConfig
from .data import DataBundle
from .models import ReverseDoseModel, VariationalAutoencoder, mixture_nll, vae_loss
from .utils import (
    atomic_json,
    atomic_torch_save,
    environment_info,
    event,
    fingerprint,
    guard_output,
    load_checkpoint,
    resolve_device,
    restore_rng,
    rng_state,
    seed_everything,
    sha256_file,
    read_json,
)

LOGGER = logging.getLogger(__name__)


def _state_fingerprint(parameters: dict[str, torch.Tensor]) -> str:
    """Hash named tensor values independently of pickle/storage serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        value = tensor.detach().cpu().contiguous()
        descriptor = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        digest.update(fingerprint(descriptor).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _encoder_fingerprint(checkpoint: dict[str, Any]) -> str:
    """Bind transferred encoder values and lineage, excluding training counters."""
    encoder = {
        name: value
        for name, value in checkpoint["model_state"].items()
        if name.startswith("encoder.")
    }
    if not encoder:
        raise ValueError("Offline checkpoint contains no VAE encoder parameters")
    return fingerprint(
        {
            "encoder_parameters": _state_fingerprint(encoder),
            "vae_config": checkpoint["config"]["vae"],
            "dataset_fingerprint": checkpoint["dataset_fingerprint"],
        }
    )


class MetadataDataset(Dataset):
    """Metadata-only VAE model cohort; never loads response labels."""

    def __init__(self, features: np.ndarray, indices: np.ndarray):
        self.features = torch.as_tensor(features[indices], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]


def _batch(dataset: Dataset, indices: torch.Tensor, device: torch.device) -> Any:
    value = default_collate([dataset[int(i)] for i in indices])
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return {key: tensor.to(device) for key, tensor in value.items()}


def _loss(
    model: nn.Module, batch: Any, stage: str, beta: float, validation: bool = False
) -> dict[str, torch.Tensor]:
    if stage == "vae":
        if validation:
            # Fixed posterior-mean reconstruction gives reproducible selection.
            mean, logvar = model.encoder(batch)
            reconstruction = model.decoder(mean)
        else:
            reconstruction, mean, logvar = model(batch)
        return vae_loss(reconstruction, batch, mean, logvar, beta=beta)
    parameters = model(batch["cell"], batch["inhibition"], batch["drug_a"], batch["drug_b"])
    weights = parameters.weights
    # These metrics do not alter the model's exact MDN NLL objective. They
    # expose the characteristic signature of conditional Gaussian collapse:
    # improving train NLL accompanied by shrinking covariance volume and poor
    # held-out NLL. Means are batch-additive, so epoch aggregation is exact.
    covariance_log_determinant = (
        2 * parameters.scales.log().sum(dim=-1)
        + ((1 - parameters.correlations) * (1 + parameters.correlations)).log()
    )
    point_rmse = (parameters.mixture_mean - batch["target"]).square().mean(dim=-1).sqrt()
    return {
        "loss": mixture_nll(parameters, batch["target"]),
        "mean_log_scale": parameters.scales.log().mean(),
        "mean_covariance_log_determinant": covariance_log_determinant.mean(),
        "mean_mixture_entropy": -(weights * weights.clamp_min(1e-30).log()).sum(-1).mean(),
        "mixture_mean_rmse": point_rmse.mean(),
    }


@torch.no_grad()
def _validate(
    model: nn.Module,
    dataset: Dataset,
    settings: TrainingConfig,
    device: torch.device,
    stage: str,
    beta: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for start in range(0, len(dataset), settings.batch_size):
        indices = torch.arange(start, min(start + settings.batch_size, len(dataset)))
        losses = _loss(model, _batch(dataset, indices, device), stage, beta, validation=True)
        for key, value in losses.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite validation {key} in {stage}")
            totals[key] = totals.get(key, 0.0) + value.item() * len(indices)
    return {key: value / len(dataset) for key, value in totals.items()}


def make_model(
    bundle: DataBundle,
    config: ExperimentConfig,
    stage: str,
    vae_checkpoint: Path | None = None,
) -> nn.Module:
    """Construct one stage, verifying offline-encoder lineage before transfer."""
    cell_dim = bundle.cell_features.shape[1]
    if stage == "vae":
        return VariationalAutoencoder(cell_dim, config.vae)
    if stage != "model":
        raise ValueError("stage must be vae or model")
    encoder = None
    if vae_checkpoint is not None:
        checkpoint = load_checkpoint(vae_checkpoint)
        if checkpoint["stage"] != "vae" or checkpoint["dataset_fingerprint"] != (
            bundle.manifest["fingerprint"]
        ):
            raise ValueError("Offline VAE belongs to a different stage or prepared dataset")
        if not checkpoint.get("selection_only", False):
            raise ValueError("Offline VAE transfer requires its selected best.pt checkpoint")
        if fingerprint(checkpoint["config"]["vae"]) != fingerprint(asdict(config.vae)):
            raise ValueError("Offline VAE architecture does not match configuration")
        vae = VariationalAutoencoder(cell_dim, config.vae)
        vae.load_state_dict(checkpoint["model_state"], strict=True)
        encoder = copy.deepcopy(vae.encoder)
    return ReverseDoseModel(
        cell_dim,
        bundle.drug_features.shape[1],
        config.vae,
        config.model,
        pretrained_encoder=encoder,
    )


def train_stage(
    bundle: DataBundle,
    config: ExperimentConfig,
    run_dir: Path,
    stage: str,
    *,
    vae_checkpoint: Path | None = None,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
) -> Path:
    """Train a stage and return its selected checkpoint.

    ``stop_after_steps`` supports controlled interruption/restart checks and
    returns ``last.pt`` while interrupted. Normal completion returns ``best.pt``.
    An existing stage directory requires explicit ``resume``. Scientific settings
    and the configured total epochs cannot silently change during resumption.
    VAE checkpoint selection uses deterministic posterior-mean reconstruction
    plus KL on metadata-only validation cells; this is a selection surrogate,
    not a Monte Carlo estimate of the expected training reconstruction loss.
    """
    if stage not in {"vae", "model"}:
        raise ValueError("Unknown training stage")
    guard_output(run_dir, Path(config.dataset_root))
    if (
        bundle.manifest.get("dataset_name") is not None
        and bundle.manifest["dataset_name"] != config.dataset_name
    ):
        raise ValueError("Training configuration and prepared dataset names differ")
    if (
        bundle.manifest.get("sources") is not None
        and bundle.manifest["sources"] != config.sources.sources
    ):
        raise ValueError("Training configuration and prepared response sources differ")
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive")
    settings = config.vae_training if stage == "vae" else config.training
    directory = run_dir / stage
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError(f"Stage already exists; explicitly resume {directory / 'last.pt'}")
    if stage == "model" and vae_checkpoint is None:
        raise ValueError("Final training requires a selected offline VAE checkpoint")
    seed_everything(config.seed + (stage == "model"), config.deterministic, config.threads)
    device = resolve_device(config.device)
    model = make_model(bundle, config, stage, vae_checkpoint).to(device)
    if stage == "vae":
        train_data = MetadataDataset(bundle.vae_features, bundle.vae_train_indices)
        validation_data = MetadataDataset(bundle.vae_features, bundle.vae_validation_indices)
    else:
        train_data, validation_data = bundle.dataset("train"), bundle.dataset("validation")
    if not len(train_data) or not len(validation_data):
        raise ValueError("Both training and validation cohorts must be nonempty")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
    )
    env = environment_info()
    identity = {
        "stage": stage,
        "dataset_fingerprint": bundle.manifest["fingerprint"],
        "config": config.to_dict(),
        "environment": env,
        "vae_encoder_fingerprint": (
            _encoder_fingerprint(load_checkpoint(vae_checkpoint)) if vae_checkpoint else None
        ),
    }
    # The file hash records provenance, while experiment continuity depends on
    # immutable encoder values. Reconstructing best.pt can change its optimizer
    # counters/serialization without changing the encoder used by the final model.
    provenance = {"vae_sha256": sha256_file(vae_checkpoint) if vae_checkpoint else None}
    experiment_id = fingerprint(identity)
    generator = torch.Generator().manual_seed(config.seed + 1001 + (stage == "model"))
    state: dict[str, Any] = {
        "epoch": 0,
        "next_batch": 0,
        "global_step": 0,
        "permutation": None,
        "epoch_totals": {},
        "epoch_count": 0,
        "best_validation": math.inf,
        "bad_epochs": 0,
        "completed": False,
        "best_model_state": None,
        "best_epoch": None,
        "best_step": None,
    }
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        if checkpoint.get("selection_only", False):
            raise ValueError(
                "best.pt is for selection/inference; resume from last.pt or step/epoch checkpoints"
            )
        if checkpoint["experiment_id"] != experiment_id:
            raise ValueError(
                "Resume rejected: dataset, configuration, encoder, source code or "
                "runtime differs from the checkpoint. Use the original experiment."
            )
        if directory.exists() and any(directory.iterdir()):
            metadata_path = directory / "experiment.json"
            if not metadata_path.exists():
                raise FileExistsError("Resume destination is nonempty and lacks experiment.json")
            if read_json(metadata_path).get("experiment_id") != experiment_id:
                raise ValueError("Resume destination belongs to a different experiment")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        state = checkpoint["progress"]
        generator.set_state(checkpoint["sampler_generator_state"])
        restore_rng(checkpoint["rng_state"])
        LOGGER.info(
            "Restored %s: epoch=%d next_batch=%d step=%d",
            resume,
            state["epoch"],
            state["next_batch"],
            state["global_step"],
        )
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        directory / "experiment.json",
        {**identity, **provenance, "experiment_id": experiment_id},
    )
    log_path = directory / "events.jsonl"
    LOGGER.info(
        "%s initialized on %s: %d train / %d validation examples, %d trainable params",
        stage,
        device,
        len(train_data),
        len(validation_data),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    event(
        log_path,
        {
            "event": "resume" if resume else "start",
            "stage": stage,
            "step": state["global_step"],
            "experiment_id": experiment_id,
        },
    )

    def save(path: Path, *, selected: bool = False) -> None:
        # Copying best weights to CPU avoids accelerator memory growth across epochs.
        model_state = state["best_model_state"] if selected else model.state_dict()
        checkpoint = {
            "format_version": 1,
            **identity,
            **provenance,
            "experiment_id": experiment_id,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "progress": state,
            "rng_state": rng_state(),
            "sampler_generator_state": generator.get_state(),
            "selection_only": selected,
        }
        atomic_torch_save(path, checkpoint)
        LOGGER.info("Checkpoint saved: %s (step=%d)", path, state["global_step"])
        event(
            log_path,
            {
                "event": "checkpoint",
                "path": path.name,
                "step": state["global_step"],
                "epoch": state["epoch"],
            },
        )

    def synchronize_selection() -> None:
        """Keep a matching selection artifact byte-for-byte, repairing stale ones."""
        path = directory / "best.pt"
        if state["best_model_state"] is None:
            if path.exists():
                path.unlink()
                event(
                    log_path,
                    {"event": "discard_stale_selection", "step": state["global_step"]},
                )
            return
        if path.exists():
            try:
                selected = load_checkpoint(path)
                matches = (
                    selected.get("selection_only", False)
                    and selected.get("experiment_id") == experiment_id
                    and _state_fingerprint(selected["model_state"])
                    == _state_fingerprint(state["best_model_state"])
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                EOFError,
                TypeError,
                AttributeError,
                pickle.UnpicklingError,
            ):
                matches = False
            if matches:
                return
            LOGGER.warning("Replacing stale or unreadable selected checkpoint %s", path)
        save(path, selected=True)

    # Signal handlers set a flag. Finish the active optimizer step, then save a
    # consistent minibatch boundary, rather than catching mid-update interrupts.
    interrupted = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        LOGGER.warning("Signal %s received; saving at the next minibatch boundary", signum)

    old_handlers = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
        if resume is not None:
            # A user may resume a periodic checkpoint into the same run
            # directory. The checkpoint is authoritative over any later artifacts.
            synchronize_selection()
        if state["completed"]:
            LOGGER.info("Stage already completed at epoch %d", state["epoch"])
            return directory / "best.pt"
        while state["epoch"] < settings.epochs:
            if state["permutation"] is None:
                state["permutation"] = torch.randperm(len(train_data), generator=generator)
            permutation = state["permutation"]
            n_batches = math.ceil(len(permutation) / settings.batch_size)
            model.train()
            for batch_number in range(state["next_batch"], n_batches):
                indices = permutation[
                    batch_number * settings.batch_size : (batch_number + 1) * settings.batch_size
                ]
                batch = _batch(train_data, indices, device)
                optimizer.zero_grad(set_to_none=True)
                losses = _loss(model, batch, stage, config.vae_beta)
                if any(not torch.isfinite(value).all() for value in losses.values()):
                    raise FloatingPointError(
                        f"Non-finite {stage} loss at step {state['global_step']}"
                    )
                losses["loss"].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    settings.gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                state["global_step"] += 1
                state["next_batch"] = batch_number + 1
                state["epoch_count"] += len(indices)
                for key, value in losses.items():
                    totals = state["epoch_totals"]
                    totals[key] = totals.get(key, 0.0) + value.item() * len(indices)
                if state["global_step"] % settings.log_every_steps == 0:
                    metrics = {key: value.item() for key, value in losses.items()}
                    LOGGER.info(
                        "%s epoch=%d batch=%d/%d step=%d loss=%.6f grad=%.4f lr=%.3g",
                        stage,
                        state["epoch"] + 1,
                        batch_number + 1,
                        n_batches,
                        state["global_step"],
                        metrics["loss"],
                        float(gradient_norm),
                        optimizer.param_groups[0]["lr"],
                    )
                    event(
                        log_path,
                        {
                            "event": "batch",
                            "epoch": state["epoch"] + 1,
                            "step": state["global_step"],
                            "metrics": metrics,
                            "gradient_norm": float(gradient_norm),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                    )
                if state["global_step"] % settings.checkpoint_every_steps == 0:
                    save(directory / "last.pt")
                if interrupted or (
                    stop_after_steps is not None and state["global_step"] >= stop_after_steps
                ):
                    save(directory / "last.pt")
                    event(log_path, {"event": "interrupted", "step": state["global_step"]})
                    return directory / "last.pt"
            validation = _validate(model, validation_data, settings, device, stage, config.vae_beta)
            training = {
                key: value / state["epoch_count"] for key, value in state["epoch_totals"].items()
            }
            improved = validation["loss"] < state["best_validation"]
            if improved:
                state["best_validation"] = validation["loss"]
                state["bad_epochs"] = 0
                state["best_model_state"] = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                state["best_epoch"] = state["epoch"] + 1
                state["best_step"] = state["global_step"]
            else:
                state["bad_epochs"] += 1
            scheduler.step(validation["loss"])
            state["epoch"] += 1
            state["next_batch"] = 0
            state["permutation"] = None
            state["epoch_totals"] = {}
            state["epoch_count"] = 0
            state["completed"] = (
                state["epoch"] >= settings.epochs
                or state["bad_epochs"] >= settings.early_stopping_patience
            )
            LOGGER.info(
                "%s epoch=%d train=%.6f validation=%.6f best=%.6f%s",
                stage,
                state["epoch"],
                training["loss"],
                validation["loss"],
                state["best_validation"],
                " selected" if improved else "",
            )
            diagnostic_keys = {
                "mean_log_scale",
                "mean_covariance_log_determinant",
                "mixture_mean_rmse",
            }
            if stage == "model" and diagnostic_keys <= validation.keys():
                LOGGER.info(
                    "model covariance diagnostics: train log_scale=%.6f log_det=%.6f "
                    "rmse=%.6f; validation log_scale=%.6f log_det=%.6f rmse=%.6f",
                    training["mean_log_scale"],
                    training["mean_covariance_log_determinant"],
                    training["mixture_mean_rmse"],
                    validation["mean_log_scale"],
                    validation["mean_covariance_log_determinant"],
                    validation["mixture_mean_rmse"],
                )
            event(
                log_path,
                {
                    "event": "epoch",
                    "epoch": state["epoch"],
                    "step": state["global_step"],
                    "training": training,
                    "validation": validation,
                    "selected": improved,
                },
            )
            if improved:
                save(directory / "best.pt", selected=True)
            save(directory / "last.pt")
            if state["epoch"] % settings.checkpoint_every_epochs == 0:
                save(directory / f"epoch_{state['epoch']:04d}.pt")
            if state["completed"]:
                break
        # Keep selected files stable across completed-stage resumption, while
        # recovering any absent/stale selection from authoritative progress.
        synchronize_selection()
        event(
            log_path,
            {
                "event": "complete",
                "epoch": state["epoch"],
                "best_validation": state["best_validation"],
            },
        )
        return directory / "best.pt"
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
