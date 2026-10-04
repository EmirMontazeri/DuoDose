"""Reverse drug-combination dose prediction architecture and probabilistic objectives.

All inputs are floating point batches. Cell/drug metadata have shape ``[B, N]``,
inhibition has shape ``[B, 1]``, and transformed dose targets have shape ``[B, 2]``.
The output distribution is over transformed doses; physical-unit conversions
belong to the fitted preprocessing state. The architecture guarantees swapped paired
representations, but its unrestricted MDN head does not guarantee output symmetry.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class VAEConfig:
    """Offline VAE widths; the decoder mirrors the encoder's hidden layers."""

    hidden_dims: tuple[int, ...] = (512, 256)
    latent_dim: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        for dimension in self.hidden_dims:
            _positive_integer(dimension, "VAE hidden dimension")
        _positive_integer(self.latent_dim, "latent_dim")


@dataclass(frozen=True)
class ModelConfig:
    """Widths and numerical covariance safeguards for the model."""

    encoder_dim: int = 64
    encoder_hidden_dim: int = 128
    mf_dim: int = 32
    deep_hidden_dims: tuple[int, ...] = (256, 128)
    deep_dim: int = 128
    mixture_components: int = 5
    min_sigma: float = 1e-4
    rho_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "deep_hidden_dims", tuple(self.deep_hidden_dims))
        for name in (
            "encoder_dim",
            "encoder_hidden_dim",
            "mf_dim",
            "deep_dim",
            "mixture_components",
        ):
            _positive_integer(getattr(self, name), name)
        for dimension in self.deep_hidden_dims:
            _positive_integer(dimension, "Deep MLP hidden dimension")
        if self.deep_dim % 2:
            raise ValueError(
                "deep_dim must be even: each drug order produces deep_dim / 2 features"
            )
        if not math.isfinite(self.min_sigma) or self.min_sigma <= 0:
            raise ValueError("min_sigma must be finite and strictly positive")
        if not math.isfinite(self.rho_epsilon) or not 0 < self.rho_epsilon < 1:
            raise ValueError("rho_epsilon must be finite and in (0, 1)")


def _finite_float(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise ValueError(f"{name} must be a floating point torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains nonfinite values")


def _matrix(tensor: Tensor, width: int, name: str) -> None:
    _finite_float(tensor, name)
    if tensor.ndim != 2 or tensor.shape[1] != width or tensor.shape[0] == 0:
        raise ValueError(
            f"{name} must have shape [B, {width}] with B > 0; got {tuple(tensor.shape)}"
        )


def _same_device_dtype(reference: Tensor, tensor: Tensor, name: str) -> None:
    if tensor.device != reference.device or tensor.dtype != reference.dtype:
        raise ValueError(f"{name} must use device {reference.device} and dtype {reference.dtype}")


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class VAEEncoder(nn.Module):
    """Diagonal Gaussian posterior encoder returning ``(mean, log_variance)``."""

    def __init__(self, input_dim: int, config: VAEConfig) -> None:
        super().__init__()
        _positive_integer(input_dim, "input_dim")
        self.input_dim = input_dim
        self.config = config
        layers: list[nn.Module] = []
        previous = input_dim
        for width in config.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.hidden = nn.Sequential(*layers)
        self.mean = nn.Linear(previous, config.latent_dim)
        self.log_variance = nn.Linear(previous, config.latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``[B,input_dim]`` metadata to two ``[B,latent_dim]`` tensors."""
        _matrix(x, self.input_dim, "cell metadata")
        _same_device_dtype(self.mean.weight, x, "cell metadata")
        hidden = self.hidden(x)
        mean, log_variance = self.mean(hidden), self.log_variance(hidden)
        _finite_float(mean, "VAE posterior mean")
        _finite_float(log_variance, "VAE posterior log variance")
        return mean, log_variance


class VariationalAutoencoder(nn.Module):
    """Standalone Gaussian VAE, optimized before final model construction.

    ``forward`` uses posterior reparameterization in both training and evaluation;
    ``encode_mean`` supplies deterministic features for the final architecture.
    """

    def __init__(self, input_dim: int, config: VAEConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.encoder = VAEEncoder(input_dim, config)
        self.decoder = _mlp(config.latent_dim, tuple(reversed(config.hidden_dims)), input_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return reconstructed metadata, posterior mean, and log variance."""
        mean, log_variance = self.encoder(x)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        reconstruction = self.decoder(latent)
        _finite_float(reconstruction, "VAE reconstruction")
        return reconstruction, mean, log_variance

    def encode_mean(self, x: Tensor) -> Tensor:
        """Return the deterministic posterior mean without sampling."""
        return self.encoder(x)[0]


def vae_loss(
    reconstruction: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1.0
) -> dict[str, Tensor]:
    """Compute mean negative ELBO, omitting the constant Gaussian normalizer.

    Reconstruction uses ``0.5 * sum_features((reconstruction - x)**2)``;
    KL uses the analytic diagonal Gaussian divergence to ``N(0, I)``.
    Both terms are summed per cell, then averaged over cells. ``beta=1`` is
    the documented baseline; changing beta is an explicit experiment choice.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    for name, tensor in (
        ("reconstruction", reconstruction),
        ("x", x),
        ("mu", mu),
        ("logvar", logvar),
    ):
        _finite_float(tensor, name)
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            raise ValueError(f"{name} must be a nonempty rank-two batch")
        _same_device_dtype(x, tensor, name)
    if reconstruction.shape != x.shape or mu.shape != logvar.shape or mu.shape[0] != x.shape[0]:
        raise ValueError(
            "VAE reconstruction/input and posterior tensor dimensions are incompatible"
        )
    reconstruction_term = 0.5 * (reconstruction - x).square().sum(dim=1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    loss = reconstruction_term + beta * kl
    _finite_float(loss, "VAE loss")
    return {"loss": loss, "reconstruction": reconstruction_term, "kl": kl}


@dataclass
class MixtureParameters:
    """Bivariate Gaussian mixture tensors, preserving their autograd graphs.

    ``logits``: ``[B,M]``; ``means`` and ``scales``: ``[B,M,2]``;
    ``correlations``: ``[B,M]``. Scales are standard deviations, not variances.
    All fields must share dtype/device, and covariances must be nonsingular.
    """

    logits: Tensor
    means: Tensor
    scales: Tensor
    correlations: Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check tensor schemas and covariance-domain invariants."""
        for name in ("logits", "means", "scales", "correlations"):
            tensor = getattr(self, name)
            _finite_float(tensor, name)
            _same_device_dtype(self.logits, tensor, name)
        if self.logits.ndim != 2 or min(self.logits.shape) < 1:
            raise ValueError("logits must have shape [B, M] with B, M > 0")
        if self.means.shape != (*self.logits.shape, 2) or self.scales.shape != self.means.shape:
            raise ValueError("means and scales must have shape [B, M, 2]")
        if self.correlations.shape != self.logits.shape:
            raise ValueError("correlations must have shape [B, M]")
        if not (self.scales > 0).all():
            raise ValueError("All Gaussian standard deviations must be positive")
        if not (self.correlations.abs() < 1).all():
            raise ValueError("All Gaussian correlations must lie strictly inside (-1, 1)")

    @property
    def weights(self) -> Tensor:
        """Normalized component probabilities, shape ``[B,M]``."""
        return torch.softmax(self.logits, dim=-1)

    @property
    def mixture_mean(self) -> Tensor:
        """Distribution mean, shape ``[B,2]``; this need not be a mode."""
        return (self.weights.unsqueeze(-1) * self.means).sum(dim=1)


class ReverseDoseModel(nn.Module):
    """Exact two-path architecture with a permanently frozen VAE encoder.

    A supplied encoder is copied, leaving the standalone VAE unchanged. Omitting
    it creates an encoder for subsequent checkpoint restoration; the training
    workflow must supply the trained encoder when beginning a new experiment.
    No VAE decoder is retained. The MDN head is intentionally unrestricted.
    """

    def __init__(
        self,
        cell_dim: int,
        drug_dim: int,
        vae_config: VAEConfig,
        model_config: ModelConfig,
        pretrained_encoder: VAEEncoder | None = None,
    ) -> None:
        super().__init__()
        _positive_integer(cell_dim, "cell_dim")
        _positive_integer(drug_dim, "drug_dim")
        self.cell_dim, self.drug_dim = cell_dim, drug_dim
        self.vae_config, self.model_config = vae_config, model_config
        if pretrained_encoder is not None:
            if not isinstance(pretrained_encoder, VAEEncoder):
                raise ValueError("pretrained_encoder must be a VAEEncoder")
            if pretrained_encoder.input_dim != cell_dim or pretrained_encoder.config != vae_config:
                raise ValueError(
                    "Pretrained VAE encoder is incompatible with the requested cell dimension/configuration"
                )
            self.vae_encoder = copy.deepcopy(pretrained_encoder)
        else:
            self.vae_encoder = VAEEncoder(cell_dim, vae_config)
        self.vae_encoder.requires_grad_(False)
        self.vae_encoder.eval()

        embedding = model_config.encoder_dim
        hidden = (model_config.encoder_hidden_dim,)
        rank = model_config.mf_dim
        self.cell_encoder = _mlp(vae_config.latent_dim, hidden, embedding)
        self.inhibition_encoder = nn.Sequential(nn.Linear(1, embedding), nn.ReLU())
        self.drug_encoder = _mlp(drug_dim, hidden, embedding)
        self.cell_factor_1 = nn.Linear(embedding, rank)
        self.cell_factor_2 = nn.Linear(embedding, rank)
        self.inhibition_factor_1 = nn.Linear(embedding, rank)
        self.inhibition_factor_2 = nn.Linear(embedding, rank)
        self.drug_pair_factor = nn.Linear(2 * embedding, rank)
        self.deep_mlp = _mlp(
            4 * embedding, model_config.deep_hidden_dims, model_config.deep_dim // 2
        )
        fusion_dim = 2 * rank + model_config.deep_dim
        components = model_config.mixture_components
        self.weight_head = nn.Linear(fusion_dim, components)
        self.mean_head = nn.Linear(fusion_dim, 2 * components)
        self.covariance_head = nn.Linear(fusion_dim, 3 * components)

    def train(self, mode: bool = True) -> ReverseDoseModel:
        """Set trainable modules' mode while keeping the VAE encoder in eval."""
        super().train(mode)
        self.vae_encoder.eval()
        self.vae_encoder.requires_grad_(False)
        return self

    def representations(
        self, cell: Tensor, inhibition: Tensor, drug_a: Tensor, drug_b: Tensor
    ) -> dict[str, Tensor]:
        """Return ``z1,z2,h_a,h_b,fusion`` with the model's exact equations."""
        for name, tensor, width in (
            ("cell", cell, self.cell_dim),
            ("inhibition", inhibition, 1),
            ("drug_a", drug_a, self.drug_dim),
            ("drug_b", drug_b, self.drug_dim),
        ):
            _matrix(tensor, width, name)
            _same_device_dtype(cell, tensor, name)
            if tensor.shape[0] != cell.shape[0]:
                raise ValueError("Cell, inhibition, and drug batch sizes must match")
        with torch.no_grad():
            cell_latent = self.vae_encoder(cell)[0]
        e_cell = self.cell_encoder(cell_latent)
        e_x = self.inhibition_encoder(inhibition)
        e_a, e_b = self.drug_encoder(drug_a), self.drug_encoder(drug_b)
        u1 = self.cell_factor_1(e_cell) * self.inhibition_factor_1(e_x)
        u2 = self.cell_factor_2(e_cell) * self.inhibition_factor_2(e_x)
        d1 = self.drug_pair_factor(torch.cat((e_a, e_b), dim=-1))
        d2 = self.drug_pair_factor(torch.cat((e_b, e_a), dim=-1))
        z1 = u1 * d1 + u2 * d2
        z2 = u1 * d2 + u2 * d1
        h_a = self.deep_mlp(torch.cat((e_cell, e_x, e_a, e_b), dim=-1))
        h_b = self.deep_mlp(torch.cat((e_cell, e_x, e_b, e_a), dim=-1))
        fusion = torch.cat((z1, z2, h_a, h_b), dim=-1)
        _finite_float(fusion, "Fusion representation")
        return {"z1": z1, "z2": z2, "h_a": h_a, "h_b": h_b, "fusion": fusion}

    def forward(
        self, cell: Tensor, inhibition: Tensor, drug_a: Tensor, drug_b: Tensor
    ) -> MixtureParameters:
        """Predict ``M`` correlated bivariate components for each input row."""
        fusion = self.representations(cell, inhibition, drug_a, drug_b)["fusion"]
        components = self.model_config.mixture_components
        means = self.mean_head(fusion).reshape(-1, components, 2)
        covariance = self.covariance_head(fusion).reshape(-1, components, 3)
        scales = F.softplus(covariance[..., :2]) + self.model_config.min_sigma
        margin = max(self.model_config.rho_epsilon, torch.finfo(covariance.dtype).eps)
        correlations = torch.tanh(covariance[..., 2]) * (1 - margin)
        return MixtureParameters(self.weight_head(fusion), means, scales, correlations)


def mixture_log_prob(params: MixtureParameters, target: Tensor) -> Tensor:
    """Evaluate stable correlated Gaussian mixture log densities, shape ``[B]``.

    The quadratic form is evaluated via a Cholesky-style expression to avoid
    cancellation near high correlations. Low-precision values are promoted to
    float32 for density arithmetic. Densities are in the target coordinate system.
    """
    params.validate()
    _matrix(target, 2, "target")
    _same_device_dtype(params.logits, target, "target")
    if target.shape[0] != params.logits.shape[0]:
        raise ValueError("Target batch size must match mixture batch size")
    dtype = torch.float32 if target.dtype in (torch.float16, torch.bfloat16) else target.dtype
    means, scales = params.means.to(dtype), params.scales.to(dtype)
    rho = params.correlations.to(dtype)
    normalized = (target.to(dtype).unsqueeze(1) - means) / scales
    first, second = normalized.unbind(dim=-1)
    residual_variance = (1 - rho) * (1 + rho)
    quadratic = (first - rho * second).square() / residual_variance + second.square()
    component_log_prob = (
        -math.log(2 * math.pi)
        - scales.log().sum(dim=-1)
        - 0.5 * residual_variance.log()
        - 0.5 * quadratic
    )
    result = torch.logsumexp(
        F.log_softmax(params.logits.to(dtype), dim=-1) + component_log_prob, dim=-1
    )
    _finite_float(result, "Mixture log probability")
    return result


def mixture_nll(
    params: MixtureParameters,
    target: Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Negative mixture log likelihood, reduced over observations as requested."""
    values = -mixture_log_prob(params, target)
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean', 'sum', or 'none'")


@torch.no_grad()
def sample_mixture(
    params: MixtureParameters, n_samples: int, generator: torch.Generator | None = None
) -> Tensor:
    """Draw ``[B,n_samples,2]`` transformed-dose samples without truncation.

    The optional random generator must be on the same device as the parameters.
    Samples inherit the Gaussian's unbounded support; physical validity is an
    inference policy handled separately, not an implicit modification here.
    """
    params.validate()
    _positive_integer(n_samples, "n_samples")
    indices = torch.multinomial(params.weights, n_samples, replacement=True, generator=generator)
    selected_means = params.means.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
    selected_scales = params.scales.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
    rho = params.correlations.gather(1, indices)
    noise = torch.randn(
        selected_means.shape,
        dtype=selected_means.dtype,
        device=selected_means.device,
        generator=generator,
    )
    first, independent = noise.unbind(dim=-1)
    second = rho * first + torch.sqrt((1 - rho) * (1 + rho)) * independent
    samples = selected_means + selected_scales * torch.stack((first, second), dim=-1)
    _finite_float(samples, "Mixture samples")
    return samples
