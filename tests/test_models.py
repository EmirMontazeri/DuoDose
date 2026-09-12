"""Mathematical and architectural invariants of the proposal implementation."""

import io
import math

import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from reverse_dose.models import (
    MixtureParameters,
    ModelConfig,
    ReverseDoseModel,
    VAEConfig,
    VariationalAutoencoder,
    mixture_log_prob,
    mixture_nll,
    sample_mixture,
    vae_loss,
)


def small_model() -> ReverseDoseModel:
    torch.manual_seed(19)
    return ReverseDoseModel(
        cell_dim=7,
        drug_dim=5,
        vae_config=VAEConfig(hidden_dims=(9,), latent_dim=3),
        model_config=ModelConfig(
            encoder_dim=4,
            encoder_hidden_dim=6,
            mf_dim=3,
            deep_hidden_dims=(8,),
            deep_dim=6,
            mixture_components=2,
        ),
    )


def inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    return tuple(torch.randn(4, width, generator=generator) for width in (7, 1, 5, 5))


def test_log_probability_matches_torch_full_covariance() -> None:
    torch.manual_seed(10)
    batch, components = 6, 4
    logits = torch.randn(batch, components, dtype=torch.float64, requires_grad=True)
    means = torch.randn(batch, components, 2, dtype=torch.float64, requires_grad=True)
    scales = torch.exp(torch.randn(batch, components, 2, dtype=torch.float64))
    rho = torch.tanh(torch.randn(batch, components, dtype=torch.float64)) * 0.999
    params = MixtureParameters(logits, means, scales, rho)
    covariance = torch.diag_embed(scales.square())
    cross_term = scales[..., 0] * scales[..., 1] * rho
    covariance[..., 0, 1] = cross_term
    covariance[..., 1, 0] = cross_term
    reference = MixtureSameFamily(
        Categorical(logits=logits),
        MultivariateNormal(means, covariance_matrix=covariance),
    )
    target = torch.randn(batch, 2, dtype=torch.float64)
    actual = mixture_log_prob(params, target)
    torch.testing.assert_close(actual, reference.log_prob(target), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(params.mixture_mean, reference.mean)
    torch.testing.assert_close(mixture_nll(params, target, "sum"), -actual.sum())
    torch.testing.assert_close(mixture_nll(params, target, "none"), -actual)


def test_vae_loss_matches_closed_form_and_has_gradients() -> None:
    x = torch.zeros(2, 4, dtype=torch.float64)
    reconstruction = torch.ones_like(x, requires_grad=True)
    mu = torch.full((2, 3), 2.0, dtype=torch.float64, requires_grad=True)
    logvar = torch.full((2, 3), math.log(2.0), dtype=torch.float64, requires_grad=True)
    result = vae_loss(reconstruction, x, mu, logvar)
    assert result["reconstruction"].item() == pytest.approx(2.0)
    assert result["kl"].item() == pytest.approx(1.5 * (5.0 - math.log(2.0)))
    result["loss"].backward()
    assert all(torch.isfinite(value.grad).all() for value in (reconstruction, mu, logvar))


def test_vae_posterior_sampling_and_deterministic_mean() -> None:
    vae = VariationalAutoencoder(7, VAEConfig(hidden_dims=(6,), latent_dim=3))
    x = inputs()[0]
    reconstruction, mean, logvar = vae(x)
    assert reconstruction.shape == x.shape
    assert mean.shape == logvar.shape == (4, 3)
    torch.testing.assert_close(vae.encode_mean(x), mean)
    torch.testing.assert_close(vae.encode_mean(x), vae.encode_mean(x), atol=0, rtol=0)
    assert not torch.equal(reconstruction, vae(x)[0])


def test_exact_paired_representations_swap() -> None:
    model = small_model()
    cell, inhibition, a, b = inputs()
    original = model.representations(cell, inhibition, a, b)
    swapped = model.representations(cell, inhibition, b, a)
    for left, right in (("z1", "z2"), ("z2", "z1"), ("h_a", "h_b"), ("h_b", "h_a")):
        torch.testing.assert_close(original[left], swapped[right], atol=0, rtol=0)
    assert original["fusion"].shape == (4, 12)
    torch.testing.assert_close(
        original["fusion"],
        torch.cat([original[key] for key in ("z1", "z2", "h_a", "h_b")], dim=-1),
    )


def test_frozen_encoder_stays_frozen_but_downstream_learns() -> None:
    model = small_model()
    before = {key: value.clone() for key, value in model.vae_encoder.state_dict().items()}
    model.train()
    assert model.training and not model.vae_encoder.training
    assert not any(parameter.requires_grad for parameter in model.vae_encoder.parameters())
    result = model(*inputs())
    assert result.means.shape == result.scales.shape == (4, 2, 2)
    assert result.weights.shape == result.correlations.shape == (4, 2)
    torch.testing.assert_close(result.weights.sum(-1), torch.ones(4))
    loss = mixture_nll(result, torch.zeros(4, 2))
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("vae_encoder."):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    for key, value in model.vae_encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_encoder_transfer_copies_and_checks_configuration() -> None:
    config = VAEConfig(hidden_dims=(9,), latent_dim=3)
    vae = VariationalAutoencoder(7, config)
    model = ReverseDoseModel(7, 5, config, ModelConfig(), vae.encoder)
    assert model.vae_encoder is not vae.encoder
    assert all(parameter.requires_grad for parameter in vae.encoder.parameters())
    assert not hasattr(model, "decoder")
    torch.testing.assert_close(model.vae_encoder(inputs()[0])[0], vae.encode_mean(inputs()[0]))
    with pytest.raises(ValueError, match="incompatible"):
        ReverseDoseModel(8, 5, config, ModelConfig(), vae.encoder)


def test_unrestricted_head_does_not_claim_final_swap_equivariance() -> None:
    model = small_model()
    with torch.no_grad():
        model.mean_head.weight.zero_()
        model.mean_head.bias.copy_(torch.tensor([1.0, 2.0, 1.0, 2.0]))
    cell, inhibition, a, b = inputs()
    original = model(cell, inhibition, a, b)
    swapped = model(cell, inhibition, b, a)
    assert not torch.allclose(original.mixture_mean, swapped.mixture_mean.flip(-1))


def test_state_dict_roundtrip_is_exact() -> None:
    model = small_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = small_model()
    restored.load_state_dict(torch.load(payload, weights_only=True), strict=True)
    original, reloaded = model(*inputs()), restored(*inputs())
    for name in ("logits", "means", "scales", "correlations"):
        torch.testing.assert_close(getattr(original, name), getattr(reloaded, name), atol=0, rtol=0)


def test_sample_moments_and_reproducible_generator() -> None:
    params = MixtureParameters(
        torch.zeros(1, 1),
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[2.0, 3.0]]]),
        torch.tensor([[0.7]]),
    )
    samples = sample_mixture(params, 50000, torch.Generator().manual_seed(83))
    repeated = sample_mixture(params, 50000, torch.Generator().manual_seed(83))
    torch.testing.assert_close(samples, repeated, rtol=0, atol=0)
    torch.testing.assert_close(samples.mean(dim=1), params.mixture_mean, atol=0.04, rtol=0)
    covariance = torch.cov(samples[0].T)
    torch.testing.assert_close(
        covariance, torch.tensor([[4.0, 4.2], [4.2, 9.0]]), atol=0.12, rtol=0
    )


def test_extreme_valid_covariance_has_finite_loss_and_gradients() -> None:
    logits = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
    means = torch.zeros(1, 2, 2, requires_grad=True)
    scales = torch.full((1, 2, 2), 1e-4, requires_grad=True)
    rho = torch.tensor([[0.99999, -0.99999]], requires_grad=True)
    loss = mixture_nll(
        MixtureParameters(logits, means, scales, rho), torch.tensor([[0.001, 0.001]])
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(value.grad).all() for value in (logits, means, scales, rho))


@pytest.mark.parametrize(
    "kwargs",
    [{"deep_dim": 3}, {"mf_dim": 0}, {"rho_epsilon": 0}, {"min_sigma": float("nan")}],
)
def test_invalid_config_fails(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_invalid_shapes_and_values_fail_explicitly() -> None:
    model = small_model()
    cell, inhibition, a, b = inputs()
    with pytest.raises(ValueError, match="shape"):
        model(cell, inhibition.squeeze(-1), a, b)
    with pytest.raises(ValueError, match="batch sizes"):
        model(cell, inhibition[:1], a, b)
    cell[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model(cell, inhibition, a, b)
    with pytest.raises(ValueError, match="strictly inside"):
        MixtureParameters(
            torch.zeros(1, 1),
            torch.zeros(1, 1, 2),
            torch.ones(1, 1, 2),
            torch.ones(1, 1),
        )
