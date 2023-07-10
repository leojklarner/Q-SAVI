"""
Objective function for Q-SAVI. The objective function is the ELBO of the
induced distribution over functions, evaluated over the context points.
The ELBO is calculated as the sum of the expected log likelihood of the
data points, minus the KL divergence between the induced distribution
and the function space prior.
"""

from functools import partial
from typing import Any, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

from .linearization import induced_function_distribution

kl_cov_jitter = 1e-3


@partial(jit, static_argnums=(8, 9))
def qsavi_loss(
    params: hk.Params,
    state: hk.State,
    prior_mean: jnp.ndarray,
    prior_cov: jnp.ndarray,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    context_points: jnp.ndarray,
    rng_key: jnp.ndarray,
    n_samples: int,
    model,
) -> jnp.ndarray:
    """
    Q-SAVI ELBO classification loss for the current task.

    Args:
        params: model parameters.
        state: model state.
        prior_mean: mean tensor of the function space prior.
        prior_cov: covariance tensor of the function space prior.
        inputs: bit vector representation of input molecules.
        targets: binary classification targets.
        context_points: bit vector representation of context point samples.
        rng_key: random number generator key.
        n_samples: number of MC samples used to estimate the ELBO.
        model: model to be trained.

    Returns:
        negative Q-SAVI ELBO classification loss for the current task.
    """

    # split params into means and variances
    params_log_var, params_mean = hk.data_structures.partition(
        lambda module_name, name, value: "logvar" in name, params
    )

    # calculate KL divergence between induced distribution over functions
    # and function space prior, evaluated over the sampled context points

    induced_mean, induced_cov = induced_function_distribution(
        model.apply_fn,
        params_mean,
        params_log_var,
        state,
        context_points,
        rng_key,
    )

    kl_divergence = function_space_kl_divergence(
        induced_mean, prior_mean, induced_cov, prior_cov, 1e-6
    )

    # caclulate Bernoulli log-likelihood of MC-sampled means
    # (targets are broadcast in the MC sample dimension)

    preds_f_samples, _, _ = model.predict_f_multisample_jitted(
        params, state, inputs, rng_key, n_samples
    )
    log_preds_y_samples = jax.nn.log_softmax(preds_f_samples, axis=-1)

    log_likelihood = jnp.mean(
        jnp.sum(jnp.sum(targets * log_preds_y_samples, axis=-1), axis=-1), axis=0
    )

    elbo = log_likelihood - kl_divergence

    return -elbo


@partial(jit, static_argnums=(4))
def function_space_kl_divergence(
    mean_q: jnp.ndarray,
    mean_p: jnp.ndarray,
    cov_q: jnp.ndarray,
    cov_p: jnp.ndarray,
    noise: float,
) -> jnp.ndarray:
    """
    Return KL(q || p) for a batch of Gaussian distributions q and p.
    The covariance matrices are 4-dimensional tensors of shape
    (batch_size, output_dim, batch_size, output_dim) where any slice of [:, i, :, i]
    or [j, :, j, :] is an identity matrix to make sure that the covariance matrix is
    well-conditioned when taking the covariance between data points into account.

    Args:
        mean_q: mean tensor of Gaussian distribution q
                of shape (batch_size, output_dim).
        mean_p: mean tensor of Gaussian distribution p
                of shape (batch_size, output_dim).
        cov_q:  covariance tensor of Gaussian distribution q
                of shape (batch_size, output_dim, batch_size, output_dim).
        cov_p:  covariance tensor of Gaussian distribution p
                of shape (batch_size, output_dim, batch_size, output_dim).
        noise: noise added to the diagonal of the covariance matrices.

    Returns:
        KL divergence between q and p.

    """

    kl = 0
    for i in range(mean_q.shape[-1]):
        mean_q_i = jnp.squeeze(mean_q[:, i])
        mean_p_i = jnp.squeeze(mean_p[:, i])
        cov_q_i = cov_q[:, i, :, i]
        cov_p_i = cov_p[:, i, :, i]
        noise_matrix = jnp.eye(cov_q_i.shape[0]) * noise
        cov_q_i += noise_matrix
        cov_p_i += noise_matrix
        # print("reached KL Full Call")
        kl += kl_full_cov(mean_q_i, mean_p_i, cov_q_i, cov_p_i)

    return kl


@jit
def kl_full_cov(
    mean_q: jnp.ndarray,
    mean_p: jnp.ndarray,
    cov_q: jnp.ndarray,
    cov_p: jnp.ndarray,
) -> jnp.ndarray:
    """
    Standard KL divergence between two multivariate Gaussian distributions.

    Args:
        mean_q: mean tensor of Gaussian distribution q
                of shape (batch_size,).
        mean_p: mean tensor of Gaussian distribution p
                of shape (batch_size,).
        cov_q:  covariance tensor of Gaussian distribution q
                of shape (batch_size, batch_size).
        cov_p:  covariance tensor of Gaussian distribution p
                of shape (batch_size, batch_size).

    Returns:
        KL divergence between q and p.
    """

    dims = mean_q.shape[0]
    _cov_q = cov_q + jnp.eye(dims) * kl_cov_jitter
    _cov_p = cov_p + jnp.eye(dims) * kl_cov_jitter

    q = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean_q.transpose(),
        covariance_matrix=_cov_q,
        validate_args=False,
        allow_nan_stats=True,
    )

    p = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean_p.transpose(),
        covariance_matrix=_cov_p,
        validate_args=False,
        allow_nan_stats=True,
    )

    kl = tfd.kl_divergence(q, p, allow_nan_stats=False)

    return kl


@partial(jit, static_argnums=(0, 1, 2, 3))
def constant_prior_fn(
    n_context_points: int, prior_mean: float, prior_cov: float, output_dim: int, rng_key
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Constant prior function that returns a constant mean and covariance matrix specified by
    the respective hyperparameters. The covariance matrices are 4-dimensional tensors of shape
    (batch_size, output_dim, batch_size, output_dim) where any slice of [:, i, :, i]
    or [j, :, j, :] is an identity matrix to make sure that the covariance matrix is
    well-conditioned when taking the covariance between data points into account.

    Args:
        n_context_points: number of context points.
        prior_mean: mean of the prior.
        prior_cov: covariance of the prior.
        output_dim: dimension of the output.
        rng_key: random number generator key.

    Returns:
        mean and covariance matrix of the prior.

    """

    shape_mean = (n_context_points, output_dim)
    mean_array = jnp.ones(shape_mean) * prior_mean

    # generate a 4-dimension identity covariance tensor, i.e. an array of shape
    # (n_context_points, output_dim, n_context_points, output_dim)

    cov_array = (
        jnp.stack(
            jnp.split(
                jnp.stack(
                    jnp.split(jnp.eye(shape_mean[0] * shape_mean[1]), shape_mean[0], 0)
                ),
                shape_mean[0],
                2,
            )
        ).transpose((0, 2, 1, 3))
        * prior_cov
    )

    return mean_array, cov_array
