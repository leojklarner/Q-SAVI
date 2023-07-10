"""
Utilties for computing an approximation of a distribution over the induced function
space of a stochastic neural network by performing a Taylor expansion of the
network around the mean of the variational distribution over its parameters.

For more details and background see:
    - Rudner, Tim GJ, et al. "Tractable function-space variational inference in 
      Bayesian neural networks." Advances in Neural Information Processing 
      Systems 35 (2022): 22686-22698.
    - Rudner, Tim GJ, et al. "Continual learning via sequential function-space 
      variational inference." International Conference on Machine Learning. PMLR, 2022.

"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, NamedTuple, Tuple

import haiku as hk
import jax
import numpy as np
import tree
from jax import eval_shape, jacobian, jit
from jax import numpy as jnp


def induced_function_distribution(
    apply_fn: Callable,
    params_mean: hk.Params,
    params_log_var: hk.Params,
    state: hk.State,
    context_points: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    """
    Compute the mean and covariance of the distribution over the induced function space
    of a neural network evaluated at a batch of context points.

    Args:
        apply_fn: apply function returned by `hk.transform_with_state`.
        params_mean: mean of Gaussian parameter distribution.
        params_log_var: log of variance of Gaussian parameter distribution.
        state: haiku state.
        context_points: a batch of unlabelled context points
        rng_key: JAX random key.

    Returns:
        mean of function distribution, array of shape (batch_dim, output_dim)
        covariance of function distribution, array of shape (batch_dim, output_dim, batch_dim, output_dim)

    """

    # get mean of function distribution
    mean = apply_fn(
        hk.data_structures.merge(params_mean, params_log_var),
        state,
        rng_key,
        context_points,
        rng_key,
        stochastic=False,
    )[0]

    # get covariance of function distribution, given by J*diag(params_var)*J^T
    # where J is the Jacobian of the network with respect to the parameters
    # evaluated at the mean of the variational distribution

    params_var = sigma_transform(params_log_var)
    f_at_mean_predict_fn = lambda mean: apply_fn(
        hk.data_structures.merge(mean, params_log_var),
        state,
        None,
        context_points,
        rng_key,
        stochastic=False,
    )[0]

    cov = get_cov(
        f_at_mean_predict_fn,
        delta_vjp_jvp,
        delta_vjp,
        params_mean,
        params_var,
    )

    return mean, cov


@partial(jit, static_argnums=(0, 1, 2))
def get_cov(
    predict_fn,
    delta_vjp_jvp: Callable,
    delta_vjp: Callable,
    params_mean: hk.Params,
    params_var: hk.Params,
) -> jnp.ndarray:
    """
    Compute the covariance matrix of the function distribution, which is
    equivalent to calculating J*diag(params_var)*J^T, where J is the Jacobian
    of the network with respect to the parameters evaluated at the mean of the
    variational distribution.

    Args:
        predict_fn: mean-returning apply function of a given network
        delta_vjp_jvp: _description_
        delta_vjp: _description_
        params_mean: parameter distribution mean
        params_var: parameter distributon variance

    Returns:
        covariance matrix of function distribution
    """

    predict_struct = eval_shape(predict_fn, params_mean)
    fx_dummy = jnp.ones(predict_struct.shape, predict_struct.dtype)
    delta_vjp_jvp_ = partial(
        delta_vjp_jvp, predict_fn, delta_vjp, params_mean, params_var
    )
    cov_mat = jacobian(delta_vjp_jvp_)(fx_dummy)
    return cov_mat


@partial(jit, static_argnums=(0,))
def delta_vjp(
    predict_fn,
    params_mean: hk.Params,
    params_var: hk.Params,
    delta: jnp.ndarray,
):
    """
    Compute the vector-Jacobian product of the network with respect to the
    parameters evaluated at the mean of the variational distribution and
    a given vector. Rescale the Jacobian by the variance of the variational
    distribution.

    Args:
        predict_fn: mean-returning apply function of a given network
        params_mean: parameter distribution mean
        params_var: parameter distributon variance
        delta: vector to be multiplied with the Jacobian

    Returns:
        vector-Jacobian product
    """

    vjp_tp = jax.vjp(predict_fn, params_mean)[1](delta)
    renamed_params_var = map_variable_name(
        params_var, lambda n: f"{n.split('_')[0]}_mu"
    )
    return (tree.map_structure(lambda x1, x2: x1 * x2, renamed_params_var, vjp_tp[0]),)


@partial(
    jit,
    static_argnums=(
        0,
        1,
    ),
)
def delta_vjp_jvp(
    predict_fn,
    delta_vjp: Callable,
    params_mean: hk.Params,
    params_var: hk.Params,
    delta: jnp.ndarray,
):
    """
    Compute the Jacobian-vector product of the network with respect to the
    parameters evaluated at the mean of the variational distribution.

    Args:
        predict_fn: mean-returning apply function of a given network
        delta_vjp: _description_
        params_mean: parameter distribution mean
        params_var: parameter distributon variance
        delta: vector to be multiplied with the Jacobian

    Returns:
        Jacobian-vector product
    """

    delta_vjp_ = partial(delta_vjp, predict_fn, params_mean, params_var)
    params_mean_renamed = hk.data_structures.to_immutable_dict(params_mean)
    return jax.jvp(predict_fn, (params_mean_renamed,), delta_vjp_(delta))[1]


def map_variable_name(params: hk.Params, fn: Callable) -> hk.Params:
    """
    Change parameters names to enable tree.map_structure in delta_vjp
    to multiply the Jacobian with the correct parameter variance.
    """

    params = hk.data_structures.to_mutable_dict(params)
    for module in params:
        params[module] = {
            fn(var_name): array for var_name, array in params[module].items()
        }
    return hk.data_structures.to_immutable_dict(params)


@jit
def sigma_transform(params_log_var: hk.Params) -> hk.Params:
    """
    Utility function to transform the log of the variance of the parameter
    distribution to the variance of the parameter distribution.

    Args:
        params_log_var: log of variance of parameter distribution

    Returns:
        variance of parameter distribution
    """
    return tree.map_structure(lambda p: jnp.exp(p), params_log_var)
