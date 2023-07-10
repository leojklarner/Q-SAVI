"""
Stochastic versions of standard multi-layer perceptrons (MLPs) for use in
the Q-SAVI algorithm. Implemented by subclassing the standard Haiku module.
"""

from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import jit

# ------------------ STOCHASTIC LINEAR LAYER ------------------ #


class DenseStochastic(hk.Module):
    """
    Stochastic fully connected layer with parameters sampled from a Gaussian
    variational distribution parameterised by means and log-varainces.
    """

    def __init__(
        self,
        output_size: int,
        init_logvar_minval: float,
        init_logvar_maxval: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Initialise the stochastic fully connected layer.

        Args:
            output_size: output dimension.
            init_logvar_minval: lower bound of the range from which to
                log-uniformly sample the log-variance initialization.
            init_logvar_maxval: upper bound of the range from which to
                log-uniformly sample the log-variance initialization.
            with_bias: whether to include a bias term.
            w_init: initializer for the weight matrix.
            b_init: initializer for the bias vector.
            name: name of the module.

        Returns:
            None.

        """

        super(DenseStochastic, self).__init__(name=name)

        self.output_size = output_size
        self.w_init = w_init
        self.b_init = b_init
        self.init_logvar_minval = init_logvar_minval
        self.init_logvar_maxval = init_logvar_maxval

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
        stochastic=True,
    ) -> jnp.ndarray:
        """
        Forward pass of the stochastic fully connected layer.

        Args:
            inputs: input data.
            rng_key: random number generator key.
            stochastic: whether to sample from the variational distribution
                over parameters or only use their means.

        Returns:
            outputs: output data.

        """

        input_size = inputs.shape[-1]
        input_dtype = inputs.dtype

        # if no initialiser is specified, use the default Haiku initialiser

        if self.w_init is None:
            stddev = 1 / np.sqrt(input_size)
            self.w_init = hk.initializers.TruncatedNormal(stddev=stddev, mean=0)

        if self.b_init is None:
            self.b_init = hk.initializers.Constant(0)

        # get parameter means and log-variances
        w_mu = hk.get_parameter(
            "w_mu",
            shape=[input_size, self.output_size],
            dtype=input_dtype,
            init=self.w_init,
        )

        b_mu = hk.get_parameter(
            "b_mu", shape=[self.output_size], dtype=input_dtype, init=self.b_init
        )

        w_logvar = hk.get_parameter(
            "w_logvar",
            shape=[input_size, self.output_size],
            dtype=input_dtype,
            init=uniform_mod(self.init_logvar_minval, self.init_logvar_maxval),
        )

        b_logvar = hk.get_parameter(
            "b_logvar",
            shape=[self.output_size],
            dtype=input_dtype,
            init=uniform_mod(
                self.init_logvar_minval,
                self.init_logvar_maxval,
            ),
        )

        key_1, key_2 = jax.random.split(rng_key)
        W = gaussian_sample(w_mu, w_logvar, stochastic, key_1)
        b = gaussian_sample(b_mu, b_logvar, stochastic, key_2)

        return jnp.dot(inputs, W) + b


# ------------------ DEFINE STOCHASTIC MLP ------------------ #


class MLP(hk.Module):
    """
    MLP class to construct a sequence of stochastic fully connected layers
    with ReLU activations and a linear output layer.
    """

    def __init__(
        self,
        output_dim: int,
        num_layers: int,
        embed_dim: int,
        init_logvar_minval: float = -10.0,
        init_logvar_maxval: float = -8.0,
    ):
        """
        Initialise the MLP.

        Args:
            output_dim: output dimension.
            num_layers: number of layers.
            embed_dim: embedding dimension.
            init_logvar_minval: lower bound of the range from which to
                log-uniformly sample the log-variance initialization.
            init_logvar_maxval: upper bound of the range from which to
                log-uniformly sample the log-variance initialization.

        Returns:
            None.

        """

        super().__init__()

        self.activation_fn = jax.nn.relu

        self.layers = []
        for _ in range(num_layers - 1):
            self.layers.append(
                DenseStochastic(
                    output_size=embed_dim,
                    init_logvar_minval=init_logvar_minval,
                    init_logvar_maxval=init_logvar_maxval,
                )
            )

        # FINAL LAYER
        self.layers.append(
            DenseStochastic(
                output_size=output_dim,
                init_logvar_minval=init_logvar_minval,
                init_logvar_maxval=init_logvar_maxval,
                name="linear_final",
            )
        )

    def __call__(
        self, inputs: jnp.ndarray, rng_key: jnp.ndarray, stochastic: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        Args:
            inputs: input data.
            rng_key: random number generator key.
            stochastic: whether to sample from the variational distribution
                over parameters or only use their means.

        Returns:
            outputs: output data.

        """

        rng_keys = random.split(rng_key, len(self.layers))

        out = inputs

        for l in range(len(self.layers) - 1):
            out = self.layers[l](out, rng_keys[l], stochastic)
            out = self.activation_fn(out)

        out = self.layers[-1](out, rng_keys[-1], stochastic)

        return out


class BayesianMLP:
    """
    Wrapper class that uses Haiku to transform the MLP defined above into
    a set of pure functions, which it exposes as a range of different methods.
    """

    def __init__(
        self,
        output_dim: int,
        num_layers: int,
        embed_dim: int,
        init_logvar_minval: float = -10.0,
        init_logvar_maxval: float = -8.0,
    ):
        """
        Initialise the Bayesian MLP.

        Args:
            output_dim: output dimension.
            num_layers: number of layers.
            embed_dim: embedding dimension.
            init_logvar_minval: lower bound of the range from which to
                log-uniformly sample the log-variance initialization.
            init_logvar_maxval: upper bound of the range from which to
                log-uniformly sample the log-variance initialization.

        Returns:
            None.

        """

        self.output_dim = output_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_logvar_minval = init_logvar_minval
        self.init_logvar_maxval = init_logvar_maxval

        forward_fn = lambda inputs, rng_key, stochastic=True: MLP(
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            init_logvar_minval=self.init_logvar_minval,
            init_logvar_maxval=self.init_logvar_maxval,
        )(inputs, rng_key, stochastic)

        self.forward = hk.transform_with_state(forward_fn)

    @property
    def apply_fn(self) -> Callable:
        return self.forward.apply

    @partial(jit, static_argnums=(0))
    def predict_f(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Draw a sample from the parameter distribution of the MLP and use it
        to predict the labels at a batch of input points.

        Args:
            params: model parameters.
            state: model state.
            inputs: a batch of input.
            rng_key: JAX random key.

        Returns:
            outputs: output data.

        """

        return self.forward.apply(
            params,
            state,
            rng_key,
            inputs,
            rng_key,
        )[0]

    def predict_f_multisample(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: jnp.ndarray,
        rng_key: jnp.ndarray,
        n_samples: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Draw multiple Monte-Carlo samples from the parameter distribution of
        the MLP and use them to predict the labels at a batch of input points.

        Args:
            params: model parameters.
            state: model state.
            inputs: a batch of input.
            rng_key: JAX random key.
            n_samples: number of MC samples.

        Returns:
            n_samples samples of logits, of shape
            (n_samples, inputs.shape[0], output_dimension).
            mean of logits, of shape (inputs.shape[0], output_dimension).
            variance of logits, of shape (inputs.shape[0], output_dimension).

        """

        pred_fn = lambda rng_key: self.predict_f(params, state, inputs, rng_key)

        return mc_sampling(
            fn=pred_fn,
            n_samples=n_samples,
            rng_key=rng_key,
        )

    @partial(jit, static_argnums=(0, 5))
    def predict_f_multisample_jitted(
        self,
        params,
        state,
        inputs,
        rng_key,
        n_samples: int,
    ):
        """
        Jitted version of predict_f_multisample for use in the training loop.

        Args:
            params: model parameters.
            state: model state.
            inputs: a batch of input.
            rng_key: JAX random key.
            n_samples: number of MC samples.

        Returns:
            n_samples samples of logits, of shape
            (n_samples, inputs.shape[0], output_dimension).
            mean of logits, of shape (inputs.shape[0], output_dimension).
            variance of logits, of shape (inputs.shape[0], output_dimension).

        """

        rng_keys = jax.random.split(rng_key, n_samples)
        _predict_multisample_fn = lambda rng_key: self.predict_f(
            params,
            state,
            inputs,
            rng_key,
        )
        predict_multisample_fn = jax.vmap(
            _predict_multisample_fn, in_axes=0, out_axes=0
        )

        preds_samples = predict_multisample_fn(rng_keys)
        preds_mean = preds_samples.mean(axis=0)
        preds_var = preds_samples.std(axis=0) ** 2

        return preds_samples, preds_mean, preds_var


# ------------------ UTILITY FUNCTIONS ------------------ #


def uniform_mod(min_val: float, max_val: float) -> Callable:
    """
    Returns a weight initializer for uniform sampling from the given range.
    Used to log-uniformly sample the log-variance of the weights.

    Args:
        min_val: lower bound of the range.
        max_val: upper bound of the range.

    Returns:
        weight initializer.

    """

    def _uniform_mod(shape, dtype):
        rng_key, _ = random.split(random.PRNGKey(0))
        return jax.random.uniform(
            rng_key, shape=shape, dtype=dtype, minval=min_val, maxval=max_val
        )

    return _uniform_mod


def mc_sampling(
    fn: Callable, n_samples: int, rng_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    """
    Performs Monte Carlo sampling of the variational distribution of the parameters
    and returns the samples themselves, as well as their mean and variance.

    Args:
        fn: a deterministic function that accepts in a random key
        n_samples: number of MC samples.
        rng_key: random key.

    Returns:
        n_samples samples of logits, of shape
        (n_samples, inputs.shape[0], output_dimension).
        mean of logits, of shape (inputs.shape[0], output_dimension).
        variance of logits, of shape (inputs.shape[0], output_dimension).

    """

    list_of_pred_samples = []
    for _ in range(n_samples):
        rng_key, subkey = jax.random.split(rng_key)
        output = fn(subkey)
        list_of_pred_samples.append(jnp.expand_dims(output, 0))
    preds_samples = jnp.concatenate(list_of_pred_samples, 0)
    preds_mean = preds_samples.mean(axis=0)
    preds_var = preds_samples.std(axis=0) ** 2
    return preds_samples, preds_mean, preds_var


def gaussian_sample(
    mu: jnp.ndarray, rho: jnp.ndarray, stochastic: bool, rng_key: jnp.ndarray
) -> jnp.ndarray:

    """
    Samples from a Gaussian distribution with mean mu and log-variance rho.

    Args:
        mu: mean of the distribution.
        rho: log-variance of the distribution.
        stochastic: whether to sample stochastically or deterministically.
        rng_key: random key.

    Returns:
        a sample from the Gaussian distribution.

    """

    if stochastic:
        jnp_eps = random.normal(rng_key, mu.shape)
        z = mu + jnp.exp((0.5 * rho).astype(jnp.float32)) * jnp_eps
    else:
        z = mu
    return z
