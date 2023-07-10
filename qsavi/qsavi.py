import copy
import json
import os
from functools import partial
from typing import Any, Dict, Iterable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from jax import jit
from jax.lib import xla_bridge
from sklearn.metrics import log_loss

from .bayesian_mlps import BayesianMLP
from .context_points import get_context_points
from .data_loader import process_data
from .objective import constant_prior_fn, qsavi_loss
from .utils import KeyHelper, initialize_random_keys

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

tf.config.experimental.set_visible_devices([], "GPU")
print("WARNING: TensorFlow is set to only use CPU.")
print("Num GPUs Available (TF): ", len(tf.config.list_physical_devices("GPU")))
print("JAX is using", xla_bridge.get_backend().platform)
print("JAX devices:", jax.devices())


class QSAVI:
    def __init__(
        self,
        hparams: dict,
    ) -> None:
        """
        Use the provided hyperparameters to initialize Q-SAVI model.
        """

        self.hparams = hparams

        # initialize random keys
        self.kh = initialize_random_keys(seed=self.hparams.seed)

        # load and preprocess data
        self.data_iterators, self.data_metadata = process_data(
            data_dir=self.hparams.datadir,
            featurization=self.hparams.featurization,
            split=self.hparams.split,
            dataset_name=self.hparams.dataset_name,
            n_classes=self.hparams.n_classes,
            batch_size=self.hparams.batch_size,
        )

        # initialize network architecture and params
        self.model = BayesianMLP(
            output_dim=self.data_metadata["output_dim"],
            num_layers=self.hparams.num_layers,
            embed_dim=self.hparams.embed_dim,
            init_logvar_minval=self.hparams.init_logvar_minval,
            init_logvar_maxval=self.hparams.init_logvar_maxval,
        )

        self.init_fn, self.apply_fn = self.model.forward
        x_init = jnp.ones(self.data_metadata["input_shape"], dtype=jnp.float32)
        init_key = jax.random.split(self.kh.next_key(), 1)[0]
        self.params, self.state = self.init_fn(init_key, x_init, init_key)

        # initialize function space prior
        prior_fn_key = self.kh.next_key()
        self.prior_mean, self.prior_cov = constant_prior_fn(
            self.hparams.n_context_points,
            self.hparams.prior_mean,
            self.hparams.prior_cov,
            self.data_metadata["output_dim"],
            prior_fn_key,
        )

        # initialize optimizer and loss function
        self.opt = optax.adam(self.hparams.learning_rate)
        self.opt_state = self.opt.init(self.params)
        self.loss = qsavi_loss

        # initialize context point sampler
        self.context_point_fn = get_context_points

    def train(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform full Q-SAVI training loop. For each epoch:
            - iterate over batches of training data,
            - sample context points from context point distribution,
            - evaluate function-space loss wrt context points and prior,
            - compute gradients and apply gradient descent update,
            - evaluate model on validation set and save best model parameters.
        """

        # get number of batches and check that it matches expected value
        num_batches = len(self.data_iterators.batch_train)
        assert (
            num_batches == self.data_metadata["n_train"] // self.hparams.batch_size
        ), "Issue with batch numbers."

        # initialize variables needed for early stopping
        best_nll = 1e3
        epochs_without_improvement = 0
        best_params = copy.deepcopy(self.params)
        best_state = copy.deepcopy(self.state)

        # pre-split random keys to speed up forward passes
        split_key = self.kh.next_key()
        rng_keys = jax.random.split(split_key, num_batches * 2 * self.hparams.epochs)
        context_rng_keys = rng_keys[: (num_batches * self.hparams.epochs)]
        update_rng_keys = rng_keys[(num_batches * self.hparams.epochs) :]

        # pred_key = self.kh.next_key()
        # val_nll = self.evaluate(self.data_iterators.full_valid, pred_key)["nll"]
        # test_nll = self.evaluate(self.data_iterators.full_test, pred_key)["nll"]
        # print(f"Initial validation NLL: {val_nll:.5f}", f"Initial test NLL: {test_nll:.5f}")

        for epoch in range(self.hparams.epochs):
            for batch_i, batch_data in enumerate(self.data_iterators.batch_train):

                x_batch, y_batch = batch_data

                # sample context points and update model parameters
                context_points = self.context_point_fn(
                    context_rng_keys[epoch * num_batches + batch_i],
                    self.hparams.n_context_points,
                    self.hparams.featurization,
                    self.hparams.datadir,
                )

                self.params, self.state, self.opt_state = update(
                    self.loss,
                    self.params,
                    self.state,
                    self.opt_state,
                    self.prior_mean,
                    self.prior_cov,
                    x_batch,
                    y_batch,
                    context_points,
                    update_rng_keys[epoch * num_batches + batch_i],
                    self.opt,
                    self.hparams.n_samples,
                    self.model,
                )

            # get negative log likelihood on validation set for early stopping
            pred_key = self.kh.next_key()
            val_nll = self.evaluate(self.data_iterators.full_valid, pred_key)["nll"]
            print("Epoch", epoch, f"{val_nll:.5f}", f"{best_nll:.5f}")

            if val_nll <= best_nll:
                # if validation set NLL improves, continue training
                # with new best model parameters and state
                best_nll = val_nll
                epochs_without_improvement = 0
                best_params = copy.deepcopy(self.params)
                best_state = copy.deepcopy(self.state)
            else:
                # if validation set NLL deteriorates, increment
                # early stopping counter and eventually stop training
                epochs_without_improvement += 1
                if epochs_without_improvement == self.hparams.early_stopping_epochs:
                    break

        self.params = best_params
        self.state = best_state

        # get validation and test set metrics and predictions
        pred_key = self.kh.next_key()
        val_metrics = self.evaluate(self.data_iterators.full_valid, pred_key)
        test_metrics = self.evaluate(self.data_iterators.full_test, pred_key)

        print(
            f"\n----- Stopped training after {epoch - epochs_without_improvement} / {self.hparams.epochs} epochs",
            f"with NLL {val_metrics['nll']:.5f} (best = {best_nll:.5f}) and test nll {test_metrics['nll']:.5f} -----\n",
        )

        return val_metrics, test_metrics

    def evaluate(
        self, iterator: Iterable, pred_key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation or test set.
        """

        assert len(iterator) == 1, "Unexpectedly long validation iterator"
        x_eval, y_eval = np.array(iterator[0][0]), np.array(iterator[0][1])
        assert y_eval.ndim == 1, f"Got targets of dimensions {y_eval.ndim}, expected 1."

        _, preds, _ = self.model.predict_f_multisample(
            params=self.params,
            state=self.state,
            inputs=x_eval,
            rng_key=pred_key,
            n_samples=self.hparams.n_samples_eval,
        )
        preds = jax.nn.softmax(preds, axis=-1)

        return {
            "nll": log_loss(y_true=y_eval, y_pred=preds[:, 1]),
            "preds": np.array(preds[:, 1]).tolist(),
            "labels": np.array(y_eval, dtype=bool).tolist(),
        }


@partial(jit, static_argnums=(0, 10, 11, 12))
def update(
    loss: qsavi_loss,
    params: hk.Params,
    state: hk.State,
    opt_state: optax.OptState,
    prior_mean: jnp.ndarray,
    prior_cov: jnp.ndarray,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    context_points: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    opt: optax.GradientTransformation,
    n_samples: int,
    model: BayesianMLP,
) -> Tuple[hk.Params, hk.State, optax.OptState]:
    """
    Get gradients and update model parameters using a single batch of data.
    """

    grads = jax.grad(loss, argnums=0)(
        params,
        state,
        prior_mean,
        prior_cov,
        x_batch,
        y_batch,
        context_points,
        rng_key,
        n_samples,
        model,
    )

    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    if len(state):
        state = model.apply_fn(params, state, rng_key, x_batch, rng_key)[1]

    return params, state, opt_state
