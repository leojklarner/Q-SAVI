"""
Contains miscellaneous utility functions to facilitate the training and testing of Q-SAVI.
"""

import os
import random as random_py

import jax
import numpy as np
import tensorflow as tf


class KeyHelper:
    def __init__(self, key: jax.random.PRNGKey) -> None:
        self._key = key

    def next_key(self) -> jax.random.PRNGKey:
        self._key, sub_key = jax.random.split(self._key)
        return sub_key


def initialize_random_keys(seed: int) -> KeyHelper:
    """Fix random seeds across packages."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    rng_key = jax.random.PRNGKey(seed)
    kh = KeyHelper(key=rng_key)
    random_py.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return kh
