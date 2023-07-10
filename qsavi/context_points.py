"""
Utility functions for sampling context points from a large database
of unlabelled molecular structures to use as context points for Q-SAVI.
"""

import os

import jax
import numpy as np
import scipy.sparse as sp
from jax import numpy as jnp

featurization_map = {
    "rdkit_bit_fp": "rdkit",
    "ec_bit_fp": "ecfp",
}


def get_context_points(
    rng_key: jax.random.PRNGKey,
    n_context_points: int,
    featurization: str,
    data_dir: str,
) -> jnp.ndarray:
    """
    Randomly sample context points from a preprocessed database of unlabelled molecular structures.

    Args:
        rng_key: JAX random key.
        n_context_points: Number of context points to sample.
        featurization: Name of featurization to use.
        data_dir: Directory containing the preprocessed data.

    Returns:
        context_points: jnp.array of context points.
    """

    # randomly choose pre-processed context point file from specified directory

    context_point_dir = os.path.join(data_dir, "zinc")
    context_point_files = [
        f
        for f in os.listdir(context_point_dir)
        if featurization_map[featurization] in f
    ]

    assert (
        len(context_point_files) > 0
    ), f"No preprocessed context points found in {context_point_dir} for featurization {featurization}."

    context_point_file = context_point_files[
        jax.random.randint(
            key=rng_key, shape=[1], minval=0, maxval=len(context_point_files)
        )[0]
    ]

    # load context point file and drop columns with zero variance in the training data

    context_points = sp.load_npz(os.path.join(context_point_dir, context_point_file))
    assert n_context_points <= context_points.shape[0], "Not enough context points."

    context_point_indices = jax.random.choice(
        key=rng_key,
        a=context_points.shape[0],
        shape=[n_context_points],
        replace=False,
    )

    info_cols = np.load(os.path.join(data_dir, "info_cols.npz"))[featurization]

    context_points = jnp.array(
        context_points.toarray()[np.ix_(context_point_indices, info_cols)]
    ).astype(jnp.float32)

    return context_points
