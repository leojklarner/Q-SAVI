"""
Data loading and preprocessing utilities.
"""

import os
from typing import Any, Callable, Dict, Iterable, NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf


def _one_hot(y: jnp.ndarray, k: int, dtype=jnp.float32) -> jnp.ndarray:
    """One-hot encoding of integer array y."""
    return jnp.array(y[:, None] == jnp.arange(k), dtype)


class Iterators(NamedTuple):
    """
    A utility class defining the typing for the iterators returned by the data loader.

    Attributes:
        batch_train (Iterable): Iterator over the training data in batches.
        full_valid (Iterable): Iterator over the full validation data.
        full_test (Iterable): Iterator over the full test data.

    """

    batch_train: Iterable
    full_valid: Iterable
    full_test: Iterable


def process_data(
    data_dir: str,
    featurization: str,
    split: str,
    dataset_name: str,
    n_classes: int = 2,
    batch_size: int = 128,
) -> Tuple[Iterators, Dict[str, Any]]:

    """
    Data processing function that returns the dataset corresponding to
    a given featurization, split, and dataset combination.

    Args:
        data_dir (str): Path to the data directory.
        featurization (str): Name of the featurization to use (ec_bit_fp, rdkit_bit_fp).
        split (str): Name of the split (spectral, mw, scaffold, random).
        dataset_name (str): Name of the dataset file.
        n_classes (int): Number of classes to predict.
        batch_size (int): Batch size to use for splitting the training set.

    Returns:
        A tuple containing the iterators over the training, validation, and test sets
        and a dictionary containing dataset configuration details.
    """

    if split == "spectral_split":
        split = "_".join([featurization, split])

    dataset_config = {
        "output_dim": n_classes,
        "featurization": featurization,
        "split": split,
        "dataset_name": dataset_name,
    }

    # read in pre-processed dataset from specified data directory
    # and split it into train, val, and test sets based on the specified
    # split type given in the precomputed split_type_cv column

    data_df = pd.read_pickle(os.path.join(data_dir, dataset_name))

    assert (
        f"{split}_cv" in data_df.columns
    ), f"Specified split {split} type not found in DataFrame."
    assert (
        featurization in data_df.columns
    ), f"Specified featurization {featurization} type not found in DataFrame."
    assert "label" in data_df.columns, "Label column not found in DataFrame."

    train_idx = data_df[data_df[f"{split}_cv"] < 3].index
    val_idx = data_df[data_df[f"{split}_cv"] == 3].index
    test_idx = data_df[data_df[f"{split}_cv"].isna()].index

    # ase bit vector fingerprints are stored as sparse matrices, use
    # scipy.sparse.vstack before converting them to a single dense tensor

    data = {
        "train": (
            sp.vstack(data_df.loc[train_idx, featurization].to_list()).A.astype(float),
            data_df.loc[train_idx, "label"].to_numpy(dtype="int"),
        ),
        "valid": (
            sp.vstack(data_df.loc[val_idx, featurization].to_list()).A.astype(float),
            data_df.loc[val_idx, "label"].to_numpy(dtype="int"),
        ),
        "test": (
            sp.vstack(data_df.loc[test_idx, featurization].to_list()).A.astype(float),
            data_df.loc[test_idx, "label"].to_numpy(dtype="int"),
        ),
    }

    dataset_config["n_train"] = len(train_idx)
    dataset_config["n_valid"] = len(val_idx)
    dataset_config["n_test"] = len(test_idx)
    dataset_config["input_shape"] = [1] + list(data["train"][0].shape[1:])

    # use tensorflow utils to batch training set and convert to jax arrays

    data = Iterators(
        [
            (jnp.array(x), _one_hot(y, n_classes))
            for x, y in tf.data.Dataset.from_tensor_slices(data["train"]).batch(
                batch_size, drop_remainder=True
            )
        ],
        [
            (jnp.array(x), jnp.array(y))
            for x, y in tf.data.Dataset.from_tensor_slices(data["valid"]).batch(
                dataset_config["n_valid"]
            )
        ],
        [
            (jnp.array(x), jnp.array(y))
            for x, y in tf.data.Dataset.from_tensor_slices(data["test"]).batch(
                dataset_config["n_test"]
            )
        ],
    )

    print(
        "\n\nUsing dataset configuration: \n",
        f"Featurization: {dataset_config['featurization']} \n",
        f"Split: {dataset_config['split']} \n",
        f"Dataset: {dataset_config['dataset_name']} \n",
        f"Number of training samples: {dataset_config['n_train']} \n",
        f"Number of training batches: {len(data.batch_train)} \n",
        f"Number of validation samples: {dataset_config['n_valid']} \n",
        f"Number of test samples: {dataset_config['n_test']} \n",
        f"Input shape: {dataset_config['input_shape']} \n",
        f"Output dimension: {dataset_config['output_dim']} \n\n\n",
    )

    return data, dataset_config
