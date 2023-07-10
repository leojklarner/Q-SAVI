"""
Script for defining the config for Q-SAVI.
"""

import itertools

NOT_SPECIFIED = "NOT_SPECIFIED"

# define split / featurization combinations and hyperparameter search space

featurizations = [
    "rdkit_bit_fp",
    "ec_bit_fp",
]

splits = [
    "spectral_split",
    "mw_split",
    "scaffold_split",
    "random_split",
]

arg_map = {
    i: (f, s) for i, (f, s) in enumerate(itertools.product(featurizations, splits))
}


# original hyperparameter grid
hyper_grid = {
    "learning_rate": [1e-4, 1e-3],
    "num_layers": [2, 4, 6],
    "embed_dim": [32, 64],
    "prior_cov": [1e-2, 1e-1, 1e0, 1e1, 1e2],
    "n_context_points": [16, 128],
}


hyper_map = {
    i: {
        "learning_rate": learning_rate,
        "num_layers": num_layers,
        "embed_dim": embed_dim,
        "prior_cov": prior_cov,
        "n_context_points": n_context_points,
    }
    for i, (
        learning_rate,
        num_layers,
        embed_dim,
        prior_cov,
        n_context_points,
    ) in enumerate(
        itertools.product(
            hyper_grid["learning_rate"],
            hyper_grid["num_layers"],
            hyper_grid["embed_dim"],
            hyper_grid["prior_cov"],
            hyper_grid["n_context_points"],
        )
    )
}


def add_qsavi_args(parser):
    """
    Add argparser arguments for running Q-SAVI.
    """

    parser.add_argument(
        "--logroot",
        type=str,
        default="experiments",
        help="The root result folder that store runs for this type of experiment",
    )

    parser.add_argument(
        "--subdir",
        type=str,
        default="qsavi_rs",
        help="The subdirectory in logroot/runs/ corresponding to this run",
    )

    parser.add_argument(
        "--datadir",
        type=str,
        default="data/datasets",
        help="The subdirectory in logroot/runs/ corresponding to this run",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="antimalarial_data_processed.pkl",
        help="The name of the dataset to use.",
    )

    parser.add_argument(
        "--seed", type=int, default=10, help="Random seed (default: 10)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training (default: 100)",
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        help="Number of hidden units (default: NOT_SPECIFIED)",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of hidden layers (default: NOT_SPECIFIED)",
    )

    parser.add_argument(
        "--featurization",
        type=str,
        help="Molecular featurization to use (default: not_specified)",
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=2,
        help="Number of classes to predict (default: 2)",
    )

    parser.add_argument(
        "--split",
        type=str,
        help="Molecular split to use (default: not_specified)",
    )

    parser.add_argument(
        "--prior_mean", type=float, default=0.0, help="Prior mean function (default: 0)"
    )

    parser.add_argument(
        "--prior_cov", type=float, default=1.0, help="Prior cov function (default: 0)"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of exp log likelihood samples (default: 10)",
    )

    parser.add_argument(
        "--n_samples_eval",
        type=int,
        default=10,
        help="Number of MC samples for evaluating the BNN prediction in validation or testing, not for training",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name (default: " ")",
        nargs="?",
        const="",
    )

    parser.add_argument(
        "--init_logvar_minval",
        default=-10.0,
        type=float,
        help="logvar initialization range minimum (default: -10.0)",
    )

    parser.add_argument(
        "--init_logvar_maxval",
        default=-8.0,
        type=float,
        help="logvar initialization range maximum (default: -8.0)",
    )

    parser.add_argument(
        "--n_context_points",
        type=int,
        help="Number of BNN context points (default: NOT_SPECIFIED)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs for each task (default: 500)",
    )

    parser.add_argument(
        "--early_stopping_epochs",
        type=int,
        default=10,
        help="Number of epochs for early stopping on val set (default:10)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate (default: 1e-3)",
    )
