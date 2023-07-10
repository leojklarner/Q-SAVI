
import os
import argparse
import json
import pickle

from qsavi.qsavi import QSAVI
from qsavi.config import add_qsavi_args, arg_map, hyper_map


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Q-SAVI Command Line Interface')

    parser.add_argument('--arg_map_id', type=int, help='Index determining which split/featurization to use.')
    parser.add_argument('--hyper_map_id', type=int, help='Index determining which hyperparameter combination to use.')
    add_qsavi_args(parser)
    kwargs = parser.parse_args()

    featurization, split = arg_map[kwargs.arg_map_id]
    hypers = hyper_map[kwargs.hyper_map_id]

    print("\n\nusing featurization:", featurization, "and split:", split, "\n\n")

    print("\n\nUsing hyperparameters:")
    for k, v in hypers.items():
        print("\t-", k, ":", v)

    kwargs.split = split
    kwargs.featurization = featurization
    kwargs.learning_rate = hypers["learning_rate"]
    kwargs.num_layers = hypers["num_layers"]
    kwargs.embed_dim = hypers["embed_dim"]
    kwargs.prior_cov = hypers["prior_cov"]
    kwargs.n_context_points = hypers["n_context_points"]
        
    print(
        "\n\nFull input arguments:",
        json.dumps(vars(kwargs), indent=4, separators=(",", ":")),
        "\n\n",
    )

    # make sure logroot and subdir directories exist
    os.makedirs(kwargs.logroot, exist_ok=True)
    os.makedirs(os.path.join(kwargs.logroot, kwargs.subdir), exist_ok=True)

    qsavi = QSAVI(kwargs)
    val_metrics, _ = qsavi.train()

    # save 
    with open(os.path.join(kwargs.logroot, kwargs.subdir, f"{kwargs.split}_{kwargs.featurization}_hyper_{kwargs.hyper_map_id}.pkl"), "wb") as f:
        pickle.dump({"val_metrics": val_metrics, **vars(kwargs)}, f)
