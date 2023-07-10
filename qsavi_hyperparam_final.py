"""
Find the hyperparameter combination with the lowest validation set
loss and use it to retrain ten models with different random seeds.
"""


import os
import argparse
import json
import pickle
import pandas as pd



from qsavi.qsavi import QSAVI
from qsavi.config import add_qsavi_args, arg_map, hyper_map


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Q-SAVI Command Line Interface')

    parser.add_argument('--arg_map_id', type=int, help='Index determining which split/featurization to use.')
    add_qsavi_args(parser)
    kwargs = parser.parse_args()

    featurization, split = arg_map[kwargs.arg_map_id]
    print("\n\nUsing featurization:", featurization, "and split:", split, "\n\n")

    hyper_results = {}

    print("\n\nLoading hyperparameter search logs:")

    # load all hyperparameter search logs
    for f in os.listdir(os.path.join(kwargs.logroot, kwargs.subdir)):
        if split in f and featurization in f:
            print("\t", f)
            with open(os.path.join(kwargs.logroot, kwargs.subdir, f), "rb") as v:
                hyper_results[int(f.split("_")[-1].removesuffix(".pkl"))] = pickle.load(v)["val_metrics"]["nll"]

    print(f"\n\nFound best hyperparameters out of {len(hyper_results)} combinations:")
    hyper_results = pd.Series(hyper_results)
    hypers = hyper_map[hyper_results.idxmin()]
    for k, v in hypers.items():
        print("\t-", k, ":", v)

    kwargs.split = split
    kwargs.featurization = featurization
    kwargs.learning_rate = hypers["learning_rate"]
    kwargs.num_layers = hypers["num_layers"]
    kwargs.embed_dim = hypers["embed_dim"]
    kwargs.prior_cov = hypers["prior_cov"]
    kwargs.n_context_points = hypers["n_context_points"]

    for i in range(10):

        kwargs.seed = i
        
        print(
            "\n\nFull input arguments:",
            json.dumps(vars(kwargs), indent=4, separators=(",", ":")),
            "\n\n",
        )

        qsavi = QSAVI(kwargs)
        val_metrics, test_metrics = qsavi.train()

        with open(os.path.join(kwargs.logroot, kwargs.subdir, f"{kwargs.split}_{kwargs.featurization}_final_{i}.pkl"), "wb") as f:
            pickle.dump({"val_metrics": val_metrics, "test_metrics": test_metrics, **vars(kwargs)}, f)
