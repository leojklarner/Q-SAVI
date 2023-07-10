"""
Utilities for preprocessing a uniform subsample of the ZINC database.
"""


import functools
import os
import pickle
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit.Chem import AllChem, MolFromSmiles, RDKFingerprint
from tqdm.contrib.concurrent import process_map


def process_ecfps(mols_per_file: int, smiles: str, output_dir: str, i: int) -> None:
    """
    Convert SMILES strings to extended connectivity fingerprints (ECFPs)
    and save to disk as scipy sparse matrices.

    Args:
        mols_per_file: number of molecules to process per file
        smiles: pandas Series of SMILES strings
        output_dir: directory to save output
        i: index of current batch

    Returns:
        None
    """

    fps = smiles[mols_per_file * i : mols_per_file * (i + 1)].apply(
        lambda x: np.array(
            AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), 2, nBits=2048)
        )
    )
    fps = np.vstack(fps.to_list())
    sp.save_npz(
        os.path.join(output_dir, f"zinc_ecfp_{i}.npz"),
        sp.csr_matrix(fps, dtype=bool),
    )


def process_rdkitfps(mols_per_file: int, smiles: str, output_dir: str, i: int, ) -> None:
    """
    Convert SMILES strings to rdkit fingerprints (ECFPs)
    and save to disk as scipy sparse matrices.

    Args:
        mols_per_file: number of molecules to process per file
        smiles: pandas Series of SMILES strings
        output_dir: directory to save output
        i: index of current batch

    Returns:
        None
    """

    fps = smiles[mols_per_file * i : mols_per_file * (i + 1)].apply(
        lambda x: np.array(RDKFingerprint(MolFromSmiles(x), maxPath=4))
    )
    fps = np.vstack(fps.to_list())
    sp.save_npz(
        os.path.join(output_dir, f"zinc_rdkit_{i}.npz"),
        sp.csr_matrix(fps, dtype=bool),
    )


if __name__ == "__main__":

    print(f"Using {cpu_count()} CPUs.")

    # read in uniform subsample of ZINC database
    zinc_smiles = pd.read_csv("datasets/zinc_smiles.csv")["0"]
    print(f"Loaded {len(zinc_smiles)} SMILES strings.")

    # make sure output directories exist
    ecfp_dir = os.path.join("datasets", "zinc", "ecfp")
    rdkit_dir = os.path.join("datasets", "zinc", "rdkit")
    os.makedirs(ecfp_dir, exist_ok=True)
    os.makedirs(rdkit_dir, exist_ok=True)

    # convert SMILES to bit vector fingerprints
    mols_per_file = 512
    n_iters = len(zinc_smiles) // mols_per_file
    chunksize = 50

    process_map(
        functools.partial(process_ecfps, mols_per_file, zinc_smiles, ecfp_dir),
        range(n_iters),
        total=n_iters,
        chunksize=chunksize,
    )

    process_map(
        functools.partial(process_rdkitfps, mols_per_file, zinc_smiles, rdkit_dir),
        range(n_iters),
        total=n_iters,
        chunksize=chunksize,
    )
