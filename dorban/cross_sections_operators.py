"""
This module contains function generating the cross sections related matrices
like the fission matrix and the absorption matrix
"""

from typing import Callable, Sequence

import numpy as np
import scipy.sparse as spar

from dorban.materials import CrossSectionData
from dorban.system import Core
from dorban.utils import sparse_columns, tensor_product


def _nufission(isotope: CrossSectionData) -> np.array:
    return (
        np.outer(isotope.chi / sum(isotope.chi), isotope.nusigmaf)
        if isotope.isfissile
        else np.zeros_like(isotope.scatter)
    )


def _absorb(isotope: CrossSectionData) -> np.array:
    return np.diag(isotope.total) - isotope.scatter


def fission_matrix(system: Core) -> spar.spmatrix:
    """
    This function generates the fission operator as a sparse matrix in csr
    format.

    Parameters
    ----------
    system: Core
        the Core describing the system

    Returns
    -------
    spar.csr_matrix
        the fission matrix in csr format
    """
    E = system.E
    row_indices = tensor_product(np.ones(E), np.arange(system.size))
    col_indices = sparse_columns(system.size, E)
    values = block_values(system.isotopes, _nufission)
    nonzero = np.nonzero(values)[0]
    try:
        return spar.csr_matrix(
            (values[nonzero], (row_indices[nonzero], col_indices[nonzero])), shape=(system.size, system.size)
        )
    except TypeError:  # happens if there is no fissile material in the system
        return spar.csr_matrix((system.size, system.size))


def block_values(
    isotopes: Sequence[CrossSectionData], material_matrix: Callable[[CrossSectionData], np.array]
) -> np.array:
    """
    Function that constructs and returns a numpy array which contains the
    values of the callable material_matrix on each of the CrossSectionsData in
    isotopes

    Parameters
    ----------
    isotopes: Sequence[CrossSectionData]
        Sequence of CrossSectionData
    material_matrix: Callable[[CrossSectionData], np.array]
        Callable which gets a CrossSectionData and returns a 2 dimensional
        numpy array

    Returns
    -------
    np.array
        array which contains the
        values of the callable material_matrix on each of the CrossSectionsData in
        isotopes
    """
    # cells usually share a few materials: compute each material's block once
    first = {}
    index = np.array([first.setdefault(id(isotope), len(first)) for isotope in isotopes], dtype=int)
    unique = {id(isotope): isotope for isotope in isotopes}
    blocks = np.array([material_matrix(isotope).flatten() for isotope in unique.values()])
    return blocks[index].ravel()


def sigma_a(system: Core) -> spar.csr_matrix:
    """
    This function generates the absorption operator as a sparse matrix in the
    csr format.

    Parameters
    ----------
    system: Core
        the core

    Returns
    -------
    spar.csr_matrix
        the absorption matrix in csr format
    """
    E = system.E
    row_indices = tensor_product(np.ones(E), np.arange(system.size))
    col_indices = sparse_columns(system.size, E)
    values = block_values(system.isotopes, _absorb)
    nonzero = np.nonzero(values)[0]
    return spar.coo_matrix(
        (values[nonzero], (row_indices[nonzero], col_indices[nonzero])), shape=(system.size, system.size)
    ).tocsr()
