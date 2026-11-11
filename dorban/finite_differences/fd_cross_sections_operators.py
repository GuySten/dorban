"""
This module will generate the FD method cross-sections operators,
the absorption and the fission operators
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags

from dorban.cross_sections_operators import fission_matrix, sigma_a
from dorban.geometry.boundary_conditions import CurrentCondition
from dorban.geometry.geometry import FiniteGeometry
from dorban.system import Core


def volumes(system: Core, inverse: bool = False) -> csr_matrix:
    """
    Computes the volumes of the mesh cells and returns a diagonal sparse
    matrix of size system.size x system.size in the csr format whose
    elements on the diagonal are the volumes of the cells.

    Parameters
    ----------
    system: Core
    inverse:bool
     if True, returns the matrix of volume inverses on the diagonal. False by default.

    Returns
    -------
    csr_matrix
     diagonal sparse matrix of the volumes in the csr format
    """
    volumes = np.array([system.geometry.volumes[cell // system.E] for cell in range(system.size)])
    return spdiags(1 / volumes if inverse else volumes, 0, system.size, system.size, format="csr")


def cmfd_fission(system: Core) -> csr_matrix:
    """
    generates the fission operator of the FD version of the diffusion equation

    Parameters
    ----------
    system: Core

    Returns
    -------
    csr_matrix
     the fission operator as a sparse matrix in the csr format
    """
    return volumes(system) @ fission_matrix(system)


def cmfd_absorption(system: Core) -> csr_matrix:
    """
    generates the absorption operator of the FD version of the diffusion
    equation

    Parameters
    ----------
    system: Core

    Returns
    -------
    csr_matrix
     the fission operator as a sparse matrix in the csr format
    """
    return volumes(system) @ sigma_a(system)


def boundary_current(geometry: FiniteGeometry, E: int, current_calc) -> np.array:
    """
    Computes the current vector on the boundary of the geometry, used for finding
    the flux that yields a known current at the boundary

    Parameters
    ----------
    geometry: FiniteGeometry
     The geometry that describes the system
    E:int
     The number of energy groups
    current_calc: CurrentCalculator
     The current calculator of the system

    Returns
    -------
    np.array
     the current vector, vector of length cellsxE whose value at a given index
     is 0 if the corresponding cell is not on the boundary and the surface
     averaged current if the cell is on the boundary
    """
    current = np.zeros(E * geometry.cells)
    for cell in range(geometry.cells):
        for face, neighbor in enumerate(geometry.neighbors[cell]):
            if isinstance(neighbor, CurrentCondition):
                index, value = neighbor.boundary_current(geometry, E, current_calc, cell, face)
                current[index] += value
    return current
