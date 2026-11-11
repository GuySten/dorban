"""
This module contains a function to build the diffusion matrix, this function is different from the function in the
finite differences module as it is compatible with numba and is faster.


"""

import numba
import numpy as np
import scipy.sparse as spar

from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator, _compute
from dorban.utils import op_face


def diffusion_matrix(
    neighbors: np.array,
    distances: np.array,
    surface_areas: np.array,
    E: int,
    current_calc: CMFDCurrentCalculator,
    albedos: np.array = None,
) -> spar.coo_matrix:
    r"""
    This function will generate a sparse matrix representing the diffusion
    operator.

    Parameters
    ----------
    neighbors: np.array
     Array of shape cells x faces which contains the neighbors of each cell,
     negative values represent boundary conditions.
    distances: np.array
     Array of shape cells x faces which contains the distances between the center of each cell to its faces.
    surface_areas: np.array
     Array of shape cells x faces which contains the surface area of each face.
    E: int
     The number of energy groups.
    current_calc: CMFDCurrentCalculator
     The current calculator.
    albedos: np.array
     The albedo of each face, see :func:`dorban.nem.cmfd_iterator.boundary_albedos`. If None, 0.5 (Void) is used on
     faces with the number -1 and infinity (ZeroFlux) on faces with the number -3.

    Returns
    -------
    spar.csr_matrix
     sparse matrix in csr format representing the linear operator flux goes to
     integral of :math:`D\nabla \phi` , :math:`\phi` being the flux.
    """

    C = neighbors.shape[0]
    if albedos is None:
        albedos = np.where(neighbors == -1, 0.5, np.where(neighbors == -3, np.inf, 0.0))
    row_indices, col_indices, values = _diffusion_matrix(
        neighbors, distances, surface_areas, E, current_calc.dc, current_calc.df, current_calc.coupling, C, albedos
    )
    return spar.coo_matrix(
        (np.hstack(values), (np.hstack(row_indices), np.hstack(col_indices))), shape=(E * C, E * C)
    ).tocsr()


@numba.njit(cache=True)
def _diffusion_matrix(
    neighbors, distances, surface_areas, E: int, dc, df, coupling, C, albedos
) -> tuple[list[np.array], list[np.array], list[np.array]]:
    row_indices = []
    col_indices = []
    values = []
    for cell in range(C):
        adj = neighbors[cell]
        for j, neighbor in enumerate(adj):
            if neighbor >= 0:
                r, c, v = _compute(
                    cell,
                    E,
                    neighbor,
                    dc[cell],
                    dc[neighbor],
                    distances[cell, j],
                    distances[neighbor, j],
                    df[cell, j],
                    df[neighbor, op_face(j)],
                    surface_areas[cell, j],
                    coupling[cell, j],
                )
            elif neighbor == -1:
                # J = albedo * phi at the boundary
                area = surface_areas[cell, j]
                r, c, v = (
                    np.arange(cell * E, (cell + 1) * E),
                    np.arange(cell * E, (cell + 1) * E),
                    coupling[cell, j] * area + np.full(E, albedos[cell, j] * area),
                )
            elif neighbor == -3:
                # zero flux at the boundary, the finite differences current is D * phi / (l / 2)
                area = surface_areas[cell, j]
                r, c, v = (
                    np.arange(cell * E, (cell + 1) * E),
                    np.arange(cell * E, (cell + 1) * E),
                    (coupling[cell, j] + dc[cell] / distances[cell, j]) * area,
                )
            elif neighbor == -2:
                area = surface_areas[cell, j]
                r, c, v = (
                    np.arange(cell * E, (cell + 1) * E),
                    np.arange(cell * E, (cell + 1) * E),
                    coupling[cell, j] * area,
                )
            else:
                raise ValueError
            row_indices.append(r)
            col_indices.append(c)
            values.append(v)

    return row_indices, col_indices, values
