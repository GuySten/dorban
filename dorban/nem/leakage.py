"""
this file will handle the leakage computations for NEM
"""

from typing import Tuple

import numba
import numpy as np

from dorban.nem.cmfd_current_calculator import _values

Cell = int
Length = float


@numba.njit(cache=True)
def leakage_coefficients(
    cell_leakage: np.array,
    left_leakage: np.array,
    right_leakage: np.array,
    length_c: Length,
    length_l: Length,
    length_r: Length,
) -> Tuple[np.array, np.array]:
    """
    The shape of the axial leakage is assumed to be quadratic, it is determined
    by the leakage in the cell and the leakage at the adjoin cells, there are
    2 coefficients that have to be computed in order to determine the leakage shape
    in each cell.
    This function computes and returns them.
    This function gets the required data about a cell and its two neighbors,
    and it computes the leakage in an axis which is transverse to the line
    that connects the cell and its two neighbors. The function implements the analytical solution of the three linear
    equations obtained from the conditions on the leakage at the cell and its neighbors.

    Parameters
    ----------
    cell_leakage: np.array
     the leakage in the main cell in the transverse axis direction
    left_leakage: np.array
     the leakage in the left cell in the transverse axis direction
    right_leakage: np.array
      the leakage in the right cell in the transverse axis direction
    length_c: float
     the length of the cell in the main axis
    length_l: float
     the length of the left cell in the main axis
    length_r: float
     the length of the right cell in the main axis

    Returns
    -------
    Tuple[np.array, np.array]
     the coefficients of the leakage function at the cell in all
    energy groups , the first value is the linear coefficient and the second is
    the quadratic one.
    """
    m = (length_c + length_l) * (length_c + length_r) * (length_c + length_l + length_r)
    p1 = (
        1
        / m
        * length_c
        * (
            (right_leakage - cell_leakage) * (length_c + 2 * length_l) * (length_c + length_l)
            + (cell_leakage - left_leakage) * (length_c + 2 * length_r) * (length_c + length_r)
        )
    )
    p2 = (
        1
        / m
        * length_c**2
        * (
            (right_leakage - cell_leakage) * (length_c + length_l)
            + (left_leakage - cell_leakage) * (length_c + length_r)
        )
    )
    return p1, p2


@numba.njit(cache=True, parallel=True)
def leakage(
    neighbors: np.array,
    distances: np.array,
    surfaces_areas: np.array,
    dc: np.array,
    df: np.array,
    coupling: np.array,
    E: int,
    flux: np.array,
    dim: int,
    albedos: np.array,
) -> np.array:
    r"""
    Computes the average leakage at all directions from each cell, assuming the flux is known. The computation of
    current is abased on the modified Fick's Law :math:`J=-D\nabla\phi+\hat{D}\phi`.

    Parameters
    ----------
    neighbors: np.array
     Array whose shape is :math:`Cells x 2\cdot dim`, at row i it contains the numbers of the neighbors of cell i.
     Negative values represent boundary conditions.
    distances: np.array
     Array whose shape is :math:`Cells x 2\cdot dim`, at (i,j) it contains the distance between the center of cell i
     to it's j face.
    surfaces_areas: np.array
     Array whose shape is :math:`Cells x 2\cdot dim`, at (i,j) it contains the surface area of face j of cell i.
    dc: np.array
     The array of diffusion coefficients of each cell.
    df: np.array
     The array of discontinuity factors of each cell.
    coupling: np.array
     The array of all the CMFD coupling coefficients.
    E: int
     The number of energy groups
    flux: np.array
     The array of the average flux at each cell
    dim: int
     The dimension of the system.
    albedos: np.array
     The albedo of each face, see :func:`dorban.nem.cmfd_iterator.boundary_albedos`.

    Returns
    -------
    np.array
     Array whose shape is :math:`Cells x dim x E` which contains the leakages from each cell in each direction.
    """
    result = np.zeros((neighbors.shape[0], dim, E))
    for index in numba.prange(neighbors.shape[0]):
        cell = np.int64(index)  # the parallel loop index is unsigned; a signed one keeps cell and neighbors alike
        adj = neighbors[cell]
        for axis in range(dim):
            left, right = adj[2 * axis : 2 * axis + 2]
            if right >= 0:
                right_current = current_between_two_cells(
                    cell, right, axis, distances, surfaces_areas, dc, df, coupling, E, flux
                )
            elif right == -1 or right == -3:
                right_current = current_cell_and_boundary(
                    surfaces_areas[cell, 2 * axis + 1],
                    coupling[cell, 2 * axis + 1],
                    _boundary_coefficient(right, albedos[cell, 2 * axis + 1], dc[cell], distances[cell, 2 * axis + 1]),
                    flux[cell * E : cell * E + E],
                )
            else:
                right_current = np.zeros(E)
            if left >= 0:
                left_current = -current_between_two_cells(
                    left, cell, axis, distances, surfaces_areas, dc, df, coupling, E, flux
                )
            elif left == -1 or left == -3:
                left_current = current_cell_and_boundary(
                    surfaces_areas[cell, 2 * axis],
                    coupling[cell, 2 * axis],
                    _boundary_coefficient(left, albedos[cell, 2 * axis], dc[cell], distances[cell, 2 * axis]),
                    flux[cell * E : cell * E + E],
                )
            else:
                left_current = np.zeros(E)
            result[cell, axis, :] = (right_current + left_current) / surfaces_areas[cell, 2 * axis + 1]
    return result


@numba.njit(cache=True)
def current_between_two_cells(
    cell: int,
    neighbor: int,
    axis: int,
    distances: np.array,
    surfaces_areas: np.array,
    dc: np.array,
    df: np.array,
    coupling: np.array,
    E: int,
    flux: np.array,
) -> np.array:
    """
    Function that computes the average current between two cells, assuming the flux is known using the modified Fick's law
    """
    return np.sum(
        np.reshape(
            (
                _values(
                    dc[cell],
                    dc[neighbor],
                    distances[cell, 2 * axis + 1],
                    distances[neighbor, 2 * axis + 1],
                    df[cell, 2 * axis + 1],
                    df[neighbor, 2 * axis],
                    surfaces_areas[cell, 2 * axis + 1],
                    coupling[cell, 2 * axis + 1],
                )
            )
            * np.hstack((flux[neighbor * E : neighbor * E + E], flux[cell * E : cell * E + E])),
            (2, E),
        ),
        axis=0,
    )


@numba.njit(cache=True)
def _boundary_coefficient(boundary: int, albedo: float, dc: np.array, distance: float) -> np.array:
    """
    The current through a boundary face per unit flux and area without the coupling coefficient: the albedo for
    J = albedo * phi (number -1) and the finite differences D / (l / 2) for zero flux (number -3).
    """
    if boundary == -3:
        return dc / distance
    return np.full(dc.shape[0], albedo)


@numba.njit(cache=True)
def current_cell_and_boundary(
    surface_area: float, coupling: np.array, coefficient: np.array, flux: np.array
) -> np.array:
    """
    Function that computes the average current between a cell and a boundary condition, assuming the flux is known
    using the modified Fick's law
    """
    return (coupling + coefficient) * surface_area * flux


@numba.njit(cache=True)
def current_cell_and_void(surface_area: float, coupling: np.array, E: int, flux: np.array) -> np.array:
    """
    Function that computes the average current between a cell and void boundary condition,
     assuming the flux is known using the modified Fick's law
    """
    return (coupling * surface_area + np.full(E, surface_area / 2)) * flux
