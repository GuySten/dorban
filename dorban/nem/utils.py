"""
This module contains some utils used in the implementation of the solution of
the NEM equations
"""

from typing import Sequence

import numba
import numpy as np


@numba.njit(cache=True)
def cells_axi_energy_slice(array: np.array, cell: int, axi: Sequence[int]) -> np.array:
    """
    Gets an array whose shape is Cells x 3 x Energy, number of a cell and numbers of axi,
     returns the axis and energy dependent array at the wanted cell and axi. This is an array of shape 2 x Energy.
     This requires a special function because the standard way of doing this in python isn't supported by numba.

    Parameters
    ----------
    array: np.array
    cell : int
    axi: Sequence[int]

    Returns
    -------
    np.array
         array of shape 2 x Energy.
    """
    result = np.zeros((len(axi), array.shape[2]))
    for i, t in enumerate(axi):
        result[i, :] = array[cell, t]
    return result


@numba.njit(cache=True)
def cells_axi_slice(array: np.array, cell: int, axi: Sequence[int]) -> np.array:
    """
    Gets an array whose shape is Cells x 3, number of a cell and numbers of axi,
     returns the axis dependent array at the wanted cell and axi. This is an array of shape 2.
     This requires a special function because the standard way of doing this in python isn't supported by numba.

    Parameters
    ----------
    array: np.array
    cell : int
    axi: Sequence[int]

    Returns
    -------
    np.array
        array of shape 2.
    """
    result = np.zeros(len(axi))
    for i, t in enumerate(axi):
        result[i] = array[cell, t]
    return result


@numba.njit(cache=True)
def compute_coupling_coefficients(
    current: np.array,
    flux1: np.array,
    flux2: np.array,
    D1: np.array,
    D2: np.array,
    l1: float,
    l2: float,
    dis1: np.array,
    dis2: np.array,
) -> np.array:
    r"""
    computes the CMFD coupling coefficient :math:`C` between two adjacent cells. Which is defined by:

        .. math:: C\cdot (flux_1+flux_2)
         =\frac{2\cdot D_1\cdot D_2(flux_2\cdot dis_2-flux_1\cdot dis_1)}
         {D_1\cdot l_2\cdot dis_2+D_2\cdot l_1\cdot dis_1}
         + current


    Parameters
    ----------
    current: np.array
     The current between the cells.
    flux1: np.array
     The average flux at the first cell.
    flux2: np.array
     The average flux at the second cell.
    D1: np.array
     The diffusion coefficient of the first cell
    D2: np.array
     The diffusion coefficient of the second cell
    l1: float
     The width of the first cell.
    l2: float
     The width of the second cell.
    dis1: np.array
     The discontinuity factor of the first cell.
    dis2: np.array
     The discontinue factor of the second cell.

    Returns
    -------
    np.array
     The coupling coefficient.

    """
    return ((2 * D1 * D2 / (D2 * l1 * dis1 + D1 * l2 * dis2)) * (flux2 * dis2 - flux1 * dis1) + current) / (
        flux2 + flux1
    )


@numba.njit(cache=True)
def boundary_coupling_coefficient(current: np.array, flux: np.array, boundary_coef: np.array) -> np.array:
    r"""
    Computes the CMFD coupling coefficient between a cell and a boundary.

                .. math:: C = \frac{current}{flux} - boundary_coefficient


    Parameters
    ----------
    current: np.array
     The current at the boundary.
    flux: np.array
     The average flux at the cell.
    boundary_coef: np.array
     A coefficient which depends on the boundary condition, its 0 for reflective boundary condition and -0.5 for void
     boundary condition

    Returns
    -------
    np.array
     The coupling coefficient.
    """
    return current / flux - boundary_coef


@numba.njit(cache=True)
def current_from_poly_exp(E: int, poly_exp: np.array, D: np.array, ll: float, face: int) -> np.array:
    r"""
    computes the current at a boundary of a cell from the coefficients of the degree 4 polynomial used to represent the
    flux inside the cell. The current is computed using Fick's law

            .. math:: J=-D\nabla\phi

    where :math:`\nabla\phi` at the boundary is computed using the polynomial representation of the flux.

    Parameters
    ----------
    E : int
     The number of energy groups
    poly_exp: np.array
     An array of shape 4E which contains the coefficients which represents the flux in the basis used by NEM.
    D: np.array
     The diffusion coefficient of the cell.
    ll: float
     The width of the cell.
    face: int
     The number of the face on which the current is computed.

    Returns
    -------
    np.array
     The current at the wanted face.
    """
    direction = 2 * (face % 2) - 1
    return (
        -D
        / ll
        * (
            3 * poly_exp[:E]
            + poly_exp[E : 2 * E] / 5
            + direction * (poly_exp[2 * E : 3 * E] + poly_exp[3 * E : 4 * E] / 2)
        )
    )


@numba.njit(cache=True)
def energy_slice(cell, E):
    return np.arange(cell * E, cell * E + E)


@numba.njit(cache=True)
def other_axi(j: int, dim: int) -> np.array:
    """
    Returns the numbers of the axi which aren't equal to j.

    Parameters
    ----------
    j : int
    dim : int

    Returns
    -------
    np.array
     Array of the numbers of axi which aren't equal to j.
    """
    if dim == 3:
        if j == 0:
            return np.array([1, 2])
        elif j == 1:
            return np.array([0, 2])
        elif j == 2:
            return np.array([0, 1])
    elif dim == 2:
        if j == 0:
            return np.array([1])
        elif j == 1:
            return np.array([0])


@numba.njit(cache=True)
def cell_indices_in_two_cell_system(E: int) -> np.array:
    """
    Returns the indices of the first cell in the two node system solved in the updating of the CMFD coupling coefficients.

    Parameters
    ----------
    E: int
     The number of energy groups.

    Returns
    -------
    np.array
    """
    return np.hstack((np.arange(2 * E), np.arange(4 * E, 6 * E)))
