r"""
This module contains the creation of the matrices used for the computation of the nodal coupling coefficients.
These matrices are used to solve 1 dimensional bi-nodal problem using NEM. The fluxes in both cells are expanded to
degree four polynomials in the base

    .. math:: f_1(x) = x; f_2(x) = 3x^2-\frac{1}{4}; f_3(x) = x^3 - \frac{1}{4}x; f_4(x) = x^4 - \frac{3}{10}x^2 + \frac{1}{80}

The flux in the cells is expanded to :math:`\phi^j(x)=\phi^1_0+\Sigma_i a^j_i\cdot f_i(x)` where :math:`j=1,2` and
  :math:`\phi^j_0` is the average flux at the cell.

The matrices constructed in the module represent the NEM equations in the variables :math:`a^j_i` for :math:`j=1,2;i=1,2,3,4`.
"""

import numba
import numpy as np


@numba.njit(cache=True)
def single_cell_matrix(E: int, ll: float, D: np.array, absorber: np.array, fission: np.array) -> np.array:
    r"""
    This function builds the NEM equations of a single cell, which are the first three moments of the diffusion equations.

    Parameters
    ----------
    E: int
     The number of energy groups
    ll: float
     The width of the cell
    D: np.array
     The diffusion coefficient of the cell. Array of shape E
    absorber: np.array
     The absorber operator of the cell, it is defined as absorption minus scattering. Array of shape ExE.
    fission: np.array
     The fission operator of the cell, it is defined as :math:`\chi\otimes\nu\Sigma_f`. Array of shape ExE.

    Returns
    -------
    np.array
     An array of shape  3 * E x 4 * E which represents the first three moment of the diffusion equation in :math:`a^1_i`
     coordinates.
    """
    D = np.diag(D) / ll
    diff_absorber = np.zeros((3 * E, 4 * E))
    diff_absorber[:E, :E] = -6 * D / ll
    diff_absorber[:E, E : 2 * E] = -2 / 5 * D / ll
    diff_absorber[E : 2 * E, :E] = -35 * absorber
    diff_absorber[E : 2 * E, E : 2 * E] = 140 * D / ll + absorber
    diff_absorber[2 * E : 3 * E, 2 * E : 3 * E] = -10 * absorber
    diff_absorber[2 * E : 3 * E, 3 * E : 4 * E] = 60 * D / ll + absorber
    fission_mat = np.zeros((3 * E, 4 * E))
    fission_mat[E : 2 * E, :E] = -35 * fission
    fission_mat[E : 2 * E, E : 2 * E] = fission
    fission_mat[2 * E : 3 * E, 2 * E : 3 * E] = -10 * fission
    fission_mat[2 * E : 3 * E, 3 * E : 4 * E] = fission
    return diff_absorber - fission_mat


@numba.njit(cache=True)
def nem_mat(
    E: int,
    l1: float,
    l2: float,
    D1: np.array,
    D2: np.array,
    absorber1: np.array,
    absorber2: np.array,
    fission1: np.array,
    fission2: np.array,
    dis1: np.array,
    dis2: np.array,
) -> np.array:
    r"""
    Function that constructs the matrix used for the calculation of the NEM coupling coefficients
    between two adjacent cells.
    There are 8 equations, six of them are the first three moments of the diffusion equation in each of the cells.
    The other two equations are the continuity of the current and the discontinuity of the flux between the cells.

    Parameters
    ----------
    E: int
     The number of energy groups.
    l1: float
     The width of the first cell.
    l2: float
     The width of the second cell.
    D1: np.array
     The diffusion coefficient of the first cell.
    D2: np.array
     The diffusion coefficient of the first cell.
    absorber1: np.array
     The absorber operator of the first cell, it is defined as absorption minus scattering. Array of shape ExE.
    absorber2: np.array
     The absorber operator of the second cell, it is defined as absorption minus scattering. Array of shape ExE.
    fission1: np.array
     The fission operator of the first cell, it is defined as :math:`\chi\otimes\nu\Sigma_f`. Array of shape ExE.
    fission2: np.array
     The fission operator of the second cell, it is defined as :math:`\chi\otimes\nu\Sigma_f`. Array of shape ExE.
    dis1: np.array
     the discontinuity factors of the first cell.
    dis2: np.array
     the discontinuity factors of the second cell.

    Returns
    -------
    np.array
     Array of shape 8E x 8E which represents the eight NEM one dimensional bi-nodal equation in the :math:`a^j_i` coordinates.
    """
    dis1 = np.diag(dis1) / 2
    dis2 = np.diag(dis2) / 2
    full_matrix = np.zeros((8 * E, 8 * E))
    rows1 = np.hstack((np.arange(0, 2 * E), np.arange(4 * E, 5 * E)))
    cols1 = np.hstack((np.arange(0, 2 * E), np.arange(4 * E, 6 * E)))
    first_cell = single_cell_matrix(E, l1, D1, absorber1, fission1)
    for index_row, row in enumerate(rows1):
        for index_col, col in enumerate(cols1):
            full_matrix[row, col] = first_cell[index_row, index_col]

    rows2 = np.hstack((np.arange(2 * E, 4 * E), np.arange(5 * E, 6 * E)))
    cols2 = np.hstack((np.arange(2 * E, 4 * E), np.arange(6 * E, 8 * E)))
    second_cell = single_cell_matrix(E, l2, D2, absorber2, fission2)
    for index_row, row in enumerate(rows2):
        for index_col, col in enumerate(cols2):
            full_matrix[row, col] = second_cell[index_row, index_col]
    D1 = np.diag(D1) / l1
    D2 = np.diag(D2) / l2
    full_matrix[6 * E : 7 * E, :] = np.hstack((-3 * D1, -D1 / 5, -3 * D2, -D2 / 5, -D1, -D1 / 2, D2, D2 / 2))
    full_matrix[
        7 * E : 8 * E,
        np.hstack((np.arange(E), np.arange(2 * E, 3 * E), np.arange(4 * E, 5 * E), np.arange(6 * E, 7 * E))),
    ] = np.hstack((dis1, -dis2, dis1, dis2))
    return full_matrix


@numba.njit(cache=True)
def nem_boundary_mat(
    E: int, ll: float, D: np.array, absorber: np.array, fission: np.array, boundary_eq: np.array
) -> np.array:
    r"""
    Function that constructs the matrix used for the calculation of the NEM coupling coefficients
    between a cell and a boundary condition.
    There are four equations, three of them are the first three moments of the diffusion equation in the cell.
    The last equation is a special equation which depends on the boundary condition.
    The equation for reflective boundary condition is :math:`J=0` at the boundary.
    The equation for void boundary condition is :math:`J=-\frac{\phi}{2}` at the boundary.

    Parameters
    ----------
    E: int
     The number of energy groups.
    ll: float
     The width of the cell.
    D: np.array
     The diffusion coefficient of the cell.
    absorber: np.array
     The absorber operator of the cell, it is defined as absorption minus scattering. Array of shape ExE.
    fission: np.array
     The fission operator of the cell, it is defined as :math:`\chi\otimes\nu\Sigma_f`. Array of shape ExE.
    boundary_eq: np.array
     The equation of the boundary condition

    Returns
    -------
    np.array
     Array of shape 4E x 4E which represents the four NEM one dimensional boundary equation in the :math:`a^1_i` coordinates.
    """
    rows = np.arange(0, 3 * E)
    mat = np.zeros((4 * E, 4 * E))
    mat[rows] = single_cell_matrix(E, ll, D, absorber, fission)
    mat[3 * E : 4 * E, :] = boundary_eq
    return mat
