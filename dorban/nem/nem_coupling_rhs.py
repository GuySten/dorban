r"""
This module contains the creation of the right hand side of the linear equations used for the computation of the nodal coupling coefficients.
These equations are the equations of the 1 dimensional bi-nodal problem. The right hands side of these equations are
composed of the average fluxes at each cell and integrals of the leakage from the cells.
Those leakage integrals are computes using the quadratic leakage approximation, the leakage is approximated to be
a quadratic polynomial and the coefficients of this polynomial are chosen to fit the average leakage of neighboring cells.
"""

import numba
import numpy as np

from dorban.nem.leakage import leakage_coefficients


@numba.njit(cache=True)
def nem_rhs(
    E: int,
    flux1: np.array,
    flux2: np.array,
    transverse_leakage: np.array,
    absorber1: np.array,
    absorber2: np.array,
    fission1: np.array,
    fission2: np.array,
    transverse_leakage_right: np.array,
    transverse_leakage_left: np.array,
    transverse_length: np.array,
    dis1: np.array,
    dis2: np.array,
    transverse_leakage_righter: np.array,
    ll: float,
    l_right: float,
    l_left: float,
    l_righter: float,
    source1: np.array,
    source2: np.array,
) -> np.array:
    r"""
    Function used to compute the right hand side of the equations of the nodal
    coupling coefficients between two adjacent nodes. These equations are the first three moment of the diffusion equation
    as well as the continuity of the current and the discontinuity of the flux.

    It uses the average fluxes, the macroscopic cross sections of the cells, and the transverse leakage at
    the cells and two other cells near them. All those are obtained from the full core diffusion computations and
    assumed to be known when solving the one dimensional bi-nodal NEM problem.
    The order of the cells is illustrated by:
        left | cell | right | righter
    The cell named cell is also referenced as the first cell, and the cell named right is also referenced as the second cell.

    Parameters
    ----------
    E: int
     The number of energy groups
    flux1: np.array
     The average flux at the first cell.
    flux2: np.array
     The average flux at the second cell
    transverse_leakage: np.array
     The leakage from the first cell in the transverse direction. Array of shape (dim - 1) x E
    absorber1: np.array
     The absorber operator of the first cell, it is defined as absorption minus scattering. Array of shape ExE.
    absorber2: np.array
     The absorber operator of the second cell, it is defined as absorption minus scattering. Array of shape ExE.
    fission1: np.array
     The fission operator of the first cell, it is defined as :math:`\chi\otimes\nu\Sigma_f`. Array of shape ExE.
    fission2: np.array
     The fission operator of the second cell, it is defined as :math:`\chi\otimes\nu\Sigma_f`. Array of shape ExE.
    transverse_leakage_right: np.array
     The leakage from the second cell in the transverse direction. Array of shape (dim - 1) x E
    transverse_leakage_left: np.array
     The leakage from the cell named left in the transverse direction. Array of shape (dim - 1) x E
    transverse_length: np.array
     The lengths of the first cell in the transverse direction. Array of shape (dim - 1), if the direction of the NEM
     one dimensional bi-nodal problem is x, this array holds the lengths in the y and z directions.
    dis1: np.array
     the discontinuity factors of the first cell.
    dis2: np.array
     the discontinuity factors of the second cell.
    transverse_leakage_righter: np.array
     The leakage from the cell named righter in the transverse direction. Array of shape (dim - 1) x E
    ll: float
     The length of the cell named cell in the direction of the NEM problem.
    l_right: float
     The length of the cell named right in the direction of the NEM problem.
    l_left: float
     The length of the cell named left in the direction of the NEM problem.
    l_righter:float
     The length of the cell named righter in the direction of the NEM problem.
    source1: np.array
     The source in the first cell. In an eigenvalue problem this is 0.
    source2: np.array
     The source in the second cell. In an eigenvalue problem this is 0.

    Returns
    -------
    np.array
     The right hand side of the one dimensional bi-nodal NEM problem between two cells. Array of shape 8E.
    """
    rhs = np.zeros(8 * E)
    rhs[:E] = -(absorber1 - fission1) @ flux1 + source1
    rhs[2 * E : 3 * E] = -(absorber2 - fission2) @ flux2 + source2
    for i, tl in enumerate(transverse_length):
        rhs[:E] -= transverse_leakage[i] / tl
        rhs[2 * E : 3 * E] -= transverse_leakage_right[i] / tl
    for i, length in enumerate(transverse_length):
        cell = transverse_leakage[i]
        left = transverse_leakage_left[i]
        right = transverse_leakage_right[i]
        rhs[np.hstack((np.arange(E, 2 * E), np.arange(4 * E, 5 * E)))] += _leakage_rhs(
            length, cell, left, right, ll, l_left, l_right
        )
        cell = transverse_leakage_right[i]
        left = transverse_leakage[i]
        right = transverse_leakage_righter[i]
        rhs[np.hstack((np.arange(3 * E, 4 * E), np.arange(5 * E, 6 * E)))] += _leakage_rhs(
            length, cell, left, right, l_right, ll, l_righter
        )
    rhs[7 * E : 8 * E] = dis2 * flux2 - dis1 * flux1
    return rhs


@numba.njit(cache=True)
def nem_boundary_rhs(
    E: int,
    flux: np.array,
    transverse_leakage: np.array,
    absorber: np.array,
    fission: np.array,
    transverse_leakage_right: np.array,
    transverse_leakage_left: np.array,
    transverse_length: np.array,
    ll: float,
    l_right: float,
    l_left: float,
    source: np.array,
    boundary_rhs: np.array,
):
    r"""
    Function used to compute the right hand side of the equations of the nodal
    coupling coefficients between a cell and void boundary condition.
    These equations are the first three moment of the diffusion equation
    as well as a specific equation describing the boundary condition.
    The equation for reflective boundary condition is :math:`J=0` at the boundary, its right hand side is 0.
    The equation for void boundary condition is :math:`J=-\frac{\phi}{2}` at the boundary, its right hands side is -minus
    half the average flux in the cell.

    It uses the average flux, the macroscopic cross sections of the cell, and the transverse leakage at
    the cells and two other cells near it. All those are obtained from the full core diffusion computations and
    assumed to be known when solving the one dimensional bi-nodal NEM problem.
    The order of the cells is illustrated by:
        left | cell | right

    Parameters
    ----------
    E: int
     The number of energy groups
    flux: np.array
     The average flux at the cell.
    transverse_leakage: np.array
     The leakage from the cell in the transverse direction. Array of shape (dim - 1) x E
    absorber: np.array
     The absorber operator of the cell, it is defined as absorption minus scattering. Array of shape ExE.
    fission: np.array
     The fission operator of the cell, it is defined as :math:`\chi\otimes\nu\Sigma_f`. Array of shape ExE.
    transverse_leakage_right: np.array
     The leakage from the second cell in the transverse direction. Array of shape (dim - 1) x E
    transverse_leakage_left: np.array
     The leakage from the cell named left in the transverse direction. Array of shape (dim - 1) x E
    transverse_length: np.array
     The lengths of the first cell in the transverse direction. Array of shape (dim - 1), if the direction of the NEM
     one dimensional bi-nodal problem is x, this array holds the lengths in the y and z directions.
    ll: float
     The length of the cell named cell in the direction of the NEM problem.
    l_right: float
     The length of the cell named right in the direction of the NEM problem.
    l_left: float
     The length of the cell named left in the direction of the NEM problem.
    source: np.array
     The source in the cell. In an eigenvalue problem this is 0.
    boundary_rhs: np.array
     The right hand side of the equation of the boundary condition.

    Returns
    -------
    np.array
     The right hand side of the one dimensional bi-nodal NEM problem between a node and a boundary. array of shape 4E.
    """
    rhs = np.zeros(4 * E)
    rhs[:E] = -(absorber - fission) @ flux + source
    for i, tl in enumerate(transverse_length):
        rhs[:E] -= transverse_leakage[i] / tl
    for i, length in enumerate(transverse_length):
        cell = transverse_leakage[i]
        left = transverse_leakage_left[i]
        right = transverse_leakage_right[i]
        rhs[E : 3 * E] += _leakage_rhs(length, cell, left, right, ll, l_left, l_right)
    rhs[3 * E :] = boundary_rhs
    return rhs


@numba.njit(cache=True)
def _leakage_rhs(length, cell_leakage, left_leakage, right_leakage, ll, l_left, l_right):
    c1, c2 = leakage_coefficients(cell_leakage, left_leakage, right_leakage, ll, l_left, l_right)
    return np.hstack((35 * c2 / length, 10 * c1 / length))
