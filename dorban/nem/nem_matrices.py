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
