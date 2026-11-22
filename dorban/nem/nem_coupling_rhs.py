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
def _leakage_rhs(length, cell_leakage, left_leakage, right_leakage, ll, l_left, l_right):
    c1, c2 = leakage_coefficients(cell_leakage, left_leakage, right_leakage, ll, l_left, l_right)
    return np.hstack((35 * c2 / length, 10 * c1 / length))
