"""
THis modules allows to compute rdfs using nem approximation, i.e assuming that
the integral of the flux across the traverse direction is a polynomial of degree
4
"""

from itertools import product
from typing import Optional

import numpy as np

from dorban.cross_sections_operators import _absorb as absorb
from dorban.cross_sections_operators import _nufission as nufission
from dorban.geometry.boundary_conditions import Boundary
from dorban.materials import CrossSectionData
from dorban.nem.nem_coupling_rhs import _leakage_rhs
from dorban.nem.nem_matrices import single_cell_matrix
from dorban.system import Core


def compute_boundary_fluxes_using_nem(
    current_right: np.array,
    current_left: np.array,
    transverse_leakage: np.array,
    ll: float,
    l_left: float,
    l_right: float,
    transverse_length: float,
    transverse_leakage_left: np.array,
    transverse_leakage_right: np.array,
    mat: CrossSectionData,
    k: float,
    flux: Optional[np.array] = None,
) -> tuple[np.array, np.array]:
    r"""
    Function to compute the fluxes on two opposite boundaries of a cartesian two dimensional cell,
    assuming the currents on the boundaries of the cell are known. The k multiplication factor is also known.
    The computation is done using the NEM approximation. The flux is integrated in the axi transverse to the direction
    of the boundaries. Then the flux expanded in the NEM basis :math:`\phi(x)=\phi_0+\Sigma_i a_i\cdot f_i(x)`.
    Where :math:`\phi_0` is the average flux in the cell and

    .. math:: f_1(x) = x; f_2(x) = 3x^2-\frac{1}{4}; f_3(x) = x^3 - \frac{1}{4}x; f_4(x) = x^4 - \frac{3}{10}x^2 + \frac{1}{80}.

    Then four equation are written to determine the values of :math:`a_i`. The first three are the first three moment of
    the diffusion equation, the last one is an equation about the current at the right boundary.
    Then the flux on the boundaries is deduced from :math:`a_i` and :math:`\phi_0`.

    See Also
    --------
    ~dorban.nem.nem_matrices
    ~dorban.nem.nem_coupling_rhs


    Parameters
    ----------
    current_right: np.array
     The current on the right boundary.
    current_left: np.array
     The current on the left boundary.
    transverse_leakage: np.array
     The average leakage from the transverse directions.
    ll: float
     The length of the cell.
    l_left: float
     The length of the cell to the left.
    l_rightL float
     The length of the cell to the left.
    transverse_length: float
     The length of the cell in the transverse direction to the main direction. If the direction of the NEM
     one dimensional bi-nodal problem is x, this is the length in the y direction.
    transverse_leakage_right: np.array
     The leakage from the second cell in the transverse direction. Array of shape E.
    transverse_leakage_left: np.array
     The leakage from the cell named left in the transverse direction. Array of shape E.
    mat: CrossSectionData
     The data about the cross section of the cell.
    k: float
     The multiplication factor.
    flux: Optional[np.array]
     The value of the average flux in the cell. If it is known it is used, if it isn't known its computed from the
     leakage from the cell and the cross sections of the cell.

    Returns
    -------
    tuple[np.array,np.array]
     Tuple of the flux on the left boundary and the flux on the right boundary.
    """
    E = mat.E
    D = np.diag(mat.diffusion / ll)
    matrix = np.zeros((4 * E, 4 * E))
    matrix[np.arange(0, 3 * E)] = single_cell_matrix(E, ll, mat.diffusion, absorb(mat), nufission(mat) / k)
    matrix[3 * E :, :] = np.hstack([-3 * D, -D / 5, -D, -D / 2])
    rhs = np.zeros(4 * E)
    rhs[:E] = (current_right - current_left) / ll
    if np.any(flux):
        average_flux = flux
    else:
        average_flux = -np.linalg.solve(
            absorb(mat) - nufission(mat) / k, (rhs[:E] + (transverse_leakage / transverse_length)).flatten()
        )
    cell = transverse_leakage
    left = transverse_leakage_left
    right = transverse_leakage_right
    rhs[E : 3 * E] = _leakage_rhs(transverse_length, cell, left, right, ll, l_left, l_right)
    rhs[3 * E : 4 * E] = current_right
    coefficients = np.linalg.solve(matrix, rhs)
    a2, a1 = coefficients[:E], coefficients[2 * E : 3 * E]
    return (average_flux - a1 / 2 + a2 / 2), (average_flux + a1 / 2 + a2 / 2)


def _fake_neighbor_leakage(neighbor, leakage, cell, cell_leakage):
    if not isinstance(neighbor, Boundary):
        return leakage[neighbor]
    # the mirror image across a reflective boundary, albedo 0, as in dorban.nem.two_node._neighbor_leakage
    reflective = getattr(neighbor, "albedo", None) == 0
    return cell_leakage if reflective else -cell_leakage


def compute_rdfs(system: Core, currents: np.array, k: float, fluxes: Optional[np.array] = None) -> np.array:
    r"""
    Function to compute reference discontinuity factors (rdfs) of a two-dimensional cartesian system using the NEM approximation.
    Assuming the currents between any two cells in known and the k eigenvalue is also known.
    This function is useful to compute rdfs from a unit cell computation, if all the currents between the
    cell are also known from the unit cell computations.
    If the average flux in each cell is given it is used, if it isn't given it is deduced from the currents.
    The computation of the average flux from the currents might be less exact than using the average flux computed
    in a unit cell computation, for example if a Monte Carlo code is used for the unit cell computation, so it is
    better to provide it.

    Parameters
    ----------
    system: Core
     The core whose rdfs are computed. Assumed to be two-dimensional.
    currents: np.array
     Array of the current between each two cells, the shape of this array is the number of cells times the number
     of faces of each cell times the number of energy groups.
    k: float
     The k multiplication factor of the system.
    fluxes: Optional[np.array]
     The average flux at each cell. If it is known the average fluxes are deduced from the currents.

    Returns
    -------
    np.array
     Array of the reference discontinuity factors of each cell on each face: the inverse of the flux the
     homogeneous NEM cell has on the face (the heterogeneous face flux is common to both sides of a face, so it
     cancels in the flux continuity condition :math:`F_1\phi_1=F_2\phi_2` and is not needed).
    """
    leakage_x = currents[:, 1] - currents[:, 0]
    leakage_y = currents[:, 3] - currents[:, 2]
    leakages = {0: leakage_x, 1: leakage_y}
    C = system.geometry.cells
    face_flux = np.zeros((C, 4, system.E))  # flux of the homogeneous NEM cell on each face
    for (cell, neighbors), axis in product(enumerate(system.geometry.neighbors), [0, 1]):
        current_right = currents[cell, 2 * axis + 1]
        current_left = currents[cell, 2 * axis]
        transverse_leakage = leakages[not axis][cell]
        ll = 2 * system.geometry.distance_to_face(cell, 2 * axis)
        l_left = (
            2 * system.geometry.distance_to_face(n, 2 * axis)
            if not isinstance(n := neighbors[2 * axis], Boundary)
            else ll
        )
        l_right = (
            2 * system.geometry.distance_to_face(n, 2 * axis)
            if not isinstance(n := neighbors[2 * axis + 1], Boundary)
            else ll
        )
        transverse_l = 2 * system.geometry.distance_to_face(cell, 2 * (not axis))
        # across a boundary the fake neighbor of the NEM kernel (dorban.nem.two_node._neighbor_leakage): the
        # mirror image (the cell's own leakage) across a reflector, minus the cell's own leakage otherwise
        transverse_left = _fake_neighbor_leakage(neighbors[2 * axis], leakages[not axis], cell, transverse_leakage)
        transverse_right = _fake_neighbor_leakage(neighbors[2 * axis + 1], leakages[not axis], cell, transverse_leakage)
        mat = system.isotopes[cell]
        if np.any(fluxes):
            face_flux[cell, slice(2 * axis, 2 * axis + 2)] = compute_boundary_fluxes_using_nem(
                current_right,
                current_left,
                transverse_leakage,
                ll,
                l_left,
                l_right,
                transverse_l,
                transverse_left,
                transverse_right,
                mat,
                k,
                fluxes[cell],
            )
        else:
            face_flux[cell, slice(2 * axis, 2 * axis + 2)] = compute_boundary_fluxes_using_nem(
                current_right,
                current_left,
                transverse_leakage,
                ll,
                l_left,
                l_right,
                transverse_l,
                transverse_left,
                transverse_right,
                mat,
                k,
            )
    return 1.0 / face_flux
