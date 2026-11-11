r"""
This module contains the function to construct diffusion operator
:math:`D\nabla\phi` as a sparse matrix
"""

import numpy as np
import scipy.sparse as spar

from dorban.current_calculator import CurrentCalculator
from dorban.geometry.geometry import BOUNDARY, FiniteGeometry


def diffusion_matrix(geometry: FiniteGeometry, E: int, current_calc: CurrentCalculator) -> spar.csr_matrix:
    r"""
    This function will generate a sparse matrix representing the diffusion
    operator.



    Parameters
    ----------
    geometry: FiniteGeometry
     The geometry of the system
    E: int
     The number of energy groups
    current_calc: CurrentCalculator
     The current calculator of the system

    Returns
    -------
    spar.csr_matrix
     sparse matrix in csr format representing the linear operator flux goes to
     integral of :math:`D\nabla \phi` , :math:`\phi` being the flux.

     Warnings
     --------
     This function is one of the bottlenecks of the code, be careful to check
     the performance when editing it. The faces between cells are computed at once, from the face arrays of the
     geometry and the face_coefficients of the current calculator; the boundary faces one by one.
    """
    if geometry.dim == 0:
        return spar.csr_matrix((E, E))
    C = geometry.cells
    neighbors, distances, areas = geometry.face_neighbors(), geometry.face_distances(), geometry.face_areas()
    cells, faces = np.nonzero(neighbors >= 0)
    others = neighbors[cells, faces]
    opposite = np.array([geometry.opposite_face(face) for face in range(neighbors.shape[1])])[faces]
    # the current through each face, seen from its cell: coupling_cell * phi_cell - coupling_other * phi_other
    coupling_cell, coupling_other = current_calc.face_coefficients(
        cells,
        others,
        faces,
        opposite,
        distances[cells, faces],
        distances[others, opposite],
        areas[cells, faces],
        geometry,
    )
    groups = np.arange(E)
    rows = (cells[:, None] * E + groups).ravel()
    row_indices = [rows, rows]
    col_indices = [rows, (others[:, None] * E + groups).ravel()]
    values = [coupling_cell.ravel(), -coupling_other.ravel()]
    for cell, face in zip(*np.nonzero(neighbors == BOUNDARY)):
        r, c, v = geometry.neighbors[cell][face].compute_boundary_coefficient(geometry, E, current_calc, cell, face)
        row_indices.append(r)
        col_indices.append(c)
        values.append(v)
    return spar.coo_matrix(
        (np.hstack(values), (np.hstack(row_indices), np.hstack(col_indices))), shape=(E * C, E * C)
    ).tocsr()
