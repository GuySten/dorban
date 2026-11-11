r"""
This module contains the function to construct diffusion operator
:math:`D\nabla\phi` as a sparse matrix
"""

import numpy as np
import scipy.sparse as spar

from dorban.current_calculator import CurrentCalculator
from dorban.geometry.boundary_conditions import Boundary
from dorban.geometry.geometry import FiniteGeometry


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
     the performance when editing it. The faces between cells are computed at once when the current calculator
     has a face_coefficients method and the geometry face_distances and face_areas methods, otherwise face by face.
    """
    if geometry.dim == 0:
        return spar.csr_matrix((E, E))
    C = geometry.cells
    face_coefficients = getattr(current_calc, "face_coefficients", None)
    if face_coefficients is None or not hasattr(geometry, "face_distances"):
        return _diffusion_matrix_by_face(geometry, E, current_calc)
    neighbors = np.array([[-1 if isinstance(n, Boundary) else n for n in row] for row in geometry.neighbors])
    distances, areas = geometry.face_distances(), geometry.face_areas()
    cells, faces = np.nonzero(neighbors >= 0)
    others = neighbors[cells, faces]
    opposite = np.array([geometry.opposite_face(face) for face in range(neighbors.shape[1])])[faces]
    # the current through each face, seen from its cell: coupling_cell * phi_cell - coupling_other * phi_other
    coupling_cell, coupling_other = face_coefficients(
        cells, others, faces, opposite, distances[cells, faces], distances[others, opposite], areas[cells, faces]
    )
    groups = np.arange(E)
    rows = (cells[:, None] * E + groups).ravel()
    row_indices = [rows, rows]
    col_indices = [rows, (others[:, None] * E + groups).ravel()]
    values = [coupling_cell.ravel(), -coupling_other.ravel()]
    for cell, face in zip(*np.nonzero(neighbors < 0)):
        r, c, v = geometry.neighbors[cell][face].compute_boundary_coefficient(geometry, E, current_calc, cell, face)
        row_indices.append(r)
        col_indices.append(c)
        values.append(v)
    return spar.coo_matrix(
        (np.hstack(values), (np.hstack(row_indices), np.hstack(col_indices))), shape=(E * C, E * C)
    ).tocsr()


def _diffusion_matrix_by_face(geometry: FiniteGeometry, E: int, current_calc: CurrentCalculator) -> spar.csr_matrix:
    """The diffusion matrix built face by face, for current calculators without face_coefficients."""
    C = geometry.cells
    row_indices = []
    col_indices = []
    values = []
    for cell in range(C):
        for j, neighbor in enumerate(geometry.neighbors[cell]):
            if not isinstance(neighbor, Boundary):
                r, c, v = current_calc.compute_current_coefficients(cell, neighbor, j, geometry)
            else:
                r, c, v = neighbor.compute_boundary_coefficient(geometry, E, current_calc, cell, j)
            row_indices.append(r)
            col_indices.append(c)
            values.append(v)
    return spar.coo_matrix(
        (np.hstack(values), (np.hstack(row_indices), np.hstack(col_indices))), shape=(E * C, E * C)
    ).tocsr()
