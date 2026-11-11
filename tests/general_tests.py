import numpy as np
import scipy.sparse as spar

from dorban.finite_differences.solve_finite_difference import (
    solve_adjoint,
    solve_k_diffusion,
)
from dorban.geometry.boundary_conditions import Boundary, KnownCurrent, Void
from dorban.solve_equation import flux_with_given_boundary_current


def test_flux_with_given_boundary_current(CAR_2dim):
    system = CAR_2dim
    geo = system.geometry
    for cell in range(geo.cells):
        for face, neighbor in enumerate(geo.neighbors[cell]):
            if isinstance(neighbor, Boundary):
                geo.neighbors[cell][face] = Void()
    k, flux = solve_k_diffusion(system, rtol_vector=1e-10, atol_vector=1e-10)
    flux = flux * np.sign(np.sum(flux))  # the sign of an eigenvector is arbitrary
    for cell in range(geo.cells):
        for face, neighbor in enumerate(geo.neighbors[cell]):
            if isinstance(neighbor, Boundary):
                row, col, val = neighbor.compute_boundary_coefficient(
                    system.geometry, system.E, system.current_calc, cell, face
                )
                mat = spar.coo_matrix(
                    (val, (row, col)), shape=(system.size, system.size)
                )
                # the current leaving through the void boundary; KnownCurrent takes the current entering
                outgoing = mat @ flux
                outgoing = outgoing[np.nonzero(outgoing)]
                geo.neighbors[cell][face] = KnownCurrent(
                    -outgoing / system.geometry.surface_area(cell, face)
                )
    flux2 = flux_with_given_boundary_current(system, k, rtol=1e-10, atol=1e-10)
    flux = flux / np.sum(flux)
    flux2 = flux2 / np.sum(flux2)
    assert np.all(np.abs((flux - flux2) / flux) < 1e-5)


def test_adjoint_eignvalue_equals_direct_eigenvalue(CAR_2dim):
    system = CAR_2dim
    k, flux = solve_k_diffusion(system)
    k_adjoint, flux_adjoint = solve_adjoint(system)
    assert np.isclose(k, k_adjoint)
