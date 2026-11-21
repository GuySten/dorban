"""
tests for the NEM source solver against the exact solution of the diffusion equation in a slab
"""

import numpy as np
import pytest

from dorban import Cartesian, Core, Fissionable, Isotope, Reflector, Void
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.cmfd_iterator import GeneratorNemCMFDSource
from dorban.nem.solve_nem import NEMSettings, solve_source_nem

E = 2
MATERIAL_DATA = dict(
    scatter=np.array([[0, 0], [0.02, 0]]),
    absorb=np.array([0.01, 0.08]),
    diffusion=np.array([1.5, 0.4]),
)
FUEL = Fissionable("Fuel", nusigmaf=np.array([0.004, 0.06]), chi=np.array([1, 0]), **MATERIAL_DATA)
ABSORBER = Isotope("Absorber", **MATERIAL_DATA)
MATERIALS = pytest.mark.parametrize("material", [FUEL, ABSORBER], ids=["fissionable", "absorber"])


def exact_slab_cell_averages(material, edges, source_density):
    r"""
    Cell averages of the exact solution of the diffusion equation in a homogeneous slab
    :math:`-a < x < a`, with a uniform source density :math:`q` and the condition of :class:`Void`,
    :math:`J = \phi / 2`, at both faces.

    The equation is :math:`-D \phi'' + T \phi = q`, where :math:`T` is the removal minus the production
    matrix. Its symmetric solution is :math:`\phi = T^{-1} q + \sum_i c_i v_i \cosh(\kappa_i x)`, where
    :math:`(\kappa_i^2, v_i)` are the eigenpairs of :math:`D^{-1} T`, and the boundary condition at
    :math:`x = a` determines the :math:`c_i`. When the production exceeds the removal, :math:`\kappa_i^2 < 0`
    and the mode is a cosine, which the complex arithmetic takes care of.
    """
    # scatter[i, j] is from group j to group i, the in-group terms cancel
    T = np.diag(material.absorb + material.scatter.sum(axis=0)) - material.scatter
    if material.isfissile:
        T -= np.outer(material.chi / material.chi.sum(), material.nusigmaf)
    D = np.diag(material.diffusion)
    q = np.full(E, source_density)
    a = edges[-1]

    particular = np.linalg.solve(T, q)
    kappa_squared, modes = np.linalg.eig(np.linalg.solve(D, T))
    kappa = np.sqrt(kappa_squared.astype(complex))
    # -D phi'(a) = phi(a) / 2
    boundary = -D @ modes @ np.diag(kappa * np.sinh(kappa * a)) - 0.5 * modes @ np.diag(np.cosh(kappa * a))
    c = np.linalg.solve(boundary, 0.5 * particular.astype(complex))
    left, right = edges[:-1, None], edges[1:, None]
    cosh_averages = (np.sinh(kappa * right) - np.sinh(kappa * left)) / (kappa * (right - left))
    return (particular + (cosh_averages * c) @ modes.T).real


def slab_core(material, n, width):
    """
    A slab of n cells along x: Void at both x faces and Reflector at the y faces, so the solution does not depend on y.
    """
    geometry = Cartesian([np.full(n, width), np.full(2, width)], [Void(), Void(), Reflector(), Reflector()])
    materials = [material] * geometry.cells
    return Core(materials, E, geometry, CMFDCurrentCalculator.from_isotopes(materials, 2))


@MATERIALS
@pytest.mark.parametrize("width, rtol", [(2.0, 2e-6), (3.0, 1.5e-5), (5.0, 1e-4)])
def test_solve_source_nem_matches_exact_slab_solution(material, width, rtol):
    """
    With the iteration converged, what remains is the error of the NEM expansion, which is of fourth order in the
    cell width: about 6e-7, 4e-6 and 4e-5 for 2, 3 and 5 cm cells.
    """
    n = 6
    core = slab_core(material, n, width)
    # the source is the total source in each cell, a density of 1 per unit volume
    source = np.repeat(core.geometry.volumes, E)
    flux = solve_source_nem(core, NEMSettings(flux_rtol=1e-7, max_iter=200), source).reshape(2, n, E)
    edges = width * (np.arange(n + 1) - n / 2)
    exact = exact_slab_cell_averages(material, edges, 1.0)
    for row in flux:
        np.testing.assert_allclose(row, exact, rtol=rtol)


@MATERIALS
def test_source_iteration_stops_on_tolerance_with_reflective_boundaries(material):
    """
    The coupling coefficients at reflective faces are zero at every iteration, and the iteration must still stop
    once it converges. A stopping test that waits for every coefficient to be nonzero never stops on tolerance here,
    and solve_source_nem, with no maximal number of iterations by default, never returns.
    """
    core = slab_core(material, 6, 3.0)
    solver = GeneratorNemCMFDSource(core, np.repeat(core.geometry.volumes, E), max_iter=200)
    for _ in solver:
        continue
    assert np.any(solver.current_calc.coupling == 0)
    assert solver.iter < solver.max_iter
