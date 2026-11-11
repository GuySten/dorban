"""
tests for the Cylindrical (R-Z) geometry: volumes and areas, and the finite difference eigenvalue of a bare finite
cylinder against the exact solution
"""

import numpy as np
import pytest
import scipy.special as sp

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import Reflector, Void, ZeroFlux
from dorban.geometry.cylindrical import Cylindrical
from dorban.materials import Fissionable
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.solve_nem import NEMSettings
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core

DIFFUSION, ABSORB, NUSIGMAF, RADIUS, HALF_HEIGHT = 1.0, 0.05, 0.06, 50.0, 50.0
FUEL = Fissionable(
    "fuel",
    scatter=np.array([[0.0]]),
    absorb=np.array([ABSORB]),
    nusigmaf=np.array([NUSIGMAF]),
    chi=np.array([1.0]),
    diffusion=np.array([DIFFUSION]),
)


def _geometry(cells):
    # the upper half of the cylinder, reflective at the midplane
    return Cylindrical(
        [np.full(cells, RADIUS / cells), np.full(cells, HALF_HEIGHT / cells)],
        [Reflector(), ZeroFlux(), Reflector(), ZeroFlux()],
    )


def _finite_difference_k(geometry, split=None):
    composition = [FUEL] * geometry.cells
    core = Core(composition, 1, geometry, CurrentCalculatorFD.from_isotopes(composition))
    k, _ = solve_k(core, FDSettings(split=split or geometry.uniform_split(1)))
    return k


def _exact_k():
    """phi = J0(2.405 r / R) cos(pi z / H): k = nu Sigma_f / (Sigma_a + D B^2)."""
    buckling = (sp.jn_zeros(0, 1)[0] / RADIUS) ** 2 + (np.pi / (2 * HALF_HEIGHT)) ** 2
    return NUSIGMAF / (ABSORB + DIFFUSION * buckling)


def test_volumes_and_areas():
    geometry = Cylindrical([np.array([1.0, 2.0]), np.array([3.0, 4.0])], [Reflector(), Void(), Void(), Void()])
    # cells are numbered along r first: (r, z) = (0, 0), (1, 0), (0, 1), (1, 1)
    assert geometry.volumes == pytest.approx(np.pi * np.array([1 * 3, 8 * 3, 1 * 4, 8 * 4]))
    assert np.sum(geometry.volumes) == pytest.approx(np.pi * 3**2 * 7)
    assert [geometry.surface_area(1, face) for face in range(4)] == pytest.approx(
        [2 * np.pi * 1 * 3, 2 * np.pi * 3 * 3, np.pi * 8, np.pi * 8]
    )
    assert geometry.surface_area(0, 0) == 0
    assert geometry.distance_to_face(1, 0) == geometry.distance_to_face(1, 1) == 1.0


def test_one_dimensional_cylinder_volumes():
    geometry = Cylindrical([np.array([1.0, 1.0])], [Reflector(), Void()])
    assert geometry.volumes == pytest.approx([np.pi, 3 * np.pi])
    assert geometry.surface_area(1, 1) == pytest.approx(4 * np.pi)


def test_the_axis_must_be_reflective():
    with pytest.raises(ValueError):
        Cylindrical([np.array([1.0])], [Void(), Void()])


def test_refined_geometry_is_cylindrical():
    refined = _geometry(5).refine_mesh(_geometry(5).uniform_split(2))
    assert isinstance(refined, Cylindrical)
    assert np.sum(refined.volumes) == pytest.approx(np.pi * RADIUS**2 * HALF_HEIGHT)


def test_finite_differences_converge_with_second_order():
    coarse, fine = _finite_difference_k(_geometry(20)), _finite_difference_k(_geometry(40))
    assert (coarse - _exact_k()) / (fine - _exact_k()) == pytest.approx(4, rel=0.01)
    assert (4 * fine - coarse) / 3 == pytest.approx(_exact_k(), abs=1e-8)


def test_refinement_by_split_matches_the_fine_geometry():
    assert _finite_difference_k(_geometry(10), _geometry(10).uniform_split(2)) == pytest.approx(
        _finite_difference_k(_geometry(20)), abs=1e-10
    )


def test_nem_is_not_implemented():
    geometry = _geometry(2)
    composition = [FUEL] * geometry.cells
    core = Core(composition, 1, geometry, CMFDCurrentCalculator.from_isotopes(composition, 2))
    with pytest.raises(NotImplementedError):
        solve_k(core, NEMSettings())
