"""
ANL benchmark book (ANL-7416, 1968), problem 3-A1: a homogeneous two dimensional slab, 7 groups with up-scattering
among the thermal groups, for which the continuum eigenvalue is known analytically.

The specification is on pages 34-35 of the book: a 67.5 x 13.5 cm rectangle with zero derivative on the
x = 0 and y = 0 sides and zero flux on the other two, and the cross sections and transfer matrix of page 35
(relative absorption h = 1). The reference is solution 3-A1-1 (pages 36-37): a cosine source gives a cosine flux,
so the continuum eigenvalue is the largest eigenvalue of the 7 x 7 problem with buckling
B^2 = (pi / 2H)^2 + (pi / 2W)^2, H = 13.5 cm and W = 67.5 cm: k = 0.7745451357.
"""

from functools import lru_cache

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import Reflector, ZeroFlux
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core

REFERENCE_K = 0.7745451357
WIDTH, HEIGHT = 67.5, 13.5
# page 35, group macroscopic cross sections (the printed table drops the decimal points: 16 is 1.6, 0 01 is 0.01)
DIFFUSION = np.array([1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4])
ABSORB = np.array([0.01, 0.02, 0.03, 0.10, 0.05, 0.07, 0.09])
REMOVAL = np.array([0.04, 0.06, 0.06, 0.06, 0.08, 0.10, 0.08])
NUSIGMAF = np.array([0.02, 0.01, 0.02, 0.04, 0.10, 0.12, 0.15])
CHI = np.array([0.90, 0.09, 0.01, 0, 0, 0, 0])
# page 35, transfer matrix: TRANSFER[n, g] is the cross section for scattering from group n into group g
TRANSFER = np.array([
    [0, 0.015, 0.01, 0.01, 0.005, 0, 0],
    [0, 0, 0.02, 0.02, 0.01, 0.01, 0],
    [0, 0, 0, 0.02, 0.02, 0.01, 0.01],
    [0, 0, 0, 0, 0.03, 0.02, 0.01],
    [0, 0, 0, 0, 0, 0.05, 0.03],
    [0, 0, 0, 0, 0.02, 0, 0.08],
    [0, 0, 0, 0, 0.02, 0.06, 0],
])  # fmt: skip
SLAB = Fissionable("slab", scatter=TRANSFER.T, absorb=ABSORB, nusigmaf=NUSIGMAF, chi=CHI, diffusion=DIFFUSION)


def _continuum_k():
    buckling = (np.pi / (2 * HEIGHT)) ** 2 + (np.pi / (2 * WIDTH)) ** 2
    loss = np.diag(DIFFUSION * buckling + ABSORB + REMOVAL) - TRANSFER.T
    return np.max(np.abs(np.linalg.eigvals(np.linalg.solve(loss, np.outer(CHI, NUSIGMAF)))))


@lru_cache(maxsize=None)
def _finite_difference_k(cells_along_y):
    mesh = HEIGHT / cells_along_y
    cells_along_x = int(round(WIDTH / mesh))
    geometry = Cartesian(
        [np.full(cells_along_x, mesh), np.full(cells_along_y, mesh)],
        [Reflector(), ZeroFlux(), Reflector(), ZeroFlux()],
    )
    composition = [SLAB] * geometry.cells
    core = Core(composition, 7, geometry, CurrentCalculatorFD.from_isotopes(composition))
    k, _ = solve_k(core, FDSettings(split=geometry.uniform_split(1)))
    return k


def test_cross_sections_reproduce_the_analytical_eigenvalue():
    # guards the transcription of the data: the removal is the row sum of the transfer matrix, and the
    # continuum eigenvalue of the 7 x 7 problem is the book's
    assert np.allclose(TRANSFER.sum(axis=1), REMOVAL)
    assert _continuum_k() == pytest.approx(REFERENCE_K, abs=1e-9)


def test_finite_differences_converge_with_second_order():
    coarse, fine = _finite_difference_k(12), _finite_difference_k(24)  # 1.125 and 0.5625 cm
    assert (coarse - REFERENCE_K) / (fine - REFERENCE_K) == pytest.approx(4, rel=0.01)


def test_extrapolated_finite_differences_match_the_analytical_eigenvalue():
    coarse, fine = _finite_difference_k(12), _finite_difference_k(24)
    richardson = (4 * fine - coarse) / 3
    assert richardson == pytest.approx(REFERENCE_K, abs=1e-6)
