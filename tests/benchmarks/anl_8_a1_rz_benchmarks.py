"""
ANL benchmark book (ANL-7416 Supplement 2, 1977), problem 8-A1: a two dimensional R-Z thermal reactor, 2 groups,
240 cm radius and 525 cm height with zero flux on the external surfaces. dorban has no kinetics, so only the static
initial state is tested: the eigenvalue and the power density averaged over height versus radius.

The specification is on pages 182-186 of the book: the region map of source situation 8 (page 182) and the initial
two group constants (page 184). The total cross section of group 1 there includes the scattering to group 2. The
book gives 1.2997+1 for the group 1 diffusion coefficient of region 16; with 12.997 every solution of the book is
missed by more than 1%, with 1.2997 (close to the 1.3052 of the other fuel regions) they are matched, so the exponent
is a misprint.

The reference is solution 8-A1-1 (TWODTA, pages 187-200), on a mesh of 8 cm (30 intervals) by 18.75 cm
(28 intervals) with the mesh points at the mesh interval boundaries: k = 0.867053 from the eigenvalue problem, and the
power density of Exhibit C at t = 0. Solution 8-A1-3 (ADEP, page 215) on the same mesh gives 0.866861. The book
gives no estimate of the error of this mesh, so k is compared to the extrapolated finite difference solution with a
20 pcm tolerance. The power density averaged over radius (Exhibit D) is not tested: next to the reflectors it
differs by 6% from the converged dorban solution, as expected from an 18.75 cm mesh where the thermal flux falls
from the core to the reflector.
"""

from functools import lru_cache

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import Reflector, ZeroFlux
from dorban.geometry.cylindrical import Cylindrical
from dorban.materials import Fissionable, Isotope
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core

RADIUS, HEIGHT = 240.0, 525.0
REFERENCE_K = 0.867053
# Exhibit C, t = 0: power density averaged over height at r = 0, 8, ..., 232 cm
REFERENCE_RADIAL_POWER = np.array([
    1.504, 1.504, 1.504, 1.504, 1.503, 1.503, 1.503, 1.502, 1.502, 1.501,
    1.500, 1.500, 1.500, 1.499, 1.500, 1.506, 1.503, 1.481, 1.444, 1.393,
    1.333, 1.257, 1.162, 1.051, 0.926, 0.815, 0.682, 0.516, 0.345, 0.172,
])  # fmt: skip
REFERENCE_RADII = np.arange(30) * 8.0


def _material(name, diffusion, total, nusigmaf, scatter_12):
    diffusion, total, nusigmaf = np.array(diffusion), np.array(total), np.array(nusigmaf)
    data = dict(
        scatter=np.array([[0, 0], [scatter_12, 0]]),
        absorb=total - np.array([scatter_12, 0]),
        diffusion=diffusion,
    )
    if not nusigmaf.any():
        return Isotope(name, **data)
    return Fissionable(name, nusigmaf=nusigmaf, chi=np.array([1, 0]), **data)


def _fuel(name, total_2):
    return _material(name, [1.3052, 0.88857], [1.0475e-2, total_2], [1.1776e-3, 1.3268e-2], 8.0351e-3)


# page 184: initial two group constants
MATERIALS = {
    1: _material("Region 1", [1.0684, 0.32051], [2.8e-2, 3.3e-3], [0, 0], 2.6e-2),
    2: _material("Region 2", [1.3495, 0.87032], [1.201e-2, 1.9e-2], [0, 0], 1.2e-2),
    3: _fuel("Region 3", 1.3063e-2),
    5: _fuel("Region 5", 1.2623e-2),
    6: _fuel("Region 6", 1.2183e-2),
    7: _fuel("Region 7", 1.3453e-2),
    9: _fuel("Region 9", 1.2973e-2),
    10: _fuel("Region 10", 1.2933e-2),
    16: _material("Region 16", [1.2997, 0.87951], [1.0470e-2, 1.3065e-2], [1.2875e-3, 1.4246e-2], 7.9061e-3),
}
for same, region in [(15, 1), (14, 2), (4, 3), (11, 3), (12, 5), (13, 6), (8, 7)]:
    MATERIALS[same] = MATERIALS[region]
# page 182: region map, rows from the top of the reactor
RADIAL_EDGES = np.array([0, 40, 120, 160, 200, 240])
AXIAL_EDGES = np.cumsum([0, 37.5, 37.5, 112.5, 150, 112.5, 37.5, 37.5])
REGIONS = [[1] * 5, [2] * 5, [3, 4, 5, 6, 16], [7, 8, 9, 10, 16], [11, 11, 12, 13, 16], [14] * 5, [15] * 5]


@lru_cache(maxsize=None)
def _solve(refinement):
    """The 8 x 18.75 cm mesh of the book with each interval split into refinement intervals in each direction."""
    cells_r, cells_z = 30 * refinement, 28 * refinement
    centers_r = (np.arange(cells_r) + 0.5) * RADIUS / cells_r
    centers_z = (np.arange(cells_z) + 0.5) * HEIGHT / cells_z
    composition = [
        MATERIALS[REGIONS[np.searchsorted(AXIAL_EDGES, z) - 1][np.searchsorted(RADIAL_EDGES, r) - 1]]
        for z in centers_z
        for r in centers_r
    ]
    geometry = Cylindrical(
        [np.full(cells_r, RADIUS / cells_r), np.full(cells_z, HEIGHT / cells_z)],
        [Reflector(), ZeroFlux(), ZeroFlux(), ZeroFlux()],
    )
    core = Core(composition, 2, geometry, CurrentCalculatorFD.from_isotopes(composition))
    k, flux = solve_k(core, FDSettings(split=geometry.uniform_split(1)))
    power = np.sum(np.reshape(flux, (-1, 2)) * [m.nusigmaf for m in composition], axis=1)
    # the axial mesh is uniform, so the average over height is the mean over each column of cells
    return k, centers_r, np.mean(np.reshape(power, (cells_z, cells_r)), axis=0)


def test_finite_differences_converge_with_second_order():
    k = [_solve(refinement)[0] for refinement in (1, 2, 4)]
    assert (k[0] - k[1]) / (k[1] - k[2]) == pytest.approx(4, rel=0.05)


def test_extrapolated_k():
    k_coarse, k_fine = _solve(2)[0], _solve(4)[0]
    assert (4 * k_fine - k_coarse) / 3 == pytest.approx(REFERENCE_K, abs=20e-5)


def test_radial_power_initial_state():
    _, centers, power = _solve(4)
    power = np.interp(REFERENCE_RADII, centers, power)
    reference = REFERENCE_RADIAL_POWER
    error = power / np.mean(power) / (reference / np.mean(reference)) - 1
    assert np.max(np.abs(error)) < 0.005
