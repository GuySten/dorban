"""
ANL benchmark book (ANL-7416 Supplement 2, 1977), problem 14-A1: the two dimensional BWR of the super
prompt-critical transient, 2 groups with axial buckling 1.0e-4, quarter core of 11 x 11 cells of 15 cm with zero
flux on the external surfaces. dorban has no kinetics, so only the static results are tested: the eigenvalue
and the assembly powers of the initial state, and the eigenvalue with the rod withdrawn (cold, no feedback).

The specification is on pages 548-556 of the book: the region map is Fig. 1 of source situation 14 (page 549),
the assembly numbering Fig. 2 (page 550) and the cross sections page 554. The rod that is withdrawn is the
dashed region R of Fig. 1, fuel 2 with rod in the initial state and fuel 2 without rod once withdrawn.
The reference results are from solution 14-A1-1 (CUBBOX, pages 557-560): k = 0.99633 for the initial state,
1.01546 with the rod withdrawn, and the normalized power densities of Exhibit A. Solution 14-A1-2 (IQSBOX,
pages 577-580) gives 0.99631 and 1.01531.

NEM does not implement ZeroFlux, so the tests use finite differences on a 0.9375 cm mesh.
"""

from functools import lru_cache

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import Reflector, ZeroFlux
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable, Isotope
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core

AXIAL_BUCKLING = 1.0e-4
SUBCELLS = 16  # 15 cm / 16 = 0.9375 cm
REFERENCE_K = {"CUBBOX": 0.99633, "IQSBOX": 0.99631}
REFERENCE_K_ROD_WITHDRAWN = {"CUBBOX": 1.01546, "IQSBOX": 1.01531}
# Exhibit A of 14-A1-1: normalized power density of the fuel assemblies, rows y = 1..9 and columns x = 1..9
# in units of 15 cm from the symmetry lines; zeros are reflector
REFERENCE_POWER = np.array([
    [0.6214, 0.4438, 0.4134, 0.5107, 0.7939, 1.3984, 1.6725, 1.4769, 0.9059],
    [0.4438, 0.4010, 0.4092, 0.4931, 0.6708, 0.9416, 1.1512, 1.2703, 0.8538],
    [0.4134, 0.4092, 0.4275, 0.4958, 0.6209, 0.7794, 0.9608, 1.1655, 0.8145],
    [0.5107, 0.4931, 0.4958, 0.5565, 0.6816, 0.8400, 1.0173, 1.2168, 0.8427],
    [0.7939, 0.6708, 0.6209, 0.6816, 0.8655, 1.1570, 1.3450, 1.4185, 0.9260],
    [1.3984, 0.9416, 0.7794, 0.8400, 1.1570, 1.8739, 2.0743, 1.6882, 0.9580],
    [1.6725, 1.1512, 0.9608, 1.0173, 1.3450, 2.0743, 2.1888, 1.6176, 0.8403],
    [1.4769, 1.2703, 1.1655, 1.2168, 1.4185, 1.6882, 1.6176, 1.3323, 0.0],
    [0.9059, 0.8538, 0.8145, 0.8427, 0.9260, 0.9580, 0.8403, 0.0, 0.0],
])  # fmt: skip


def _material(name, diffusion, absorb, nusigmaf, scatter_12):
    # the axial leakage D * B^2 is added to the absorption
    diffusion, absorb, nusigmaf = np.array(diffusion), np.array(absorb), np.array(nusigmaf)
    data = dict(
        scatter=np.array([[0, 0], [scatter_12, 0]]),
        absorb=absorb + diffusion * AXIAL_BUCKLING,
        diffusion=diffusion,
    )
    if not nusigmaf.any():
        return Isotope(name, **data)
    return Fissionable(name, nusigmaf=nusigmaf, chi=np.array([1, 0]), **data)


# page 554: initial two group constants
MATERIALS = {
    1: _material("Fuel 1 with rod", [1.255, 0.211], [0.008252, 0.1003], [0.004602, 0.1091], 0.02533),
    2: _material("Fuel 1 without rod", [1.268, 0.1902], [0.007181, 0.07047], [0.004609, 0.08675], 0.02767),
    3: _material("Fuel 2 with rod", [1.259, 0.2091], [0.008002, 0.08344], [0.004663, 0.1021], 0.02617),
    4: _material("Fuel 2 without rod", [1.259, 0.2091], [0.008002, 0.073324], [0.004663, 0.1021], 0.02617),
    5: _material("Reflector", [1.257, 0.1592], [0.0006034, 0.01911], [0, 0], 0.04754),
}


def _regions(rod_withdrawn):
    """Fig. 1 on the 15 cm grid: regions[j, i] with i along x and j along y, both from the symmetry lines."""
    regions = np.full((11, 11), 5)
    regions[0:7, 0:7] = 1  # x, y < 105 cm
    for j, i in [(0, 0), (0, 5), (0, 6), (5, 0), (6, 0), (5, 5), (5, 6), (6, 5), (6, 6)]:
        regions[j, i] = 2
    regions[0:7, 7:9] = 3  # 105 < x < 135 cm, y < 105 cm
    regions[7:9, 0:7] = 3  # x < 105 cm, 105 < y < 135 cm
    regions[7, 7] = 4
    if rod_withdrawn:
        regions[5:7, 7:9] = 4  # R: 105 < x < 135 cm, 75 < y < 105 cm
    return regions


@lru_cache(maxsize=None)
def _solve(rod_withdrawn):
    composition = [MATERIALS[region] for region in _regions(rod_withdrawn).flatten()]
    geometry = Cartesian([np.full(11, 15.0)] * 2, [Reflector(), ZeroFlux(), Reflector(), ZeroFlux()])
    core = Core(composition, 2, geometry, CurrentCalculatorFD.from_isotopes(composition))
    k, flux = solve_k(core, FDSettings(split=geometry.uniform_split(SUBCELLS)))
    # solve_k returns the flux averaged over each 15 cm cell; nu is the same in every region, so the
    # power density is proportional to the production rate
    production = np.sum(np.reshape(flux, (geometry.cells, 2)) * [m.nusigmaf for m in composition], axis=1)
    return k, np.reshape(production, (11, 11))


def test_k_initial_state():
    k, _ = _solve(rod_withdrawn=False)
    assert k == pytest.approx(REFERENCE_K["CUBBOX"], abs=10e-5)


@pytest.mark.parametrize("solution", sorted(REFERENCE_K_ROD_WITHDRAWN))
def test_k_rod_withdrawn(solution):
    k, _ = _solve(rod_withdrawn=True)
    assert k == pytest.approx(REFERENCE_K_ROD_WITHDRAWN[solution], abs=20e-5)


def test_assembly_power_initial_state():
    _, production = _solve(rod_withdrawn=False)
    fuel = REFERENCE_POWER > 0
    power = production[:9, :9] / np.mean(production[:9, :9][fuel])
    assert np.max(np.abs(power[fuel] / REFERENCE_POWER[fuel] - 1)) < 0.025
