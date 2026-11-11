"""
ANL benchmark book (ANL-7416 Supplement 2, 1977), problem 11-A2: the two dimensional IAEA PWR, the midplane
of problem 11-A1, 2 groups with axial buckling 0.8e-4 for all regions and groups. Quarter core, 170 x 170 cm.

The specification is on pages 437-439 of the book and the core map is Fig. 1 of source situation 11 (page 277).
The reference results are from solution 11-A2-1 (pages 440-445): the extrapolated mesh centered finite difference
solution, k = 1.02959, its zone average fluxes (Tables 2 and 3), and the mesh centered finite difference
k at each mesh spacing (Table 1).

The book's finite difference form of the vacuum boundary, d(phi)/dn = -0.4692 phi / D, is ExtrapolationLength,
used by both the finite difference and the NEM tests.
"""

from functools import lru_cache

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import ExtrapolationLength, Reflector, Void
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable, Isotope
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.solve_nem import NEMSettings
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core

AXIAL_BUCKLING = 0.8e-4
REFERENCE_K = 1.02959
# Table 1, mesh centered VENTURE: mesh spacing in cm -> k
VENTURE_K = {10.0: 1.02965, 5.0: 1.02924, 2.5: 1.02944, 1.25: 1.02954}


def _material(name, absorb_1, absorb_2, nusigmaf_2, diffusion_1, diffusion_2, scatter_12):
    # the axial leakage D * Bz^2 is added to the absorption
    data = dict(
        scatter=np.array([[0, 0], [scatter_12, 0]]),
        absorb=np.array([absorb_1 + diffusion_1 * AXIAL_BUCKLING, absorb_2 + diffusion_2 * AXIAL_BUCKLING]),
        diffusion=np.array([diffusion_1, diffusion_2]),
    )
    if nusigmaf_2 == 0:
        return Isotope(name, **data)
    return Fissionable(name, nusigmaf=np.array([0, nusigmaf_2]), chi=np.array([1, 0]), **data)


# page 437: two group constants
MATERIALS = {
    1: _material("Fuel 1", 0.01, 0.08, 0.135, 1.5, 0.4, 0.02),
    2: _material("Fuel 2", 0.01, 0.085, 0.135, 1.5, 0.4, 0.02),
    3: _material("Fuel 2 + Rod", 0.01, 0.13, 0.135, 1.5, 0.4, 0.02),
    4: _material("Reflector", 0, 0.01, 0, 2.0, 0.3, 0.04),
}
# Fig. 1 at the midplane: quarter core rows from y = 0 outwards, columns from x = 0 outwards; the first row and
# column are 10 cm wide (half assemblies on the symmetry lines), the others 20 cm. Cells beyond each row are outside
# the core. The rod of assembly 18 is only inserted above z = 280 cm, so it is fuel 2 at the midplane.
ROWS = [
    [3, 2, 2, 2, 3, 2, 2, 1, 4],
    [2, 2, 2, 2, 2, 2, 2, 1, 4],
    [2, 2, 2, 2, 2, 2, 1, 1, 4],
    [2, 2, 2, 2, 2, 2, 1, 4, 4],
    [3, 2, 2, 2, 3, 1, 1, 4],
    [2, 2, 2, 2, 1, 1, 4, 4],
    [2, 2, 1, 1, 1, 4, 4],
    [1, 1, 1, 4, 4, 4],
    [4, 4, 4, 4],
]
WIDTHS = np.array([10.0] + [20.0] * 8)
# Fig. 1, lower octant: zone numbers 1-38 run along the rows, row j starting on the diagonal
ZONES_PER_ROW = [9, 8, 7, 6, 4, 3, 1]
# Tables 2 and 3, extrapolated column: zone average fast and thermal flux, zones 1-38
REFERENCE_FAST = [
    18.34, 23.67, 25.92, 21.86, 15.01, 16.92, 16.62, 11.49, 1.906, 25.61,
    26.39, 23.46, 19.34, 18.49, 16.86, 11.22, 1.821, 26.18, 23.99, 21.03,
    19.03, 16.43, 9.413, 1.380, 21.28, 17.45, 16.10, 12.76, 3.361, 0.4033,
    11.52, 11.73, 8.157, 1.369, 7.918, 2.255, 0.3243, 0.3634,
]  # fmt: skip
REFERENCE_THERMAL = [
    3.120, 5.482, 6.085, 5.068, 2.552, 3.914, 3.911, 3.160, 4.514, 6.006,
    6.193, 5.503, 4.477, 4.336, 3.977, 3.079, 4.293, 6.149, 5.630, 4.935,
    4.480, 4.080, 2.896, 3.307, 4.992, 4.047, 3.794, 3.540, 7.024, 1.590,
    1.969, 2.869, 2.498, 3.414, 2.448, 4.707, 1.266, 1.448,
]  # fmt: skip


def _zones():
    """(column, row) of each zone of Fig. 1, in zone number order."""
    return [(i, j) for j, count in enumerate(ZONES_PER_ROW) for i in range(j, j + count)]


def _zone_weights():
    """Area each octant zone stands for in the quarter core: zones off the diagonal appear twice."""
    weights = []
    for i, j in _zones():
        area = (10 if i == 0 else 20) * (10 if j == 0 else 20)
        weights.append(area if i == j else 2 * area)
    return np.array(weights)


FUEL_ZONES = np.array([ROWS[j][i] != 4 for i, j in _zones()])


def _normalize(fast, thermal):
    """Scale the zone fluxes so that the average power density over the fuel is 1."""
    power = MATERIALS[1].nusigmaf[1] * thermal
    weights = _zone_weights()
    mean_power = np.sum((power * weights)[FUEL_ZONES]) / np.sum(weights[FUEL_ZONES])
    return fast / mean_power, thermal / mean_power


def _geometry(boundary):
    black = [j * len(WIDTHS) + i for j, row in enumerate(ROWS) for i in range(len(row), len(WIDTHS))]
    geometry = Cartesian([WIDTHS, WIDTHS], [Reflector(), boundary(), Reflector(), boundary()], black)
    # Cartesian puts Void next to the cells outside the core; use the same condition on every external face
    geometry.neighbors = [[boundary() if isinstance(n, Void) else n for n in neigh] for neigh in geometry.neighbors]
    return geometry


def _composition():
    return [MATERIALS[m] for row in ROWS for m in row]


def _zone_fluxes(geometry, flux):
    """
    Average fast and thermal flux in each zone of Fig. 1, normalized to an average fuel power density of 1.
    solve_k returns the flux averaged over the cells of the unrefined geometry, here the assemblies.
    """
    assembly_flux = np.reshape(flux, (geometry.cells, 2))
    cells = [(i, j) for j, row in enumerate(ROWS) for i in range(len(row))]
    zone_flux = np.array([assembly_flux[cells.index(zone)] for zone in _zones()])
    return _normalize(zone_flux[:, 0], zone_flux[:, 1])


def _reference_zone_fluxes():
    return _normalize(np.array(REFERENCE_FAST), np.array(REFERENCE_THERMAL))


def _relative_errors(solution):
    fast, thermal = solution["zone_fluxes"]
    reference_fast, reference_thermal = _reference_zone_fluxes()
    power_error = np.abs(thermal / reference_thermal - 1)[FUEL_ZONES]
    flux_error = np.max([np.abs(fast / reference_fast - 1), np.abs(thermal / reference_thermal - 1)])
    return np.max(power_error), flux_error


@lru_cache(maxsize=None)
def _solve_finite_differences(mesh):
    geometry = _geometry(ExtrapolationLength)
    composition = _composition()
    core = Core(composition, 2, geometry, CurrentCalculatorFD.from_isotopes(composition))
    subcells = int(round(10 / mesh))
    split = [np.array([subcells] + [2 * subcells] * 8)] * 2
    k, flux = solve_k(core, FDSettings(split=split))
    return {"k": k, "zone_fluxes": _zone_fluxes(geometry, flux)}


def _solve_nem(nodes_per_assembly):
    geometry = _geometry(ExtrapolationLength)
    composition = _composition()
    core = Core(composition, 2, geometry, CMFDCurrentCalculator.from_isotopes(composition, 2))
    split = geometry.uniform_split(nodes_per_assembly) if nodes_per_assembly > 1 else None
    k, flux = solve_k(core, NEMSettings(split=split))
    return {"k": k, "zone_fluxes": _zone_fluxes(geometry, flux)}


@pytest.fixture(scope="module", params=sorted(VENTURE_K, reverse=True), ids=lambda mesh: f"{mesh:g}cm")
def finite_differences(request):
    return request.param, _solve_finite_differences(request.param)


@pytest.fixture(scope="module")
def finite_differences_finest_mesh():
    return _solve_finite_differences(min(VENTURE_K))


@pytest.fixture(scope="module")
def nem_assembly_nodes():
    return _solve_nem(1)


@pytest.fixture(scope="module")
def nem_half_assembly_nodes():
    return _solve_nem(2)


def test_finite_differences_k_agrees_with_mesh_centered_solution_at_the_same_mesh(finite_differences):
    mesh, solution = finite_differences
    assert solution["k"] == pytest.approx(VENTURE_K[mesh], abs=2e-5)


def test_finite_differences_on_the_finest_mesh(finite_differences_finest_mesh):
    solution = finite_differences_finest_mesh
    power_error, _ = _relative_errors(solution)
    assert solution["k"] == pytest.approx(REFERENCE_K, abs=10e-5)
    assert power_error < 0.025


def test_nem_one_node_per_assembly(nem_assembly_nodes):
    power_error, _ = _relative_errors(nem_assembly_nodes)
    assert nem_assembly_nodes["k"] == pytest.approx(REFERENCE_K, abs=10e-5)
    assert power_error < 0.03


def test_nem_four_nodes_per_assembly(nem_half_assembly_nodes):
    power_error, flux_error = _relative_errors(nem_half_assembly_nodes)
    assert nem_half_assembly_nodes["k"] == pytest.approx(REFERENCE_K, abs=5e-5)
    assert power_error < 0.005
    assert flux_error < 0.025
