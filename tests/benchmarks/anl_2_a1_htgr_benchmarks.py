"""
ANL benchmark book (ANL-7416, 1968), problem 2-A1: an (x, y) HTGR, 7 groups with up-scattering among the four
thermal groups. A core of radius 300 cm (region 1) with seven control rod channels of radius 50 cm (region 3) at
the center and on a ring of radius 165 cm every 60 degrees, in a reflector of outer radius 360 cm (region 2).
Quarter core with a vacuum boundary on the outer surface.

The specification is on pages 24-29 of the book: Fig. 1 of source situation 2 (page 24) and the cross sections
of pages 26-28. The vacuum boundary condition is d(phi)/dn = -0.4695 phi / D. The scattering tables are printed one
row per group of origin.

The reference is solution 2-A1-1 (GAMBLE-5, pages 30-33), five-point box differences with the mesh points on the
corners of the mesh intervals. The circles are drawn as staircases of mesh intervals in Figs. 3 and 4 (page 33):
an interval belongs to a region if its center does, and the reactor ends with the last interval whose center is
inside 360 cm. That map is the problem solved by the book, so the tests use the 5 cm map of Fig. 4 and refine it by
splitting its cells, which keeps the geometry. The reference is the fine mesh result on that map, k = 1.006418, and
its region average fluxes (Exhibit B).

The book gives no estimate of the error of GAMBLE's 5 cm mesh. The extrapolated dorban solution on the same map is
36 pcm above it. The fast flux of the reflector is 5% below GAMBLE: GAMBLE's own result for it changes by 13%
between its 10 cm and 5 cm meshes (Exhibit B), while dorban's converged values on the two maps agree within 2%.
"""

from functools import lru_cache

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import ExtrapolationLength, Reflector, Void
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable, Isotope
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core

REFERENCE_K = 1.006418
MESH = 5.0  # Fig. 4
# Exhibit B, fine mesh: average flux of each group (rows) in regions 1, 2 and 3 (columns)
REFERENCE_REGION_FLUX = np.array([
    [8.6048, 0.9328, 7.6753],
    [18.4850, 3.1899, 16.7050],
    [2.9280, 0.6966, 2.5876],
    [2.9357, 0.9921, 2.4655],
    [5.3363, 7.4586, 4.1608],
    [1.4063, 3.1975, 1.0609],
    [0.3746, 1.0808, 0.2778],
])  # fmt: skip
CORE_RADIUS, REACTOR_RADIUS, ROD_RADIUS, ROD_RING = 300.0, 360.0, 50.0, 165.0
# the rods whose centers are in the quarter core x, y >= 0
RODS = [(0.0, 0.0), (ROD_RING, 0.0), (ROD_RING * np.cos(np.pi / 3), ROD_RING * np.sin(np.pi / 3))]
CHI = np.array([0.9675, 0.0325, 0, 0, 0, 0, 0])


class Vacuum(ExtrapolationLength):
    constant = 0.4695


def _scatter(rows):
    """rows[from] = {to: cross section}, groups numbered from 1; dorban's scatter[to, from]."""
    scatter = np.zeros((7, 7))
    for origin, targets in rows.items():
        for target, value in targets.items():
            scatter[target - 1, origin - 1] = value
    return scatter


def _fuel_scatter(scatter_12, scatter_23):
    return {
        1: {2: scatter_12},
        2: {3: scatter_23},
        3: {4: 2.140e-2},
        4: {3: 7.827e-4, 5: 3.643e-2, 6: 1.821e-4, 7: 1.890e-4},
        5: {4: 1.049e-2, 6: 1.856e-2, 7: 2.305e-3},
        6: {4: 1.234e-4, 5: 5.946e-2, 7: 1.490e-2},
        7: {4: 4.051e-5, 5: 2.257e-2, 6: 4.869e-2},
    }


def _material(name, diffusion, capture, fission, nusigmaf, scatter):
    diffusion, capture, fission, nusigmaf = map(np.array, (diffusion, capture, fission, nusigmaf))
    data = dict(scatter=_scatter(scatter), absorb=capture + fission, diffusion=diffusion)
    if not nusigmaf.any():
        return Isotope(name, **data)
    return Fissionable(name, nusigmaf=nusigmaf, chi=CHI, **data)


FISSION = [4.286e-5, 2.553e-4, 7.504e-4, 1.277e-3, 4.127e-3, 7.115e-3, 1.395e-2]
NUSIGMAF = [1.074e-4, 6.207e-4, 1.823e-3, 3.104e-3, 1.003e-2, 1.729e-2, 3.390e-2]
# pages 26-28: diffusion coefficient, capture, fission and production cross sections and scattering matrix
MATERIALS = {
    1: _material(
        "Region 1",
        [2.384, 1.221, 1.227, 1.225, 1.145, 0.9930, 0.9175],
        [5.328e-5, 1.726e-3, 9.050e-4, 8.905e-4, 2.453e-3, 4.023e-3, 7.479e-3],
        FISSION,
        NUSIGMAF,
        _fuel_scatter(1.198e-2, 3.546e-3),
    ),
    2: _material(
        "Region 2",
        [1.839, 0.9480, 0.9329, 0.9432, 0.8932, 0.7971, 0.7730],
        [5.621e-8, 2.497e-6, 2.028e-5, 6.043e-5, 1.172e-4, 1.910e-4, 3.419e-4],
        [0] * 7,
        [0] * 7,
        {
            1: {2: 1.562e-2},
            2: {3: 6.369e-3},
            3: {4: 2.986e-2},
            4: {3: 4.936e-4, 5: 7.600e-2, 6: 3.616e-4, 7: 3.757e-5},
            5: {4: 7.031e-3, 6: 2.893e-2, 7: 3.421e-3},
            6: {4: 7.066e-5, 5: 6.481e-2, 7: 1.899e-2},
            7: {4: 2.137e-5, 5: 2.209e-2, 6: 5.513e-2},
        },
    ),
    3: _material(
        "Region 3",
        [2.381, 1.217, 1.222, 1.221, 1.142, 0.9909, 0.9158],
        [6.942e-5, 2.184e-3, 1.916e-3, 1.800e-3, 3.193e-3, 4.737e-3, 8.224e-3],
        FISSION,
        NUSIGMAF,
        _fuel_scatter(1.200e-2, 3.551e-3),
    ),
}


def _regions():
    """Fig. 4: the region of each 5 cm mesh interval of the quarter core, 0 outside the reactor."""
    cells = int(round(REACTOR_RADIUS / MESH))
    centers = (np.arange(cells) + 0.5) * MESH
    x, y = np.meshgrid(centers, centers)
    radius = np.hypot(x, y)
    regions = np.where(radius < CORE_RADIUS, 1, np.where(radius < REACTOR_RADIUS, 2, 0))
    for x0, y0 in RODS:
        regions[np.hypot(x - x0, y - y0) < ROD_RADIUS] = 3
    return regions.flatten()


def _geometry(regions):
    cells = int(round(REACTOR_RADIUS / MESH))
    outside = [int(i) for i in np.flatnonzero(regions == 0)]
    geometry = Cartesian([np.full(cells, MESH)] * 2, [Reflector(), Vacuum(), Reflector(), Vacuum()], outside)
    # Cartesian puts Void next to the cells outside the reactor; use the book's vacuum condition there too
    geometry.neighbors = [[Vacuum() if isinstance(n, Void) else n for n in neigh] for neigh in geometry.neighbors]
    return geometry


@lru_cache(maxsize=None)
def _solve(split):
    """Each 5 cm interval of Fig. 4 split into split x split cells; returns k and the region average fluxes."""
    regions = _regions()
    regions = regions[regions > 0]
    geometry = _geometry(_regions())
    composition = [MATERIALS[region] for region in regions]
    core = Core(composition, 7, geometry, CurrentCalculatorFD.from_isotopes(composition))
    k, flux = solve_k(core, FDSettings(split=geometry.uniform_split(split)))
    # solve_k returns the flux averaged over each 5 cm interval
    flux = np.reshape(flux, (-1, 7))
    volumes = geometry.volumes
    region_flux = np.array([
        np.sum(flux[regions == region] * volumes[regions == region, None], axis=0) / np.sum(volumes[regions == region])
        for region in (1, 2, 3)
    ]).T  # fmt: skip
    return k, region_flux


def test_extrapolated_k():
    k_coarse, k_fine = _solve(1)[0], _solve(2)[0]
    assert (4 * k_fine - k_coarse) / 3 == pytest.approx(REFERENCE_K, abs=55e-5)


def test_region_fluxes():
    _, region_flux = _solve(2)
    # the normalization of Exhibit B is not given: scale to the same total
    region_flux *= np.sum(REFERENCE_REGION_FLUX) / np.sum(region_flux)
    error = np.abs(region_flux / REFERENCE_REGION_FLUX - 1)
    assert np.max(error[:, [0, 2]]) < 0.012  # core and rods
    assert np.max(error[:, 1]) < 0.08  # reflector
