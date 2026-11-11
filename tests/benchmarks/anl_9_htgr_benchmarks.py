"""
ANL benchmark book, problem 9-1: two dimensional hexagonal HTGR, 4 groups, solved with finite differences on
a triangular mesh. Reference k = 1.11835. Transcribed from docs/source/HTGR.ipynb.
"""

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import BluePortal, Boundary, OrangePortal, ZeroFlux, glue_orange_blue
from dorban.geometry.triangular_tessellation import TriangularTessellation, hex2triangles
from dorban.materials import Fissionable, Isotope
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core, mesh_refinement
from dorban.utils import insert_black_absorber, tensor_product, triangular


def _compute():
    # notebook cell 5
    A = Fissionable(
        "A",
        np.array([[0, 0, 0, 0], [1.09e-2, 0, 0, 0], [0, 3.28e-3, 0, 0], [0, 0, 2.04e-2, 0]]),
        np.array([2.205e-4, 4.699e-3, 9.528e-3, 1.09e-2]),
        np.array([5.58e-5, 3.61e-4, 9.88e-4, 4.76e-3]),
        np.array([0.9675, 0.0325, 0, 0]),
        fission=np.array([2.25e-5, 1.49e-4, 4.08e-4, 1.97e-3]),
    )
    B = Fissionable(
        "B",
        np.array([[0, 0, 0, 0], [1.09e-2, 0, 0, 0], [0, 3.28e-3, 0, 0], [0, 0, 2.04e-2, 0]]),
        np.array([5.86e-5, 9.18e-4, 8.36e-4, 2.84e-3]),
        np.array([5.58e-5, 3.61e-4, 9.88e-4, 4.76e-3]),
        np.array([0.9675, 0.0325, 0, 0]),
        fission=np.array([2.25e-5, 1.49e-4, 4.08e-4, 1.97e-3]),
    )
    C = Fissionable(
        "C",
        np.array([[0, 0, 0, 0], [1.23e-2, 0, 0, 0], [0, 3.67e-3, 0, 0], [0, 0, 2.28e-2, 0]]),
        np.array([9.77e-5, 1.622e-3, 1.56e-3, 5.35e-3]),
        np.array([9.83e-5, 6.36e-4, 1.74e-3, 8.39e-3]),
        np.array([0.9675, 0.0325, 0, 0]),
        fission=np.array([3.95e-5, 2.62e-4, 7.18e-4, 3.46e-3]),
    )
    D = Fissionable(
        "D",
        np.array([[0, 0, 0, 0], [1.23e-2, 0, 0, 0], [0, 3.67e-3, 0, 0], [0, 0, 2.28e-2, 0]]),
        np.array([9.84e-5, 1.666e-3, 1.424e-3, 4.99e-3]),
        np.array([9.31e-5, 5.73e-4, 1.57e-3, 7.56e-3]),
        np.array([0.9675, 0.0325, 0, 0]),
        fission=np.array([3.76e-5, 2.36e-4, 6.47e-4, 3.12e-3]),
    )
    E = Fissionable(
        "E",
        np.array([[0, 0, 0, 0], [1.23e-2, 0, 0, 0], [0, 3.67e-3, 0, 0], [0, 0, 2.28e-2, 0]]),
        np.array([9.86e-5, 1.683e-3, 1.341e-3, 4.77e-3]),
        np.array([8.97e-5, 5.34e-4, 1.46e-3, 7.05e-3]),
        np.array([0.9675, 0.0325, 0, 0]),
        fission=np.array([3.63e-5, 2.2e-4, 6.03e-4, 2.91e-3]),
    )
    F = Fissionable(
        "F",
        np.array([[0, 0, 0, 0], [1.23e-2, 0, 0, 0], [0, 3.67e-3, 0, 0], [0, 0, 2.28e-2, 0]]),
        np.array([9.91e-5, 1.704e-3, 1.258e-3, 4.55e-3]),
        np.array([8.65e-5, 4.95e-4, 1.36e-3, 6.54e-3]),
        np.array([0.9675, 0.0325, 0, 0]),
        fission=np.array([3.51e-5, 2.04e-4, 5.59e-4, 2.7e-3]),
    )
    G = Isotope(
        "G",
        np.array([[0, 0, 0, 0], [1.77e-2, 0, 0, 0], [0, 5.33e-3, 0, 0], [0, 0, 3.31e-2, 0]]),
        np.array([1.39e-5, 2.18e-6, 1.97e-5, 1.06e-4]),
    )
    AB_diffusion = [[2.65, 1.37, 1.34, 1.31]]
    CDEF_diffusion = [[2.35, 1.21, 1.19, 1.16]]
    G_diffusion = [[1.64, 8.5e-1, 8.32e-1, 8.21e-1]]
    isotopes = {}
    A_cells = [11]
    B_Cells = [5, 19, 21, 31, 42]
    C_cells = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 32, 33, 34]
    D_cells = [30, 35]
    E_cells = [29, 40, 41]
    F_cells = [28, 39, 43]
    G_cells = [36, 37, 38] + list(range(44, 71))
    keys = [A, B, C, D, E, F, G]
    locations = [B_Cells, C_cells, D_cells, E_cells, F_cells, G_cells]
    diffusions = AB_diffusion * 2 + CDEF_diffusion * 4 + G_diffusion
    diffusion_coefficients = np.zeros(71 * 6 * 4 + 4)
    isotopes[A] = np.hstack(([True], np.repeat([i in A_cells for i in range(71)], 6)))
    for j, isotope in enumerate(keys[1:]):
        isotopes[isotope] = np.hstack(([False], np.repeat([i in locations[j] for i in range(71)], 6)))
    for i, isotope in enumerate(keys):
        diffusion_coefficients += tensor_product(diffusions[i], isotopes[isotope])
    cs = np.zeros(71 * 6 + 1, dtype=object)
    for isotope, den in isotopes.items():
        indices = [i for i, d in enumerate(den) if d]
        cs[indices] = isotope

    # notebook cell 7
    def gen_neigh():
        yield [-2 / 3, BluePortal(), BluePortal(), 1, 2, OrangePortal()]
        for i in range(1, 13):
            cell = triangular(i + 1)
            yield [cell - i, BluePortal(), BluePortal(), cell + i + 1, cell + i + 2, cell + 1]
            for j in range(i - 1):
                cell += 1
                yield [cell - i, cell - i - 1, cell - 1, cell + i + 1, cell + i + 2, cell + 1]
            cell += 1
            yield [OrangePortal(), cell - i - 1, cell - 1, cell + i + 1, cell + i + 2, OrangePortal()]

    black = [55, 56, 66, 67, 68, 69, 70, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 88, 89, 90] + list(range(91, 105))
    nonblack = [cell for cell in range(91) if cell not in black]
    neighbors = hex2triangles(
        np.array([[insert_black_absorber(cell, black, ZeroFlux()) for cell in nlist] for nlist in gen_neigh()])[
            nonblack
        ]
    )
    plane_neighbors = [[OrangePortal(), 1, BluePortal()]] + [
        [cell if isinstance(cell, Boundary) else int(cell + 1) for cell in n] for n in neighbors
    ]
    plane_neighbors[384] = [ZeroFlux(), 383, 379]

    # notebook cell 9
    system = Core(
        cs,
        4,
        TriangularTessellation(71 * 6 + 1, 20.9, plane_neighbors),
        CurrentCalculatorFD(np.reshape(diffusion_coefficients, (4, 71 * 6 + 1), order="F").transpose()),
    )
    split = 7
    system = mesh_refinement(system, (split,))
    geo = system.geometry
    neighbors = np.array(geo.neighbors)
    blue = [i for i in range(geo.tri) if np.any([isinstance(n, BluePortal) for n in neighbors[i]])]
    orange = [i for i in range(geo.tri) if np.any([isinstance(n, OrangePortal) for n in neighbors[i]])]
    glue_orange_blue(geo, orange, blue)
    system = Core(system.isotopes, 4, geo, system.current_calc)

    # notebook cell 11
    k, flux = solve_k(system, settings=FDSettings(split=system.geometry.uniform_split(1)))
    return locals()


@pytest.fixture(scope="module")
def solution():
    return _compute()


def test_k(solution):
    assert solution["k"] == pytest.approx(1.11835, abs=1e-4)
