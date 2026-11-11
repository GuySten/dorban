"""
ANL benchmark book, problem 11: the three dimensional IAEA PWR, 2 groups. Finite differences on a 10 cm mesh
and NEM on 20 cm nodes. Reference k = 1.02909. Transcribed from docs/source/IAEA3D.ipynb.
"""

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.finite_differences.solve_finite_difference import solve_k_diffusion
from dorban.geometry.boundary_conditions import Boundary, Reflector, Void
from dorban.geometry.cartesian import Cartesian
from dorban.geometry.polybox import PolyBox
from dorban.materials import Fissionable, Isotope
from dorban.system import Core


def _compute():
    # notebook cell 4
    mat1 = Fissionable(
        "Fuel1",
        scatter=np.array([[0, 0], [0.02, 0]]),
        absorb=np.array([0.01, 0.08]),
        nusigmaf=np.array([0, 0.135]),
        chi=np.array([1, 0]),
        diffusion=np.array([1.5, 0.4]),
    )
    mat2 = Fissionable(
        "Fuel2",
        scatter=np.array([[0, 0], [0.02, 0]]),
        absorb=np.array([0.01, 0.085]),
        nusigmaf=np.array([0, 0.135]),
        chi=np.array([1, 0]),
        diffusion=np.array([1.5, 0.4]),
    )
    mat3 = Fissionable(
        "Fuel2+Rod",
        scatter=np.array([[0, 0], [0.02, 0]]),
        absorb=np.array([0.01, 0.13]),
        nusigmaf=np.array([0, 0.135]),
        chi=np.array([1, 0]),
        diffusion=np.array([1.5, 0.4]),
    )
    mat4 = Isotope(
        "Reflector", scatter=np.array([[0, 0], [0.04, 0]]), absorb=np.array([0, 0.01]), diffusion=np.array([2, 0.3])
    )
    mat5 = Isotope(
        "Reflector+Rod",
        scatter=np.array([[0, 0], [0.04, 0]]),
        absorb=np.array([0, 0.055]),
        diffusion=np.array([2, 0.3]),
    )

    # notebook cell 6
    lengths_x = np.array([10, 20, 20, 20, 20, 20, 20, 20, 20])
    lengths_z = np.array([20, 260, 80, 20])
    incore = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    black = [i for i in range(81) if incore[i] == 0]
    plane = Cartesian([lengths_x, lengths_x], [Reflector(), Void(), Reflector(), Void()], black)

    # notebook cell 8
    composition = [
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        3,
        2,
        2,
        2,
        3,
        2,
        2,
        1,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        4,
        4,
        3,
        2,
        2,
        2,
        3,
        1,
        1,
        4,
        2,
        2,
        2,
        2,
        1,
        1,
        4,
        4,
        2,
        2,
        1,
        1,
        1,
        4,
        4,
        1,
        1,
        1,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        3,
        2,
        2,
        2,
        3,
        2,
        2,
        1,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        4,
        2,
        2,
        3,
        2,
        2,
        2,
        1,
        1,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        4,
        4,
        3,
        2,
        2,
        2,
        3,
        1,
        1,
        4,
        2,
        2,
        2,
        2,
        1,
        1,
        4,
        4,
        2,
        2,
        1,
        1,
        1,
        4,
        4,
        1,
        1,
        1,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    names = {1: mat1, 2: mat2, 3: mat3, 4: mat4, 5: mat5}
    isotopes = [names[c] for c in composition]

    # notebook cell 10
    subcells = 2
    split = [np.array([1 * subcells] + [2 * subcells] * 8)] * 2
    refined_plane = plane.refine_mesh(split).to_polybox()

    # notebook cell 11
    z_neighbors = [(Void(), 1)] + [(a, a + 2) for a in range(subcells * 19 - 2)] + [(subcells * 19 - 2, Void())]
    neighbors = [
        [n if isinstance(n, Boundary) else n + h * refined_plane.cells for n in neigh]
        + list(map(lambda n: n if isinstance(n, Boundary) else n * refined_plane.cells + cell, pair))
        for h, pair in enumerate(z_neighbors)
        for cell, neigh in enumerate(refined_plane.neighbors)
    ]
    z_lengths = [20 / subcells] * 19 * subcells
    lengths = [list(plane_length) + [z_length] for z_length in z_lengths for plane_length in refined_plane.lengths]
    refined_geo = PolyBox(3, neighbors, lengths)

    # notebook cell 12
    incore = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ] * 4
    black = [i for i in range(81 * 4) if incore[i] == 0]
    original_geometry = Cartesian(
        [lengths_x, lengths_x, lengths_z], [Reflector(), Void(), Reflector(), Void(), Void(), Void()], black
    )

    # notebook cell 14
    system = Core(isotopes, 2, original_geometry, CurrentCalculatorFD.from_isotopes(isotopes))

    # notebook cell 15
    spliting = [[subcells] + [2 * subcells] * 8] * 2 + [[subcells, 13 * subcells, 4 * subcells, subcells]]
    spliting

    # notebook cell 16
    calc = system.current_calc.mesh_refine(system.geometry, spliting, refined_geo)
    assemblies = system.geometry.array_refine(np.arange(system.geometry.cells), spliting).astype(int)
    isotopes = [system.isotopes[assembly] for assembly in assemblies]
    core = Core(isotopes, 2, refined_geo, calc)

    # notebook cell 18
    k, flux = solve_k_diffusion(core, rtol_vector=1e-8, lin_rtol=1e-10)

    # notebook cell 21
    thermal = flux[1::2]
    fission_source = thermal * np.array([iso.nusigmaf[1] for iso in core.isotopes])
    fuel = [i for i, iso in enumerate(core.isotopes) if iso.isfissile]
    core_source = fission_source[fuel]

    # notebook cell 23
    np.max(core_source) / np.mean(core_source)

    # notebook cell 25
    intermidate_split = [np.array([1] + [1] * 8)] * 2 + [np.array([1, 13, 4, 1])]
    intermidte_geo = original_geometry.refine_mesh(intermidate_split)
    rest_split = [[subcells] + [2 * subcells] * 8, [subcells] + [2 * subcells] * 8, [subcells] * 19]
    fission_source = np.bincount(
        intermidte_geo.array_refine(np.arange(intermidte_geo.cells), rest_split).astype(int), fission_source
    )
    rsource = np.reshape(fission_source, (len(fission_source) // (19), 19), order="F")

    # notebook cell 26
    radial = np.mean(rsource, axis=1)
    radial = radial / [intermidte_geo.volumes[i] for i in range(69)]
    radial = radial[radial.nonzero()]
    radial / np.mean(radial)

    # notebook cell 28
    axial = np.mean(rsource, axis=0)
    17.026 * axial / np.sum(axial)

    # notebook cell 30
    subcells = 1
    spliting = [[subcells] + [2 * subcells] * 8] * 2 + [[subcells, 13 * subcells, 4 * subcells, subcells]]
    spliting

    # notebook cell 31
    from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
    from dorban.nem.solve_nem import NEMSettings
    from dorban.solve_equation import solve_k
    from dorban.system import mesh_refinement

    nem_system = Core(system.isotopes, system.E, system.geometry, CMFDCurrentCalculator(system.current_calc.dc, 3))
    nem_system = mesh_refinement(nem_system, spliting)
    nem_k, nem_flux = solve_k(nem_system, NEMSettings(split=None))

    # notebook cell 34
    nem_flux = nem_flux / np.sum(nem_flux)

    # notebook cell 35
    nem_thermal = nem_flux[1::2]
    nem_fission_source = nem_thermal * np.array([iso.nusigmaf[1] for iso in nem_system.isotopes])
    fuel = [i for i, iso in enumerate(nem_system.isotopes) if iso.isfissile]
    nem_core_source = nem_fission_source[fuel]

    # notebook cell 36
    intermidate_split = spliting
    intermidte_geo = original_geometry.refine_mesh(intermidate_split)
    rest_split = intermidte_geo.uniform_split(1)
    nem_fission_source = np.bincount(
        intermidte_geo.array_refine(np.arange(intermidte_geo.cells), rest_split).astype(int), nem_fission_source
    )
    nem_rsource = np.reshape(nem_fission_source, (len(nem_fission_source) // (19), 19), order="F")

    # notebook cell 37
    nem_axial = np.mean(nem_rsource, axis=0)
    17.026 * nem_axial / np.sum(nem_axial)
    return locals()


@pytest.fixture(scope="module")
def solution():
    return _compute()


def test_k(solution):
    assert solution["k"] == pytest.approx(1.02909, abs=1e-3)


def test_nem_k_agrees_with_finite_differences(solution):
    assert solution["nem_k"] == pytest.approx(solution["k"], abs=1e-3)


def test_nem_axial_power_agrees_with_finite_differences(solution):
    nem_axial, axial = solution["nem_axial"], solution["axial"]
    nem_normalized_axial = (nem_axial / np.sum(nem_axial))[1:-1]
    normalized_axial = (axial / np.sum(axial))[1:-1]
    assert np.max(np.abs((nem_normalized_axial - normalized_axial) / normalized_axial)) < 0.2
