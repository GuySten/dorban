"""
ANL benchmark book (ANL-7416), problem 11: the three dimensional IAEA PWR with 2 energy groups, a quarter core of
20 cm assemblies, 380 cm high. Solved with finite differences on a mesh refined by hand (5 cm in the plane and 10 cm
axially) and with NEM on 10 cm x 10 cm x 20 cm nodes. Reference k = 1.02909.

This is the specification shared by the documentation notebook IAEA3D.ipynb and the benchmark tests, together with the
post-processing both use: the thermal fission source and its sums by plane cell and axial level.
"""

import numpy as np

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import Boundary, Reflector, Void
from dorban.geometry.cartesian import Cartesian
from dorban.geometry.polybox import PolyBox
from dorban.materials import Fissionable, Isotope
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.system import Core, mesh_refinement

REFERENCE_K = 1.02909

# fmt: off
# 1 inside the core, 0 outside: one row of the 9 x 9 quarter core map per line
INCORE = [
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0, 0,
]

# the material of each cell inside the core, by axial level and then by row of the map
COMPOSITION = [
    # axial level 0, 20 cm
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4,
    4, 4, 4, 4,
    # axial level 1, 260 cm
    3, 2, 2, 2, 3, 2, 2, 1, 4,
    2, 2, 2, 2, 2, 2, 2, 1, 4,
    2, 2, 2, 2, 2, 2, 1, 1, 4,
    2, 2, 2, 2, 2, 2, 1, 4, 4,
    3, 2, 2, 2, 3, 1, 1, 4,
    2, 2, 2, 2, 1, 1, 4, 4,
    2, 2, 1, 1, 1, 4, 4,
    1, 1, 1, 4, 4, 4,
    4, 4, 4, 4,
    # axial level 2, 80 cm
    3, 2, 2, 2, 3, 2, 2, 1, 4,
    2, 2, 2, 2, 2, 2, 2, 1, 4,
    2, 2, 3, 2, 2, 2, 1, 1, 4,
    2, 2, 2, 2, 2, 2, 1, 4, 4,
    3, 2, 2, 2, 3, 1, 1, 4,
    2, 2, 2, 2, 1, 1, 4, 4,
    2, 2, 1, 1, 1, 4, 4,
    1, 1, 1, 4, 4, 4,
    4, 4, 4, 4,
    # axial level 3, 20 cm
    5, 4, 4, 4, 5, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 5, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 4, 4, 4, 5, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4,
    4, 4, 4, 4,
]
# fmt: on


def materials():
    """The five materials, keyed by their number in COMPOSITION."""
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
    return {1: mat1, 2: mat2, 3: mat3, 4: mat4, 5: mat5}


def assembly_isotopes():
    """The isotope of each cell of the original geometry inside the core."""
    names = materials()
    return [names[c] for c in COMPOSITION]


def plane() -> Cartesian:
    """The quarter core plane: a half assembly and 8 assemblies along x and y, reflective on the symmetry planes."""
    lengths_x = np.array([10, 20, 20, 20, 20, 20, 20, 20, 20])
    black = [i for i in range(81) if INCORE[i] == 0]
    return Cartesian([lengths_x, lengths_x], [Reflector(), Void(), Reflector(), Void()], black)


def original_geometry() -> Cartesian:
    """The core geometry: the plane times the 4 axial regions (20, 260, 80 and 20 cm), a cell per assembly and region."""
    lengths_x = np.array([10, 20, 20, 20, 20, 20, 20, 20, 20])
    lengths_z = np.array([20, 260, 80, 20])
    incore = INCORE * 4
    black = [i for i in range(81 * 4) if incore[i] == 0]
    return Cartesian(
        [lengths_x, lengths_x, lengths_z], [Reflector(), Void(), Reflector(), Void(), Void(), Void()], black
    )


def splitting(subcells: int):
    """
    The split of the original geometry: each assembly into 2 * subcells cells along x and y (the half assemblies into
    subcells) and each axial region into subcells cells per 20 cm.
    """
    return [[subcells] + [2 * subcells] * 8] * 2 + [[subcells, 13 * subcells, 4 * subcells, subcells]]


def refined_geometry(subcells: int = 2) -> PolyBox:
    """
    The original geometry refined by splitting(subcells), built by hand and not with the refinement tools in order to
    save memory and run time: the refined plane stacked 19 * subcells times.
    """
    split = [np.array([1 * subcells] + [2 * subcells] * 8)] * 2
    refined_plane = plane().refine_mesh(split).to_polybox()
    z_neighbors = [(Void(), 1)] + [(a, a + 2) for a in range(subcells * 19 - 2)] + [(subcells * 19 - 2, Void())]
    neighbors = [
        [n if isinstance(n, Boundary) else n + h * refined_plane.cells for n in neigh]
        + list(map(lambda n: n if isinstance(n, Boundary) else n * refined_plane.cells + cell, pair))
        for h, pair in enumerate(z_neighbors)
        for cell, neigh in enumerate(refined_plane.neighbors)
    ]
    z_lengths = [20 / subcells] * 19 * subcells
    lengths = [list(plane_length) + [z_length] for z_length in z_lengths for plane_length in refined_plane.lengths]
    return PolyBox(3, neighbors, lengths)


def assembly_core() -> Core:
    """The core on the original geometry, with the finite difference current calculator."""
    isotopes = assembly_isotopes()
    return Core(isotopes, 2, original_geometry(), CurrentCalculatorFD.from_isotopes(isotopes))


def fd_core(subcells: int = 2) -> Core:
    """The finite difference core: the assembly core refined onto refined_geometry(subcells)."""
    refined_geo = refined_geometry(subcells)
    system = assembly_core()
    split = splitting(subcells)
    calc = system.current_calc.mesh_refine(system.geometry, split, refined_geo)
    assemblies = system.geometry.array_refine(np.arange(system.geometry.cells), split).astype(int)
    isotopes = [system.isotopes[assembly] for assembly in assemblies]
    return Core(isotopes, 2, refined_geo, calc)


def nem_core(subcells: int = 1) -> Core:
    """The NEM core: the assembly core with the CMFD current calculator, split by splitting(subcells)."""
    system = assembly_core()
    nem_system = Core(system.isotopes, system.E, system.geometry, CMFDCurrentCalculator(system.current_calc.dc, 3))
    return mesh_refinement(nem_system, splitting(subcells))


def thermal_fission_source(flux: np.ndarray, isotopes) -> np.ndarray:
    """The thermal fission source (nu sigma_f times the thermal flux) of each cell."""
    thermal = flux[1::2]
    return thermal * np.array([iso.nusigmaf[1] for iso in isotopes])


def _fission_source_by_level(fission_source, intermediate_split, rest_split=None):
    intermediate_geo = original_geometry().refine_mesh(intermediate_split)
    if rest_split is None:
        rest_split = intermediate_geo.uniform_split(1)
    fission_source = np.bincount(
        intermediate_geo.array_refine(np.arange(intermediate_geo.cells), rest_split).astype(int), fission_source
    )
    return intermediate_geo, np.reshape(fission_source, (len(fission_source) // (19), 19), order="F")


def fd_fission_source_by_level(fission_source: np.ndarray, subcells: int = 2):
    """
    The fission source of fd_core(subcells) summed over each assembly of each of the 19 axial levels of 20 cm.

    Returns the geometry of these sums (the original geometry with its axial regions split into 20 cm levels) and the
    sums, a row per assembly and a column per level.
    """
    intermediate_split = [np.array([1] + [1] * 8)] * 2 + [np.array([1, 13, 4, 1])]
    rest_split = [[subcells] + [2 * subcells] * 8, [subcells] + [2 * subcells] * 8, [subcells] * 19]
    return _fission_source_by_level(fission_source, intermediate_split, rest_split)


def nem_fission_source_by_level(fission_source: np.ndarray):
    """
    The fission source of nem_core() (with subcells = 1, a node per 20 cm level) arranged by axial level.

    Returns the geometry of nem_core() and the source, a row per node of the plane and a column per level.
    """
    return _fission_source_by_level(fission_source, splitting(1))
