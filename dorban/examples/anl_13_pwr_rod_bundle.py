"""
ANL benchmark book (ANL-7416), problem 13-2: a PWR rod bundle with 2 energy groups, modelled pin cell by pin cell
with the steel and the water around it, solved with finite differences and with NEM. Reference k = 1.0855, power
peaking factor 1.418, and the book's fast and thermal flux distribution.

This is the specification shared by the documentation notebook PWR_rod_bundle.ipynb and the benchmark tests.
"""

import numpy as np

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import Reflector
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable, Isotope
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.system import Core
from dorban.utils import space_energy_reshape

REFERENCE_K = 1.0855
REFERENCE_PPF = 1.418

# The number of sub cells per cell, along each axis, of the finite differences solve and of the NEM solve.
FD_SUB_CELLS = 10
NEM_SUB_CELLS = 3

# fmt: off
# The book's reference fast and thermal fluxes, on a triangle of the 12 x 12 map: line i (counting from 0) holds
# row i of the map from its cell 11 - i back to its cell 0, the order reorder_flux puts the computed flux in.
REFERENCE_FAST = [
    15.62, 15.66, 15.74, 16.01, 16.47, 16.8, 17.01, 17.12, 17.11, 16.96, 16.84, 16.82,
    15.72, 15.81, 16.11, 16.58, 16.91, 17.12, 17.24, 17.23, 17.06, 16.93, 16.9,
    15.92, 16.25, 16.75, 17.08, 17.29, 17.41, 17.41, 17.22, 17.06, 17.01,
    16.7, 17.26, 17.58, 17.78, 17.93, 17.74, 17.71, 17.46, 17.38,
    17.89, 18.21, 18.3, 18.52, 18.56, 18.33, 18.03, 17.94,
    18.57, 18.74, 18.85, 18.82, 18.61, 18.34, 18.25,
    18.95, 18.99, 18.86, 18.72, 18.48, 18.4,
    19.07, 19.02, 18.81, 18.54, 18.46,
    19.03, 18.78, 18.48, 18.39,
    18.51, 18.24, 18.16,
    18.04, 17.98,
    17.94,
]
REFERENCE_THERMAL = [
    11.95, 11.77, 11.52, 10.77, 9.669, 8.915, 8.534, 8.561, 8.921, 9.527, 9.916, 10.01,
    11.52, 11.17, 10.37, 9.24, 8.972, 8.078, 8.108, 8.476, 9.105, 9.553, 9.718,
    10.72, 9.901, 8.739, 7.955, 7.546, 7.58, 7.956, 8.612, 9.09, 9.335,
    8.969, 7.729, 6.889, 6.41, 6.512, 6.929, 7.636, 8.202, 8.435,
    6.4, 5.469, 4.793, 5.134, 5.616, 6.35, 6.966, 7.216,
    4.76, 4.414, 4.476, 4.783, 5.563, 6.203, 6.456,
    4.241, 4.167, 4.205, 5.175, 5.862, 6.119,
    4.23, 4.531, 5.304, 5.936, 6.19,
    5.001, 5.731, 6.338, 6.589,
    6.46, 7.034, 7.274,
    7.534, 7.765,
    7.929,
]
# fmt: on


def geometry() -> Cartesian:
    """The geometry: a 12 x 12 cartesian map of the pin cells, the steel and the water, with reflective boundaries."""
    ref = Reflector()
    lengths = np.array(
        [0.47498, 0.34544, 1.87452, 1.87452, 1.87452, 1.87452, 1.87452, 1.87452, 1.87452, 0.34544, 0.47625, 0.47625]
    )
    return Cartesian([lengths, lengths[::-1]], [ref] * 4)


def materials():
    """The materials: the five kinds of fuel pins m1, ..., m5, the steel and the water."""
    m1 = Fissionable(
        "1",
        scatter=np.array([[0, 0], [1.069e-2, 0]]),
        absorb=np.array([8.983e-3, 5.892e-2]),
        nusigmaf=np.array([5.925e-3, 9.817e-2]),
        chi=np.array([1, 0]),
        transport=np.array([2.531e-1, 5.732e-1]),
        fission=np.array([2.281e-3, 4.038e-2]),
    )
    m2 = Fissionable(
        "2",
        scatter=np.array([[0, 0], [1.095e-2, 0]]),
        absorb=np.array([8.726e-3, 5.174e-2]),
        nusigmaf=np.array([5.242e-3, 8.228e-2]),
        chi=np.array([1, 0]),
        transport=np.array([2.536e-1, 5.767e-1]),
        fission=np.array([2.003e-3, 3.385e-2]),
    )
    m3 = Fissionable(
        "3",
        scatter=np.array([[0, 0], [1.112e-2, 0]]),
        absorb=np.array([8.587e-3, 4.717e-2]),
        nusigmaf=np.array([4.820e-3, 7.2e-2]),
        chi=np.array([1, 0]),
        transport=np.array([2.535e-1, 5.797e-1]),
        fission=np.array([1.830e-3, 2.962e-2]),
    )
    m4 = Fissionable(
        "4",
        scatter=np.array([[0, 0], [1.113e-2, 0]]),
        absorb=np.array([8.48e-3, 4.14e-2]),
        nusigmaf=np.array([4.337e-3, 5.9e-2]),
        chi=np.array([1, 0]),
        transport=np.array([2.533e-1, 5.837e-1]),
        fission=np.array([1.632e-3, 2.428e-2]),
    )
    m5 = Fissionable(
        "5",
        scatter=np.array([[0, 0], [1.016e-2, 0]]),
        absorb=np.array([9.593e-3, 1.626e-1]),
        nusigmaf=np.array([5.605e-3, 2.424e-2]),
        chi=np.array([1, 0]),
        transport=np.array([2.506e-1, 5.853e-1]),
        fission=np.array([2.155e-3, 9.968e-3]),
    )
    st = Isotope(
        "steel",
        scatter=np.array([[0, 0], [9.095e-3, 0]]),
        absorb=np.array([1.043e-3, 4.394e-3]),
        transport=np.array([2.172e-1, 4.748e-1]),
    )
    wa = Isotope(
        "water",
        scatter=np.array([[0, 0], [3.682e-2, 0]]),
        absorb=np.array([1.983e-4, 7.796e-3]),
        transport=np.array([2.476e-1, 1.123]),
    )
    return m1, m2, m3, m4, m5, st, wa


def composition() -> list:
    """The arrangement of the materials in the 12 x 12 map, one row per line."""
    m1, m2, m3, m4, m5, st, wa = materials()
    # fmt: off
    comp = [
        wa, wa, wa, wa, wa, wa, wa, wa, wa, wa, wa, wa,
        wa, wa, wa, wa, wa, wa, wa, wa, wa, wa, wa, wa,
        wa, st, st, st, st, st, st, st, st, st, wa, wa,
        wa, st, m3, m2, m2, m2, m3, m3, m4, st, wa, wa,
        wa, st, m1, m1, m1, m5, m1, m2, m3, st, wa, wa,
        wa, st, m1, m1, m1, m1, m1, m1, m3, st, wa, wa,
        wa, st, m1, m5, m1, m1, m1, m5, m2, st, wa, wa,
        wa, st, m1, m1, m1, m1, m1, m1, m2, st, wa, wa,
        wa, st, m1, m1, m1, m5, m1, m1, m2, st, wa, wa,
        wa, st, m2, m1, m1, m1, m1, m1, m3, st, wa, wa,
        wa, st, st, st, st, st, st, st, st, st, wa, wa,
        wa, wa, wa, wa, wa, wa, wa, wa, wa, wa, wa, wa,
    ]
    # fmt: on
    return comp


def core() -> Core:
    """
    The core for finite differences: the geometry as a PolyBox and the diffusion coefficients from the transport
    cross sections.
    """
    comp = composition()
    calc = CurrentCalculatorFD.from_isotopes(comp, transport=True)
    return Core(comp, 2, geometry().to_polybox(), calc)


def nem_core() -> Core:
    """The core for NEM: the cartesian geometry and a CMFD current calculator with the same diffusion coefficients."""
    comp = composition()
    calc = CurrentCalculatorFD.from_isotopes(comp, transport=True)
    return Core(comp, 2, geometry(), CMFDCurrentCalculator(calc.dc, dim=2))


def power_peaking_factor(refined_system: Core, fine_flux: np.ndarray) -> float:
    """
    The power peaking factor: the largest fission source of a cell over the mean one, over the cells with fuel,
    from the flux of the refined core (one cell per sub cell of the solve).
    """
    fission_source = np.sum(
        space_energy_reshape(fine_flux, refined_system.geometry.cells, refined_system.E)
        * np.vstack([iso.fission for iso in refined_system.isotopes]),
        axis=1,
    )
    nonzero = fission_source.nonzero()
    fission_source = fission_source[nonzero]
    return np.max(fission_source) / np.mean(fission_source)


def reorder_flux(flux: np.ndarray) -> np.ndarray:
    """The flux of the 12 x 12 map in the order of the reference (REFERENCE_FAST + REFERENCE_THERMAL), normalized."""
    fastflux = np.reshape(flux[::2], (12, 12))
    thermal_flux = np.reshape(flux[1::2], (12, 12))
    fast = []
    thermal = []
    for i in range(12):
        fast.extend(fastflux[i][: 12 - i][::-1])
        thermal.extend(thermal_flux[i][: 12 - i][::-1])
    reorded_flux = np.array(fast + thermal)
    return reorded_flux / np.sum(reorded_flux)


def reference_flux() -> np.ndarray:
    """The reference fast and thermal fluxes, normalized as reorder_flux normalizes the computed flux."""
    reference = np.array(REFERENCE_FAST + REFERENCE_THERMAL)
    return reference / np.sum(reference)
