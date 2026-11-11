"""
ANL benchmark book, problem 13-2: a PWR rod bundle, 2 groups, pin cell by pin cell with steel and water
around it, solved with finite differences and NEM. Reference k = 1.0855, power peaking factor 1.418, and
the fast and thermal flux distribution. Transcribed from docs/source/PWR_rod_bundle.ipynb.
"""

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import Reflector
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable, Isotope
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core, mesh_refinement


def _compute():
    # notebook cell 4
    ref = Reflector()
    lengths = np.array(
        [0.47498, 0.34544, 1.87452, 1.87452, 1.87452, 1.87452, 1.87452, 1.87452, 1.87452, 0.34544, 0.47625, 0.47625]
    )
    geo = Cartesian([lengths, lengths[::-1]], [ref] * 4)

    # notebook cell 6
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

    # notebook cell 8
    comp = [
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        st,
        st,
        st,
        st,
        st,
        st,
        st,
        st,
        st,
        wa,
        wa,
        wa,
        st,
        m3,
        m2,
        m2,
        m2,
        m3,
        m3,
        m4,
        st,
        wa,
        wa,
        wa,
        st,
        m1,
        m1,
        m1,
        m5,
        m1,
        m2,
        m3,
        st,
        wa,
        wa,
        wa,
        st,
        m1,
        m1,
        m1,
        m1,
        m1,
        m1,
        m3,
        st,
        wa,
        wa,
        wa,
        st,
        m1,
        m5,
        m1,
        m1,
        m1,
        m5,
        m2,
        st,
        wa,
        wa,
        wa,
        st,
        m1,
        m1,
        m1,
        m1,
        m1,
        m1,
        m2,
        st,
        wa,
        wa,
        wa,
        st,
        m1,
        m1,
        m1,
        m5,
        m1,
        m1,
        m2,
        st,
        wa,
        wa,
        wa,
        st,
        m2,
        m1,
        m1,
        m1,
        m1,
        m1,
        m3,
        st,
        wa,
        wa,
        wa,
        st,
        st,
        st,
        st,
        st,
        st,
        st,
        st,
        st,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
        wa,
    ]

    # notebook cell 10
    calc = CurrentCalculatorFD.from_isotopes(comp, transport=True)
    system = Core(comp, 2, geo.to_polybox(), calc)

    # notebook cell 12
    from dorban.utils import space_energy_reshape

    sub_cells = 10
    split = system.geometry.uniform_split(sub_cells)
    settings = FDSettings(split=split)
    k, flux = solve_k(system, settings)
    k_from_split_solve = k

    # notebook cell 15
    refined_system = mesh_refinement(system, split)
    k, fine_flux = solve_k(refined_system, FDSettings(split=refined_system.geometry.uniform_split(1)))
    fission_source = np.sum(
        space_energy_reshape(fine_flux, refined_system.geometry.cells, system.E)
        * np.vstack([iso.fission for iso in refined_system.isotopes]),
        axis=1,
    )
    nonzero = fission_source.nonzero()
    fission_source = fission_source[nonzero]
    ppf = np.max(fission_source) / np.mean(fission_source)

    # notebook cell 17
    reference_fast = [
        15.62,
        15.66,
        15.74,
        16.01,
        16.47,
        16.8,
        17.01,
        17.12,
        17.11,
        16.96,
        16.84,
        16.82,
        15.72,
        15.81,
        16.11,
        16.58,
        16.91,
        17.12,
        17.24,
        17.23,
        17.06,
        16.93,
        16.9,
        15.92,
        16.25,
        16.75,
        17.08,
        17.29,
        17.41,
        17.41,
        17.22,
        17.06,
        17.01,
        16.7,
        17.26,
        17.58,
        17.78,
        17.93,
        17.74,
        17.71,
        17.46,
        17.38,
        17.89,
        18.21,
        18.3,
        18.52,
        18.56,
        18.33,
        18.03,
        17.94,
        18.57,
        18.74,
        18.85,
        18.82,
        18.61,
        18.34,
        18.25,
        18.95,
        18.99,
        18.86,
        18.72,
        18.48,
        18.4,
        19.07,
        19.02,
        18.81,
        18.54,
        18.46,
        19.03,
        18.78,
        18.48,
        18.39,
        18.51,
        18.24,
        18.16,
        18.04,
        17.98,
        17.94,
    ]

    reference_thermal = [
        11.95,
        11.77,
        11.52,
        10.77,
        9.669,
        8.915,
        8.534,
        8.561,
        8.921,
        9.527,
        9.916,
        10.01,
        11.52,
        11.17,
        10.37,
        9.24,
        8.972,
        8.078,
        8.108,
        8.476,
        9.105,
        9.553,
        9.718,
        10.72,
        9.901,
        8.739,
        7.955,
        7.546,
        7.58,
        7.956,
        8.612,
        9.09,
        9.335,
        8.969,
        7.729,
        6.889,
        6.41,
        6.512,
        6.929,
        7.636,
        8.202,
        8.435,
        6.4,
        5.469,
        4.793,
        5.134,
        5.616,
        6.35,
        6.966,
        7.216,
        4.76,
        4.414,
        4.476,
        4.783,
        5.563,
        6.203,
        6.456,
        4.241,
        4.167,
        4.205,
        5.175,
        5.862,
        6.119,
        4.23,
        4.531,
        5.304,
        5.936,
        6.19,
        5.001,
        5.731,
        6.338,
        6.589,
        6.46,
        7.034,
        7.274,
        7.534,
        7.765,
        7.929,
    ]
    reference = np.array(reference_fast + reference_thermal)
    fastflux = np.reshape(flux[::2], (12, 12))
    thermal_flux = np.reshape(flux[1::2], (12, 12))
    fast = []
    thermal = []
    for i in range(12):
        fast.extend(fastflux[i][: 12 - i][::-1])
        thermal.extend(thermal_flux[i][: 12 - i][::-1])
    reorded_flux = np.array(fast + thermal)
    reorded_flux = reorded_flux / np.sum(reorded_flux)
    reference = reference / np.sum(reference)

    # notebook cell 19
    from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator

    nem_system = Core(comp, 2, geo, CMFDCurrentCalculator(calc.dc, dim=2))

    # notebook cell 20
    from dorban.nem.solve_nem import NEMSettings

    sub_cells = 3
    split = nem_system.geometry.uniform_split(sub_cells)
    settings = NEMSettings(split=split)
    k_nem, flux_nem = solve_k(nem_system, settings)

    # notebook cell 22
    normalized_flux = flux / np.sum(flux)
    normalized_nem = flux_nem / np.sum(flux_nem)
    return locals()


@pytest.fixture(scope="module")
def solution():
    return _compute()


def test_k(solution):
    assert solution["k_from_split_solve"] == pytest.approx(1.0855, abs=1e-4)


def test_power_peaking_factor(solution):
    assert solution["ppf"] == pytest.approx(1.418, abs=0.1)


def test_flux_distribution(solution):
    reorded_flux, reference = solution["reorded_flux"], solution["reference"]
    assert np.linalg.norm((reorded_flux - reference) / reference) / len(reorded_flux - reference) < 0.0005


def test_nem_k_agrees_with_finite_differences(solution):
    assert solution["k_nem"] == pytest.approx(solution["k"], abs=1e-4)


def test_nem_flux_agrees_with_finite_differences(solution):
    normalized_flux, normalized_nem = solution["normalized_flux"], solution["normalized_nem"]
    assert np.max(np.abs(normalized_flux - normalized_nem) / normalized_flux) < 1e-3
