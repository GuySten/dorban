"""
OECD/NEA PWR MOX/UO2 core transient benchmark, initial core: 2 group nodal cross sections with assembly
discontinuity factors, all rods out and all rods in. Reference k = 1.06379 (rods out), control rod worth
6850 pcm, and assembly powers. Not from the ANL benchmark book. Transcribed from docs/source/PWRMOX.ipynb.
"""

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from dorban.finite_differences.finite_difference_current_calculator import DiscontinuityCurrentCalculator
from dorban.geometry.boundary_conditions import Reflector, Void
from dorban.geometry.cartesian import Cartesian
from dorban.materials import CrossSectionData, Fissionable, Isotope
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core
from dorban.utils import normalize


def _compute():
    # notebook cell 4
    gropu_cs = r"\* GROUP[^\n]+\n([^\*]+)"
    group_cs = re.compile(gropu_cs)
    burnup = r"\* BURNUP([^\n]+)"
    burnup = re.compile(burnup)
    order = ["transport", "absorb", "nusigmaf", "kappa", "scatter", "adf"]
    groups = 2
    row = 2
    col = 3

    def parse_file(
        file: str, row: int, col: int, groups: int, th_conditions_num, name: str = ""
    ) -> Dict[float, CrossSectionData]:
        library = {}
        with open(file) as data:
            data = data.read()
            result = group_cs.findall(data)
            result = [r for r in result if len(r.split()) == th_conditions_num]
            depletion = burnup.findall(data)
            depletion = [float(d) for d in depletion]
            index = 0
            for i, dep in enumerate(depletion):
                d = {}
                d["chi"] = np.array([1, 0])
                for cs in order:
                    if cs == "scatter":
                        value = np.zeros((groups, groups))
                        for g1 in range(groups):
                            for g2 in range(groups):
                                rows = result[index].split("\n")
                                value[g2, g1] = float(rows[row - 1].split()[col - 1])
                                index += 1
                    else:
                        value = np.zeros(groups)
                        for g in range(groups):
                            rows = result[index].split("\n")
                            value[g] = float(rows[row - 1].split()[col - 1])
                            index += 1
                    d[cs] = value
                library[dep] = Fissionable(name=f"{name} burnup {dep}", **d)
        return library

    directory = str(Path(__file__).resolve().parent.parent / "mox_bench_2g_nodal_xsec") + "/"
    m40file = directory + "2G_XSEC_m40_unrodded"
    m40 = parse_file(m40file, row, col, groups, 27, "m40")
    m43file = directory + "2G_XSEC_m43_unrodded"
    m43 = parse_file(m43file, row, col, groups, 27, "m43")
    u42file = directory + "2G_XSEC_u42_unrodded"
    u42 = parse_file(u42file, row, col, groups, 27, "u42")
    u45file = directory + "2G_XSEC_u45_unrodded"
    u45 = parse_file(u45file, row, col, groups, 27, "u45")
    u42rodfile = directory + "2G_XSEC_u42_rodded"
    u42rod = parse_file(u42rodfile, row, col, groups, 27, "u42 rod")
    u45rodfile = directory + "2G_XSEC_u45_rodded"
    u45rod = parse_file(u45rodfile, row, col, groups, 27, "u45 rod")

    # notebook cell 6
    M40_015 = m40[0.15]
    M40_225 = m40[22.5]
    M40_375 = m40[37.5]
    M43_015 = m43[0.15]
    M43_175 = m43[17.5]
    M43_350 = m43[35.0]
    U45_015 = u45[0.15]
    U45_175 = u45[17.5]
    U45_20 = u45[20.0]
    U45_325 = u45[32.5]
    U45_375 = u45[37.5]
    U42_015 = u42[0.15]
    U42_175 = u42[17.5]
    U42_225 = u42[22.5]
    U42_325 = u42[32.5]
    U42_35 = u42[35.0]
    U42_375 = u42[37.5]
    U42r_35 = u42rod[35.0]
    U42r_225 = u42rod[22.5]
    U45r_375 = u45rod[37.5]
    U45r_015 = u45rod[0.15]
    U42r_325 = u42rod[32.5]
    U45r_20 = u45rod[20.0]
    U42r_375 = u42rod[37.5]
    U42r_175 = u42rod[17.5]
    ref = Isotope(
        "",
        scatter=np.array([[0, 0], [2.75262e-02, 0]]),
        absorb=np.array([2.42533e-03, 3.72376e-02]),
        transport=np.array([3.01887e-01, 1.22567e00]),
        adf=np.array([1.17639, 0.22755]),
    )

    # notebook cell 8
    geometry = Cartesian(
        [[10.71] + [21.42] * 8, [21.42] * 8 + [10.71]],
        [Reflector(), Void(), Void(), Reflector()],
        [5, 6, 7, 8, 16, 17, 26, 35],
    ).to_polybox()

    # notebook cell 10
    composition = [
        ref,
        ref,
        ref,
        ref,
        ref,
        U42_325,
        U45_175,
        M43_350,
        U45_20,
        ref,
        ref,
        ref,
        U45_015,
        M40_015,
        U45_015,
        M43_015,
        U42_175,
        U45_325,
        ref,
        ref,
        M43_175,
        U42_325,
        M43_175,
        U45_20,
        U45_015,
        M43_015,
        U45_325,
        ref,
        U45_375,
        U42_015,
        U42_225,
        U42_015,
        U42_375,
        U45_015,
        U42_175,
        ref,
        ref,
        U45_015,
        M40_225,
        U42_015,
        M40_375,
        U42_015,
        U45_20,
        M43_015,
        U45_20,
        ref,
        U42_225,
        U45_325,
        U42_225,
        U42_015,
        U42_225,
        M43_175,
        U45_015,
        M43_350,
        ref,
        U42_015,
        U42_175,
        U45_325,
        M40_225,
        U42_015,
        U42_325,
        M40_015,
        U45_175,
        ref,
        U42_35,
        U42_015,
        U42_225,
        U45_015,
        U45_375,
        M43_175,
        U45_015,
        U42_325,
        ref,
    ]
    ari_composition = [
        ref,
        ref,
        ref,
        ref,
        ref,
        U42_325,
        U45_175,
        M43_350,
        U45_20,
        ref,
        ref,
        ref,
        U45r_015,
        M40_015,
        U45r_015,
        M43_015,
        U42r_175,
        U45_325,
        ref,
        ref,
        M43_175,
        U42r_325,
        M43_175,
        U45r_20,
        U45_015,
        M43_015,
        U45_325,
        ref,
        U45r_375,
        U42_015,
        U42_225,
        U42_015,
        U42r_375,
        U45_015,
        U42r_175,
        ref,
        ref,
        U45_015,
        M40_225,
        U42_015,
        M40_375,
        U42_015,
        U45r_20,
        M43_015,
        U45_20,
        ref,
        U42r_225,
        U45_325,
        U42r_225,
        U42_015,
        U42_225,
        M43_175,
        U45r_015,
        M43_350,
        ref,
        U42_015,
        U42_175,
        U45_325,
        M40_225,
        U42_015,
        U42r_325,
        M40_015,
        U45_175,
        ref,
        U42r_35,
        U42_015,
        U42r_225,
        U45_015,
        U45r_375,
        M43_175,
        U45r_015,
        U42_325,
        ref,
    ]

    # notebook cell 12
    mixtures = set(composition)
    isotopes = {m: [c == m for c in composition] for m in mixtures}
    diff = sum(np.outer(den, 1 / (3 * isotope.transport)) for isotope, den in isotopes.items())
    adf = {mat: np.array([list(mat.adf)] * 4) for mat in mixtures}
    discontinuity = np.sum([np.multiply.outer(den, adf[mat]) for mat, den in isotopes.items()], axis=0)

    calc = DiscontinuityCurrentCalculator(diff, discontinuity)
    ari_calc = calc.from_isotopes(ari_composition, transport=True, face_num=4)

    system = Core(np.array(composition), 2, geometry, calc)
    ari_system = Core(ari_composition, 2, geometry, ari_calc)

    # notebook cell 14
    subcells = 30
    split = [[subcells, subcells]] * system.geometry.cells
    settings = FDSettings(split=split, flux_atol=1e-10)

    # notebook cell 16
    k, flux = solve_k(system, settings)
    ari_k, ari_flux = solve_k(ari_system, settings)

    # notebook cell 22
    reference_power = [
        0.4205,
        0.4967,
        0.404,
        0.3407,
        1.0112,
        0.9855,
        1.0067,
        0.8907,
        0.5817,
        0.2796,
        1.0438,
        0.9297,
        1.1275,
        1.1463,
        1.0685,
        0.7462,
        0.2796,
        1.0414,
        1.3504,
        1.2508,
        1.3137,
        0.9146,
        1.0685,
        0.5817,
        1.5102,
        1.2775,
        1.4428,
        1.0912,
        1.3137,
        1.1463,
        0.8907,
        0.3407,
        1.3882,
        1.2271,
        1.3094,
        1.4428,
        1.2508,
        1.1275,
        1.0067,
        0.404,
        1.6824,
        1.516,
        1.2271,
        1.2775,
        1.3504,
        0.9297,
        0.9855,
        0.4967,
        1.3415,
        1.6824,
        1.3882,
        1.5102,
        1.0414,
        1.0438,
        1.0112,
        0.4205,
    ]
    reference_ari = [
        0.2033,
        0.2629,
        0.2002,
        0.1813,
        0.2972,
        0.4841,
        0.3295,
        0.4438,
        0.1865,
        0.1825,
        0.6688,
        0.4484,
        0.9897,
        0.529,
        0.6968,
        0.5486,
        0.1825,
        0.736,
        1.8573,
        1.975,
        1.7111,
        0.5056,
        0.6968,
        0.1865,
        2.1958,
        2.1194,
        2.5058,
        1.885,
        1.7111,
        0.529,
        0.4438,
        0.1813,
        1.1614,
        1.7879,
        1.185,
        2.5058,
        1.975,
        0.9897,
        0.3295,
        0.2002,
        2.4402,
        2.3545,
        1.7879,
        2.1194,
        1.8573,
        0.4484,
        0.4841,
        0.2629,
        1.1443,
        2.4402,
        1.1614,
        2.1958,
        0.736,
        0.6688,
        0.2972,
        0.2033,
    ]
    transport_power = [
        0.413,
        0.491,
        0.393,
        0.34,
        0.997,
        0.978,
        0.991,
        0.892,
        0.585,
        0.281,
        1.032,
        0.917,
        1.14,
        1.142,
        1.067,
        0.754,
        0.281,
        1.035,
        1.348,
        1.247,
        1.308,
        0.904,
        1.067,
        0.585,
        1.525,
        1.277,
        1.446,
        1.076,
        1.308,
        1.143,
        0.892,
        0.341,
        1.418,
        1.245,
        1.325,
        1.446,
        1.247,
        1.114,
        0.991,
        0.393,
        1.735,
        1.563,
        1.245,
        1.277,
        1.349,
        0.918,
        0.978,
        0.491,
        1.374,
        1.735,
        1.418,
        1.525,
        1.035,
        1.032,
        0.997,
        0.413,
    ]

    # notebook cell 23
    power = np.array([np.dot(flux[cell * 2 : (cell * 2 + 2)], iso.kappa) for cell, iso in enumerate(system.isotopes)])
    no = power.nonzero()
    power = power[no]
    power = power / np.sum(power * normalize(system.geometry.volumes[no]))
    ari_power = np.array(
        [np.dot(ari_flux[cell * 2 : (cell * 2 + 2)], iso.kappa) for cell, iso in enumerate(ari_system.isotopes)]
    )[no]
    ari_power = ari_power / np.sum(ari_power * normalize(system.geometry.volumes[no]))

    # notebook cell 24
    difference = np.abs(reference_power - power) / reference_power
    transport_dif = np.abs(np.array(reference_power) - np.array(transport_power)) / np.array(transport_power)
    np.linalg.norm(transport_dif) / np.sqrt(len(transport_dif))
    adiffernece = np.abs(reference_ari - ari_power) / reference_ari

    # notebook cell 25
    np.max(transport_dif)

    # notebook cell 29
    from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
    from dorban.nem.solve_nem import NEMSettings

    aro_nem = Core(
        system.isotopes,
        system.E,
        system.geometry,
        CMFDCurrentCalculator(dc=system.current_calc.dc, df=system.current_calc.df, dim=2),
    )
    ari_nem = Core(
        ari_system.isotopes,
        ari_system.E,
        ari_system.geometry,
        CMFDCurrentCalculator(dc=ari_system.current_calc.dc, df=ari_system.current_calc.df, dim=2),
    )

    # notebook cell 30
    aro_k_nem, aro_flux_nem = solve_k(aro_nem, NEMSettings(split=aro_nem.geometry.uniform_split(3)))

    # notebook cell 31
    ari_k_nem, ari_flux_nem = solve_k(ari_nem, NEMSettings(split=ari_nem.geometry.uniform_split(3)))
    return locals()


@pytest.fixture(scope="module")
def solution():
    return _compute()


def test_k_all_rods_out(solution):
    assert solution["k"] == pytest.approx(1.06379, abs=1e-4)


def test_control_rod_worth(solution):
    k, ari_k = solution["k"], solution["ari_k"]
    assert abs(1 / k - 1 / ari_k) == pytest.approx(6850e-5, abs=1e-4)


def test_assembly_power_all_rods_out(solution):
    assert np.max(solution["difference"]) < 1e-2


def test_assembly_power_all_rods_in(solution):
    assert np.max(solution["adiffernece"]) < 1e-2


def test_nem_k_agrees_with_finite_differences(solution):
    assert solution["aro_k_nem"] == pytest.approx(solution["k"], abs=1e-4)


def test_nem_rod_worth_agrees_with_finite_differences(solution):
    s = solution
    fd_worth = abs(1 / s["k"] - 1 / s["ari_k"])
    nem_worth = abs(1 / s["aro_k_nem"] - 1 / s["ari_k_nem"])
    assert nem_worth == pytest.approx(fd_worth, abs=1e-4)


def test_nem_flux_agrees_with_finite_differences_all_rods_out(solution):
    flux = solution["flux"]
    assert max(np.abs((solution["aro_flux_nem"] - flux) / flux)) < 2e-2


def test_nem_flux_agrees_with_finite_differences_all_rods_in(solution):
    ari_flux = solution["ari_flux"]
    assert max(np.abs((solution["ari_flux_nem"] - ari_flux) / ari_flux)) < 2e-2
