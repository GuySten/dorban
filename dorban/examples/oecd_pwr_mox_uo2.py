"""
OECD/NEA PWR MOX/UO2 core transient benchmark, the steady states of the initial core: a quarter of a PWR core with UO2
and MOX fuel assemblies, 2 group nodal cross sections with assembly discontinuity factors, all rods out and all rods
in, solved with finite differences and with NEM. Reference k = 1.06379 (all rods out), control rod worth 6850 pcm, and
the assembly powers of both configurations computed by PARCS. Not from the ANL benchmark book.

The cross sections of the fuel assemblies are read from the benchmark's cross section files, which are package data in
dorban/examples/data/oecd_pwr_mox_uo2.

This is the specification shared by the documentation notebook PWRMOX.ipynb and the benchmark tests.
"""

import re
from importlib.resources import files

import numpy as np

from dorban.finite_differences.finite_difference_current_calculator import DiscontinuityCurrentCalculator
from dorban.geometry.boundary_conditions import Reflector, Void
from dorban.geometry.cartesian import Cartesian
from dorban.materials import CrossSectionData, Fissionable, Isotope
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.settings import FDSettings
from dorban.system import Core
from dorban.utils import normalize

REFERENCE_K = 1.06379
REFERENCE_ROD_WORTH = 6850e-5  # 6850 pcm

# The assembly powers of the fuel assemblies, in the order of the cells of the core, one line per row of the core.
# fmt: off
REFERENCE_POWER = [  # PARCS, all rods out
    0.4205, 0.4967, 0.404, 0.3407,
    1.0112, 0.9855, 1.0067, 0.8907, 0.5817, 0.2796,
    1.0438, 0.9297, 1.1275, 1.1463, 1.0685, 0.7462, 0.2796,
    1.0414, 1.3504, 1.2508, 1.3137, 0.9146, 1.0685, 0.5817,
    1.5102, 1.2775, 1.4428, 1.0912, 1.3137, 1.1463, 0.8907, 0.3407,
    1.3882, 1.2271, 1.3094, 1.4428, 1.2508, 1.1275, 1.0067, 0.404,
    1.6824, 1.516, 1.2271, 1.2775, 1.3504, 0.9297, 0.9855, 0.4967,
    1.3415, 1.6824, 1.3882, 1.5102, 1.0414, 1.0438, 1.0112, 0.4205,
]
REFERENCE_POWER_ARI = [  # PARCS, all rods in
    0.2033, 0.2629, 0.2002, 0.1813,
    0.2972, 0.4841, 0.3295, 0.4438, 0.1865, 0.1825,
    0.6688, 0.4484, 0.9897, 0.529, 0.6968, 0.5486, 0.1825,
    0.736, 1.8573, 1.975, 1.7111, 0.5056, 0.6968, 0.1865,
    2.1958, 2.1194, 2.5058, 1.885, 1.7111, 0.529, 0.4438, 0.1813,
    1.1614, 1.7879, 1.185, 2.5058, 1.975, 0.9897, 0.3295, 0.2002,
    2.4402, 2.3545, 1.7879, 2.1194, 1.8573, 0.4484, 0.4841, 0.2629,
    1.1443, 2.4402, 1.1614, 2.1958, 0.736, 0.6688, 0.2972, 0.2033,
]
TRANSPORT_POWER = [  # the transport code DeCART, all rods out
    0.413, 0.491, 0.393, 0.34,
    0.997, 0.978, 0.991, 0.892, 0.585, 0.281,
    1.032, 0.917, 1.14, 1.142, 1.067, 0.754, 0.281,
    1.035, 1.348, 1.247, 1.308, 0.904, 1.067, 0.585,
    1.525, 1.277, 1.446, 1.076, 1.308, 1.143, 0.892, 0.341,
    1.418, 1.245, 1.325, 1.446, 1.247, 1.114, 0.991, 0.393,
    1.735, 1.563, 1.245, 1.277, 1.349, 0.918, 0.978, 0.491,
    1.374, 1.735, 1.418, 1.525, 1.035, 1.032, 0.997, 0.413,
]
# fmt: on

GROUP_CS = re.compile(r"\* GROUP[^\n]+\n([^\*]+)")
BURNUP = re.compile(r"\* BURNUP([^\n]+)")
ORDER = ["transport", "absorb", "nusigmaf", "kappa", "scatter", "adf"]


def parse_file(
    file, row: int, col: int, groups: int, th_conditions_num, name: str = ""
) -> dict[float, CrossSectionData]:
    """
    Read a cross section file of the benchmark into a library of materials by burnup. For each burnup the file has the
    cross sections in the order ORDER, a table per group (per pair of groups for the scattering) of th_conditions_num
    values, one per set of thermal hydraulic conditions; the value in the given row and column of each table is taken.
    file is a path or a Traversable from importlib.resources.
    """
    library = {}
    with file.open() as data:
        data = data.read()
        result = GROUP_CS.findall(data)
        result = [r for r in result if len(r.split()) == th_conditions_num]
        depletion = BURNUP.findall(data)
        depletion = [float(d) for d in depletion]
        index = 0
        for i, dep in enumerate(depletion):
            d = {}
            d["chi"] = np.array([1, 0])
            for cs in ORDER:
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


def libraries() -> dict[str, dict[float, CrossSectionData]]:
    """
    The cross sections of the fuel assemblies by burnup: the MOX assemblies m40 and m43 and the UO2 assemblies u42 and
    u45 unrodded, and u42 and u45 rodded.
    """
    groups = 2
    row = 2
    col = 3
    directory = files("dorban.examples") / "data" / "oecd_pwr_mox_uo2"
    library = {}
    library["m40"] = parse_file(directory / "2G_XSEC_m40_unrodded", row, col, groups, 27, "m40")
    library["m43"] = parse_file(directory / "2G_XSEC_m43_unrodded", row, col, groups, 27, "m43")
    library["u42"] = parse_file(directory / "2G_XSEC_u42_unrodded", row, col, groups, 27, "u42")
    library["u45"] = parse_file(directory / "2G_XSEC_u45_unrodded", row, col, groups, 27, "u45")
    library["u42 rod"] = parse_file(directory / "2G_XSEC_u42_rodded", row, col, groups, 27, "u42 rod")
    library["u45 rod"] = parse_file(directory / "2G_XSEC_u45_rodded", row, col, groups, 27, "u45 rod")
    return library


def compositions() -> tuple[list, list]:
    """
    The material of each cell of the core, all rods out and all rods in: the fuel assemblies at their burnups (an r
    marks a rodded assembly) and the reflector ref.
    """
    library = libraries()
    m40, m43, u42, u45 = library["m40"], library["m43"], library["u42"], library["u45"]
    u42rod, u45rod = library["u42 rod"], library["u45 rod"]
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
    # One line per row of the core (see geometry()).
    # fmt: off
    composition = [
        ref,      ref,      ref,      ref,      ref,
        U42_325,  U45_175,  M43_350,  U45_20,   ref,      ref,      ref,
        U45_015,  M40_015,  U45_015,  M43_015,  U42_175,  U45_325,  ref,      ref,
        M43_175,  U42_325,  M43_175,  U45_20,   U45_015,  M43_015,  U45_325,  ref,
        U45_375,  U42_015,  U42_225,  U42_015,  U42_375,  U45_015,  U42_175,  ref,      ref,
        U45_015,  M40_225,  U42_015,  M40_375,  U42_015,  U45_20,   M43_015,  U45_20,   ref,
        U42_225,  U45_325,  U42_225,  U42_015,  U42_225,  M43_175,  U45_015,  M43_350,  ref,
        U42_015,  U42_175,  U45_325,  M40_225,  U42_015,  U42_325,  M40_015,  U45_175,  ref,
        U42_35,   U42_015,  U42_225,  U45_015,  U45_375,  M43_175,  U45_015,  U42_325,  ref,
    ]
    ari_composition = [
        ref,      ref,      ref,      ref,      ref,
        U42_325,  U45_175,  M43_350,  U45_20,   ref,      ref,      ref,
        U45r_015, M40_015,  U45r_015, M43_015,  U42r_175, U45_325,  ref,      ref,
        M43_175,  U42r_325, M43_175,  U45r_20,  U45_015,  M43_015,  U45_325,  ref,
        U45r_375, U42_015,  U42_225,  U42_015,  U42r_375, U45_015,  U42r_175, ref,      ref,
        U45_015,  M40_225,  U42_015,  M40_375,  U42_015,  U45r_20,  M43_015,  U45_20,   ref,
        U42r_225, U45_325,  U42r_225, U42_015,  U42_225,  M43_175,  U45r_015, M43_350,  ref,
        U42_015,  U42_175,  U45_325,  M40_225,  U42_015,  U42r_325, M40_015,  U45_175,  ref,
        U42r_35,  U42_015,  U42r_225, U45_015,  U45r_375, M43_175,  U45r_015, U42_325,  ref,
    ]
    # fmt: on
    return composition, ari_composition


def geometry():
    """
    The geometry: a quarter of the core on a 9 x 9 Cartesian mesh of assemblies, numbered row by row from the outer
    edge, with half assemblies on the symmetry lines (the first column and the last row) and black absorber in the 8
    cells of the corner outside the reflector.
    """
    return Cartesian(
        [[10.71] + [21.42] * 8, [21.42] * 8 + [10.71]],
        [Reflector(), Void(), Void(), Reflector()],
        [5, 6, 7, 8, 16, 17, 26, 35],
    ).to_polybox()


def cores() -> tuple[Core, Core]:
    """
    The cores all rods out and all rods in, with finite difference current calculators that use the assembly
    discontinuity factors.
    """
    composition, ari_composition = compositions()
    mixtures = set(composition)
    isotopes = {m: [c == m for c in composition] for m in mixtures}
    diff = sum(np.outer(den, 1 / (3 * isotope.transport)) for isotope, den in isotopes.items())
    adf = {mat: np.array([list(mat.adf)] * 4) for mat in mixtures}
    discontinuity = np.sum([np.multiply.outer(den, adf[mat]) for mat, den in isotopes.items()], axis=0)

    calc = DiscontinuityCurrentCalculator(diff, discontinuity)
    ari_calc = calc.from_isotopes(ari_composition, transport=True, face_num=4)

    geo = geometry()
    system = Core(np.array(composition), 2, geo, calc)
    ari_system = Core(ari_composition, 2, geo, ari_calc)
    return system, ari_system


def fd_settings(system: Core, subcells: int = 30) -> FDSettings:
    """The finite difference settings: each cell split into subcells x subcells."""
    split = [[subcells, subcells]] * system.geometry.cells
    return FDSettings(split=split, flux_atol=1e-10)


def assembly_powers(system: Core, flux: np.ndarray, ari_system: Core, ari_flux: np.ndarray):
    """
    The powers of the fuel assemblies all rods out and all rods in, from the fluxes of the cores of cores(), each
    normalized to a volume weighted mean of 1, in the order of REFERENCE_POWER.
    """
    power = np.array([np.dot(flux[cell * 2 : (cell * 2 + 2)], iso.kappa) for cell, iso in enumerate(system.isotopes)])
    no = power.nonzero()
    power = power[no]
    power = power / np.sum(power * normalize(system.geometry.volumes[no]))
    ari_power = np.array(
        [np.dot(ari_flux[cell * 2 : (cell * 2 + 2)], iso.kappa) for cell, iso in enumerate(ari_system.isotopes)]
    )[no]
    ari_power = ari_power / np.sum(ari_power * normalize(system.geometry.volumes[no]))
    return power, ari_power


def nem_core(system: Core) -> Core:
    """The core of cores() with a CMFD current calculator for NEM, with the same diffusion coefficients and discontinuity factors."""
    return Core(
        system.isotopes,
        system.E,
        system.geometry,
        CMFDCurrentCalculator(dc=system.current_calc.dc, df=system.current_calc.df, dim=2),
    )
