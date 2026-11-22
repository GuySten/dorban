"""
ANL benchmark book, problem 13-2: a PWR rod bundle, 2 groups, pin cell by pin cell with steel and water
around it, solved with finite differences and NEM. Reference k = 1.0855, power peaking factor 1.418, and
the fast and thermal flux distribution. The problem is dorban.examples.anl_13_pwr_rod_bundle, shared with
docs/source/PWR_rod_bundle.ipynb.
"""

import numpy as np
import pytest

from dorban.examples import anl_13_pwr_rod_bundle
from dorban.nem.solve_nem import NEMSettings
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import mesh_refinement


@pytest.fixture(scope="module")
def solution():
    system = anl_13_pwr_rod_bundle.core()
    split = system.geometry.uniform_split(anl_13_pwr_rod_bundle.FD_SUB_CELLS)
    k_from_split_solve, flux = solve_k(system, FDSettings(split=split))

    refined_system = mesh_refinement(system, split)
    k, fine_flux = solve_k(refined_system, FDSettings(split=refined_system.geometry.uniform_split(1)))
    ppf = anl_13_pwr_rod_bundle.power_peaking_factor(refined_system, fine_flux)

    nem_system = anl_13_pwr_rod_bundle.nem_core()
    nem_split = nem_system.geometry.uniform_split(anl_13_pwr_rod_bundle.NEM_SUB_CELLS)
    k_nem, flux_nem = solve_k(nem_system, NEMSettings(split=nem_split))
    return {
        "k_from_split_solve": k_from_split_solve,
        "k": k,
        "ppf": ppf,
        "reorded_flux": anl_13_pwr_rod_bundle.reorder_flux(flux),
        "reference": anl_13_pwr_rod_bundle.reference_flux(),
        "k_nem": k_nem,
        "normalized_flux": flux / np.sum(flux),
        "normalized_nem": flux_nem / np.sum(flux_nem),
    }


def test_k(solution):
    assert solution["k_from_split_solve"] == pytest.approx(anl_13_pwr_rod_bundle.REFERENCE_K, abs=1e-4)


def test_power_peaking_factor(solution):
    assert solution["ppf"] == pytest.approx(anl_13_pwr_rod_bundle.REFERENCE_PPF, abs=0.1)


def test_flux_distribution(solution):
    reorded_flux, reference = solution["reorded_flux"], solution["reference"]
    assert np.linalg.norm((reorded_flux - reference) / reference) / len(reorded_flux - reference) < 0.0005


def test_nem_k_agrees_with_finite_differences(solution):
    assert solution["k_nem"] == pytest.approx(solution["k"], abs=1e-4)


def test_nem_flux_agrees_with_finite_differences(solution):
    normalized_flux, normalized_nem = solution["normalized_flux"], solution["normalized_nem"]
    assert np.max(np.abs(normalized_flux - normalized_nem) / normalized_flux) < 1e-3
