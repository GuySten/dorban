"""
OECD/NEA PWR MOX/UO2 core transient benchmark, initial core: 2 group nodal cross sections with assembly
discontinuity factors, all rods out and all rods in. Reference k = 1.06379 (rods out), control rod worth
6850 pcm, and assembly powers. Not from the ANL benchmark book. The problem is dorban.examples.oecd_pwr_mox_uo2,
shared with docs/source/PWRMOX.ipynb.
"""

import numpy as np
import pytest

from dorban.examples import oecd_pwr_mox_uo2
from dorban.nem.solve_nem import NEMSettings
from dorban.solve_equation import solve_k


@pytest.fixture(scope="module")
def solution():
    system, ari_system = oecd_pwr_mox_uo2.cores()
    settings = oecd_pwr_mox_uo2.fd_settings(system)
    k, flux = solve_k(system, settings)
    ari_k, ari_flux = solve_k(ari_system, settings)
    power, ari_power = oecd_pwr_mox_uo2.assembly_powers(system, flux, ari_system, ari_flux)
    aro_nem = oecd_pwr_mox_uo2.nem_core(system)
    ari_nem = oecd_pwr_mox_uo2.nem_core(ari_system)
    aro_k_nem, aro_flux_nem = solve_k(aro_nem, NEMSettings(split=aro_nem.geometry.uniform_split(3)))
    ari_k_nem, ari_flux_nem = solve_k(ari_nem, NEMSettings(split=ari_nem.geometry.uniform_split(3)))
    return {
        "k": k,
        "flux": flux,
        "ari_k": ari_k,
        "ari_flux": ari_flux,
        "power": power,
        "ari_power": ari_power,
        "aro_k_nem": aro_k_nem,
        "aro_flux_nem": aro_flux_nem,
        "ari_k_nem": ari_k_nem,
        "ari_flux_nem": ari_flux_nem,
    }


def test_k_all_rods_out(solution):
    assert solution["k"] == pytest.approx(oecd_pwr_mox_uo2.REFERENCE_K, abs=1e-4)


def test_control_rod_worth(solution):
    k, ari_k = solution["k"], solution["ari_k"]
    assert abs(1 / k - 1 / ari_k) == pytest.approx(oecd_pwr_mox_uo2.REFERENCE_ROD_WORTH, abs=1e-4)


def test_assembly_power_all_rods_out(solution):
    reference_power = oecd_pwr_mox_uo2.REFERENCE_POWER
    difference = np.abs(reference_power - solution["power"]) / reference_power
    assert np.max(difference) < 1e-2


def test_assembly_power_all_rods_in(solution):
    reference_ari = oecd_pwr_mox_uo2.REFERENCE_POWER_ARI
    difference = np.abs(reference_ari - solution["ari_power"]) / reference_ari
    assert np.max(difference) < 1e-2


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
