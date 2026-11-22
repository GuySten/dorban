"""
ANL benchmark book, problem 11: the three dimensional IAEA PWR, 2 groups, solved with finite differences and with NEM.
The problem is dorban.examples.anl_11_iaea3d, shared with docs/source/IAEA3D.ipynb.
"""

import numpy as np
import pytest

from dorban.examples import anl_11_iaea3d
from dorban.finite_differences.solve_finite_difference import solve_k_diffusion
from dorban.nem.solve_nem import NEMSettings
from dorban.solve_equation import solve_k


@pytest.fixture(scope="module")
def solution():
    core = anl_11_iaea3d.fd_core()
    k, flux = solve_k_diffusion(core, rtol_vector=1e-8, lin_rtol=1e-10)
    fission_source = anl_11_iaea3d.thermal_fission_source(flux, core.isotopes)
    _, rsource = anl_11_iaea3d.fd_fission_source_by_level(fission_source)

    nem_system = anl_11_iaea3d.nem_core()
    nem_k, nem_flux = solve_k(nem_system, NEMSettings(split=None))
    nem_flux = nem_flux / np.sum(nem_flux)
    nem_fission_source = anl_11_iaea3d.thermal_fission_source(nem_flux, nem_system.isotopes)
    _, nem_rsource = anl_11_iaea3d.nem_fission_source_by_level(nem_fission_source)
    return {
        "k": k,
        "flux": flux,
        "axial": np.mean(rsource, axis=0),
        "nem_k": nem_k,
        "nem_flux": nem_flux,
        "nem_axial": np.mean(nem_rsource, axis=0),
    }


def test_k(solution):
    assert solution["k"] == pytest.approx(anl_11_iaea3d.REFERENCE_K, abs=1e-3)


def test_nem_k_agrees_with_finite_differences(solution):
    assert solution["nem_k"] == pytest.approx(solution["k"], abs=1e-3)


def test_nem_axial_power_agrees_with_finite_differences(solution):
    nem_axial, axial = solution["nem_axial"], solution["axial"]
    nem_normalized_axial = (nem_axial / np.sum(nem_axial))[1:-1]
    normalized_axial = (axial / np.sum(axial))[1:-1]
    assert np.max(np.abs((nem_normalized_axial - normalized_axial) / normalized_axial)) < 0.2
