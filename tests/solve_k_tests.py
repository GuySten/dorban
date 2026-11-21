"""
Tests that every eigenvalue solver selectable through Settings.solver_name works when called via solve_k
"""

import numpy as np
import pytest

from dorban import Cartesian, Core, CurrentCalculatorFD, Fissionable, Isotope, Reflector, Void, solve_equation, solve_k
from dorban.settings import FDSettings


@pytest.fixture()
def small_core():
    fuel = Fissionable(
        "Fuel",
        scatter=np.array([[0, 0], [0.02, 0]]),
        absorb=np.array([0.01, 0.08]),
        nusigmaf=np.array([0.005, 0.135]),
        chi=np.array([1, 0]),
        diffusion=np.array([1.5, 0.4]),
    )
    geometry = Cartesian([np.array([10.0, 10.0]), np.array([10.0, 10.0])], [Void()] * 4)
    materials = [fuel] * geometry.cells
    return Core(materials, 2, geometry, CurrentCalculatorFD.from_isotopes(materials))


@pytest.mark.parametrize("adjoint", [False, True])
def test_power_iteration_agrees_with_arpack(small_core, adjoint):
    split = small_core.geometry.uniform_split(3)
    k_arpack, _ = solve_k(small_core, FDSettings(split=split, solver_name="arpack", adjoint=adjoint))
    k_power, _ = solve_k(small_core, FDSettings(split=split, solver_name="power iteration", adjoint=adjoint))
    assert k_power == pytest.approx(k_arpack, rel=1e-4)


def test_slepc_finds_the_fundamental_mode():
    # With Jacobi-Davidson, the former default, SLEPc converged to the second eigenvalue (k=0.3047) here.
    fuel = Fissionable(
        "Fuel",
        scatter=np.array([[0, 0], [0.02, 0]]),
        absorb=np.array([0.01, 0.08]),
        nusigmaf=np.array([0, 0.135]),
        chi=np.array([1, 0]),
        diffusion=np.array([1.5, 0.4]),
    )
    reflector = Isotope(
        "Reflector",
        scatter=np.array([[0, 0], [0.04, 0]]),
        absorb=np.array([0, 0.01]),
        diffusion=np.array([2, 0.3]),
    )
    geometry = Cartesian([np.array([2.713, 5.514]), np.array([9.34, 9.18])], [Reflector()] * 4)
    materials = [fuel, reflector, fuel, reflector]
    core = Core(materials, 2, geometry, CurrentCalculatorFD.from_isotopes(materials))
    split = geometry.uniform_split(50)
    k_slepc, flux = solve_k(core, FDSettings(split=split, solver_name="slepc"))
    k_power, _ = solve_k(core, FDSettings(split=split, solver_name="power iteration"))
    assert k_slepc == pytest.approx(k_power, rel=1e-4)
    assert np.all(flux > 0)


def test_solve_k_rejects_a_flux_that_changes_sign(small_core, monkeypatch):
    # The fundamental mode is the only eigenvector that does not change sign, so a flux with a
    # significant negative part means the eigensolver converged to another mode.
    def higher_mode(A, M, **kwargs):
        flux = np.ones(A.shape[0])
        flux[: A.shape[0] // 2] = -0.5
        return 0.5, flux

    monkeypatch.setattr(solve_equation, "generalized_eigenvalue_slepc", higher_mode)
    with pytest.raises(RuntimeError, match="fundamental mode"):
        solve_k(small_core, FDSettings(split=small_core.geometry.uniform_split(3), solver_name="slepc"))
