"""
Tests that every eigenvalue solver selectable through Settings.solver_name works when called via solve_k
"""

import numpy as np
import pytest

from dorban import Cartesian, Core, CurrentCalculatorFD, Fissionable, Void, solve_k
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
