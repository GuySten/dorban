"""
tests for the stopping criteria of the NEM CMFD iterators
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from dorban import Cartesian, Core, Fissionable, Void
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.cmfd_iterator import GeneratorNemCMFDSource


def test_source_iteration_stops_with_some_zero_coupling_coefficients():
    fuel = Fissionable(
        "Fuel",
        scatter=np.array([[0, 0], [0.02, 0]]),
        absorb=np.array([0.01, 0.08]),
        nusigmaf=np.array([0.004, 0.06]),
        chi=np.array([1, 0]),
        diffusion=np.array([1.5, 0.4]),
    )
    geometry = Cartesian([np.full(3, 3.0), np.full(3, 3.0)], [Void()] * 4)
    materials = [fuel] * geometry.cells
    current_calc = CMFDCurrentCalculator.from_isotopes(materials, 2)
    # coupling coefficients have been computed, but some of them are zero
    current_calc.coupling = np.full_like(current_calc.coupling, 0.01)
    current_calc.coupling[0, 0, 0] = 0
    core = Core(materials, 2, geometry, current_calc)
    source = np.ones(geometry.cells * 2)
    solver = GeneratorNemCMFDSource(core, source)
    converged_flux = spsolve((solver.M - solver.F).tocsc(), source)
    solver.iter = 3
    solver.v = converged_flux
    solver.k_at_update, solver.v_at_update = 1, converged_flux  # the coupling coefficients came from this flux
    assert solver.should_stop(1, converged_flux)


def test_source_iteration_does_not_stop_before_the_coupling_coefficients_converge():
    fuel = Fissionable(
        "Fuel",
        scatter=np.array([[0, 0], [0.02, 0]]),
        absorb=np.array([0.01, 0.08]),
        nusigmaf=np.array([0.004, 0.06]),
        chi=np.array([1, 0]),
        diffusion=np.array([1.5, 0.4]),
    )
    geometry = Cartesian([np.full(3, 3.0), np.full(3, 3.0)], [Void()] * 4)
    materials = [fuel] * geometry.cells
    current_calc = CMFDCurrentCalculator.from_isotopes(materials, 2)
    current_calc.coupling = np.full_like(current_calc.coupling, 0.01)
    core = Core(materials, 2, geometry, current_calc)
    source = np.ones(geometry.cells * 2)
    solver = GeneratorNemCMFDSource(core, source)
    # the CMFD equations are solved, but the flux moved by 1% since the coupling coefficients were computed
    converged_flux = spsolve((solver.M - solver.F).tocsc(), source)
    solver.iter = 3
    solver.v = converged_flux
    solver.k_at_update, solver.v_at_update = 1, 1.01 * converged_flux
    assert not solver.should_stop(1, converged_flux)
