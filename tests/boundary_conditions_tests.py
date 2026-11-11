"""
tests for the finite difference vacuum boundary conditions against the exact solution of a bare slab
"""

import numpy as np
import pytest
import scipy.optimize as op

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import ExtrapolationLength, Reflector, Void
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import Core

DIFFUSION, ABSORB, NUSIGMAF, HALF_WIDTH = 1.0, 0.05, 0.06, 10.0
FUEL = Fissionable(
    "fuel",
    scatter=np.array([[0.0]]),
    absorb=np.array([ABSORB]),
    nusigmaf=np.array([NUSIGMAF]),
    chi=np.array([1.0]),
    diffusion=np.array([DIFFUSION]),
)


def exact_k(albedo):
    r"""
    One group slab, reflective at x = 0 and J = albedo * phi at x = a: phi = cos(Bx), so the boundary condition is
    :math:`B\tan(Ba) = \alpha / D` and :math:`k = \nu\Sigma_f / (\Sigma_a + DB^2)`.
    """
    buckling = op.brentq(
        lambda b: b * np.tan(b * HALF_WIDTH) - albedo / DIFFUSION,
        1e-9,
        np.pi / (2 * HALF_WIDTH) * (1 - 1e-12),
        xtol=1e-15,
    )
    return NUSIGMAF / (ABSORB + DIFFUSION * buckling**2)


def finite_difference_k(boundary, cells):
    geometry = Cartesian([np.full(cells, HALF_WIDTH / cells)], [Reflector(), boundary()])
    composition = [FUEL] * cells
    core = Core(composition, 1, geometry, CurrentCalculatorFD.from_isotopes(composition))
    k, _ = solve_k(core, FDSettings(split=geometry.uniform_split(1)))
    return k


@pytest.mark.parametrize("boundary, albedo", [(Void, 0.5), (ExtrapolationLength, ExtrapolationLength.constant)])
def test_vacuum_boundary_converges_with_second_order(boundary, albedo):
    reference = exact_k(albedo)
    coarse, fine = finite_difference_k(boundary, 20), finite_difference_k(boundary, 40)
    assert (coarse - reference) / (fine - reference) == pytest.approx(4, rel=0.05)
    assert (4 * fine - coarse) / 3 == pytest.approx(reference, abs=1e-5)
