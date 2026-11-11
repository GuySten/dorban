"""
tests for the vacuum boundary conditions of finite differences and NEM against the exact solution of a bare slab,
and for the incoming current boundary condition against the exact solution of an absorbing slab
"""

import numpy as np
import pytest
import scipy.optimize as op

from dorban.finite_differences.finite_difference_current_calculator import CurrentCalculatorFD
from dorban.geometry.boundary_conditions import ExtrapolationLength, IncomingCurrent, Reflector, Void, ZeroFlux
from dorban.geometry.cartesian import Cartesian
from dorban.materials import Fissionable, Isotope
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.solve_nem import NEMSettings
from dorban.settings import FDSettings
from dorban.solve_equation import flux_with_given_boundary_current, solve_k
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
    :math:`B\tan(Ba) = \alpha / D` and :math:`k = \nu\Sigma_f / (\Sigma_a + DB^2)`. Zero flux is an infinite albedo,
    :math:`B = \pi / 2a`.
    """
    if np.isinf(albedo):
        return NUSIGMAF / (ABSORB + DIFFUSION * (np.pi / (2 * HALF_WIDTH)) ** 2)
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


@pytest.mark.parametrize(
    "boundary, albedo", [(Void, 0.5), (ExtrapolationLength, ExtrapolationLength.constant), (ZeroFlux, np.inf)]
)
def test_nem_vacuum_boundary(boundary, albedo):
    # the slab as a strip of two 5 cm nodes, reflective above and below
    geometry = Cartesian(
        [np.full(2, HALF_WIDTH / 2), np.array([5.0])], [Reflector(), boundary(), Reflector(), Reflector()]
    )
    composition = [FUEL] * geometry.cells
    core = Core(composition, 1, geometry, CMFDCurrentCalculator.from_isotopes(composition, 2))
    k, _ = solve_k(core, NEMSettings(split=None))
    assert k == pytest.approx(exact_k(albedo), abs=0.5e-5)


ABSORBER = Isotope("absorber", scatter=np.array([[0.0]]), absorb=np.array([ABSORB]), diffusion=np.array([DIFFUSION]))
INCOMING = 2.0


def exact_cell_flux(cells):
    r"""
    Absorbing slab, reflective at x = 0 and an incoming partial current :math:`J^-` at x = a: :math:`\phi=C\cosh(\kappa x)`
    with :math:`\kappa^2=\Sigma_a/D`, and the Marshak condition :math:`J^-=\phi/4+D\phi'/2` at x = a gives C.
    Returns the average flux of each of the cells.
    """
    kappa = np.sqrt(ABSORB / DIFFUSION)
    c = INCOMING / (np.cosh(kappa * HALF_WIDTH) / 4 + DIFFUSION * kappa * np.sinh(kappa * HALF_WIDTH) / 2)
    edges = np.linspace(0, HALF_WIDTH, cells + 1)
    return c * np.diff(np.sinh(kappa * edges)) / kappa / np.diff(edges)


def incoming_current_flux(cells):
    geometry = Cartesian([np.full(cells, HALF_WIDTH / cells)], [Reflector(), IncomingCurrent(np.array([INCOMING]))])
    composition = [ABSORBER] * cells
    core = Core(composition, 1, geometry, CurrentCalculatorFD.from_isotopes(composition))
    return flux_with_given_boundary_current(core, 1.0, rtol=1e-12, atol=1e-14)


def test_incoming_current_converges_with_second_order():
    errors = [np.max(np.abs(incoming_current_flux(n) / exact_cell_flux(n) - 1)) for n in (20, 40)]
    assert errors[0] / errors[1] == pytest.approx(4, rel=0.05)
    assert errors[1] < 6e-4
