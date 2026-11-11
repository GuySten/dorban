from typing import Tuple

import numpy as np
import pytest
import scipy.optimize as op

from dorban.eigenvalue_solvers.slepc_methods import generalized_eigenvalue_slepc
from dorban.finite_differences.finite_difference_current_calculator import (
    CurrentCalculatorFD,
)
from dorban.finite_differences.solve_finite_difference import solve_k_diffusion
from dorban.geometry.boundary_conditions import Void
from dorban.geometry.cartesian import Cartesian
from dorban.materials import CrossSectionData, Fissionable
from dorban.system import Core


def diml_group1(
    absorb: float, fission: float, diff: float, length: float, cells: int
) -> Tuple[float, np.array]:
    """
    measures the difference between the given k and the actual eigenvalue of
    the system. system should represent a 1 dimensional homogenous system
    with void boundary conditions
    Parameters
    ----------
    absorb the macroscopic cross-section of absorption
    fission the macroscopic cross-section of nusigmaf
    diff the diffusion coefficient
    lengths the length of the system
    """
    ll = length / 2
    a = op.bisect(lambda x: x * np.tan(x) - ll / (2 * diff), 0, 1.1) / ll
    k = fission / (a**2 * diff + absorb)
    cells *= 2
    flux = np.cos([a * ll * (-1 + 2 * i / cells) for i in np.arange(cells)])[1::2]
    assert np.allclose(flux, flux[::-1])
    return k.item(), flux


def fuel_in_water(
    fuel: CrossSectionData,
    water: CrossSectionData,
    half_fuel: float,
    water_thickness: float,
    diffusion: Tuple[float, float],
    x: np.array,
) -> Tuple[float, np.array]:
    r"""
    Exact one-group diffusion solution of a fuel slab :math:`|x|<a` surrounded by water slabs
    :math:`a<|x|<a+b`, with the Marshak vacuum condition :math:`\phi+2D\phi'=0` outside the water.

    In the fuel :math:`\phi=\cos(Bx)` and :math:`k=\nu\Sigma_f/(\Sigma_a+D_fB^2)`. The water does not
    multiply, so there :math:`\phi=P\cosh(\kappa(|x|-a))+Q\sinh(\kappa(|x|-a))` with
    :math:`\kappa=\sqrt{\Sigma_{a,w}/D_w}`. Continuity of the flux and the current at :math:`|x|=a` gives
    :math:`P` and :math:`Q`, and the vacuum condition gives the equation for the fundamental :math:`B`.

    Parameters
    ----------
    fuel - Fissionable Isotope with cross sections in 1 energy group
    water - non Fissionable Isotope with cross sections in 1 energy group
    half_fuel - half the thickness of the fuel slab
    water_thickness - the thickness of 1 water slab
    diffusion - the diffusion coefficients of the fuel and of the water
    x - the points, measured from the center of the fuel, at which to evaluate the flux

    Returns
    -------
    The k eigenvalue of the problem and the flux at x, up to normalization.
    """
    d_f, d_w = (float(np.squeeze(d)) for d in diffusion)
    kappa = np.sqrt(water.absorb[0] / d_w)
    a, b = half_fuel, water_thickness

    def water_coefficients(buckling):
        return np.cos(buckling * a), -d_f * buckling * np.sin(buckling * a) / (d_w * kappa)

    def vacuum_condition(buckling):
        p, q = water_coefficients(buckling)
        return p * (np.cosh(kappa * b) + 2 * d_w * kappa * np.sinh(kappa * b)) + q * (
            np.sinh(kappa * b) + 2 * d_w * kappa * np.cosh(kappa * b)
        )

    buckling = op.brentq(vacuum_condition, 1e-12, np.pi / (2 * a) * (1 - 1e-12), xtol=1e-15)
    k = fuel.nusigmaf[0] / (fuel.absorb[0] + d_f * buckling**2)
    p, q = water_coefficients(buckling)
    r = np.abs(x)
    flux = np.where(r <= a, np.cos(buckling * x), p * np.cosh(kappa * (r - a)) + q * np.sinh(kappa * (r - a)))
    return k, flux


def test_hom_1dim_1group():
    cells = 10000
    length = 10

    fuel = Fissionable(
        "1",
        scatter=np.array([0]),
        absorb=np.array([8.983e-3]),
        nusigmaf=np.array([5.925e-3]),
        chi=np.array([1]),
        transport=np.array([2.531e-1]),
        fission=np.array([2.281e-3]),
    )
    diff_coefficent = np.array([1 / (3 * fuel.transport)])
    isotope = fuel
    isotopes = np.array([fuel] * cells)
    geo = Cartesian([np.full((cells,), length / cells)], (Void(), Void())).to_polybox()
    current = CurrentCalculatorFD(
        np.reshape(np.full((cells,), diff_coefficent), (cells, 1))
    )
    system = Core(isotopes, 1, geo, current)
    solver = generalized_eigenvalue_slepc
    k, flux = solve_k_diffusion(system, solver=solver)
    k_analytical, flux_analytical = diml_group1(
        isotope.absorb,
        isotope.nusigmaf,
        system.current_calc.dc[0].item(),
        length,
        cells,
    )
    flux_analytical = flux_analytical * np.sum(flux) / np.sum(flux_analytical)
    kerror = np.abs(k_analytical - k)
    fluxerror = np.linalg.norm(np.abs(flux - flux_analytical) / flux_analytical) / cells
    assert kerror < 1e-2
    assert fluxerror < 1e-2


def test_fuel_in_water(water, uranium):
    """
    This test compares two different solutions of the 1dim one-speed probelm of fuel surrounded by water from
    both sides with void boundary conditions after the water. The first solution is obtained from the finite
    differences kernel of DORBAN and the other is the analytic solution. Both the k eigenvalue and the
     corresponding eigenflux are obtained. The eigenvalues are compared as is and the fluxes are compared
     after normalization as the eigenflux isn't well-defined.
    """
    water_den = 1
    ur_den = 1
    diff_water = 1 / (3 * water.transport)
    diff_fuel = 1 / (3 * uranium.transport)
    water_length = 10
    fuel_length = 20
    cells_water = 3000
    cells_fuel = 6000
    diff_coefficient = np.hstack(
        [
            np.full((cells_water,), diff_water),
            np.full((cells_fuel,), diff_fuel),
            np.full((cells_water,), diff_water),
        ]
    )
    den = {water: (water_den, 0, water_den), uranium: (0, ur_den, 0)}
    cells = (cells_water, cells_fuel, cells_water)
    help_isotopes = {
        mat: np.hstack([np.full((c,), d) for c, d in zip(cells, densities)])
        for mat, densities in den.items()
    }
    isotopes = np.zeros_like(help_isotopes[water], dtype=object)
    for mat, d in help_isotopes.items():
        density = d[np.nonzero(d)[0]][0]
        isotopes[np.nonzero(d)] = mat * density
    clengths = (water_length, fuel_length, water_length)
    lengths = np.hstack(
        [np.full((c,), length / c) for c, length in zip(cells, clengths)]
    )
    geo = Cartesian((lengths,), (Void(), Void())).to_polybox()
    system = Core(
        isotopes,
        1,
        geo,
        CurrentCalculatorFD(np.reshape(diff_coefficient, (len(diff_coefficient), 1))),
    )
    solver = generalized_eigenvalue_slepc
    k, flux = solve_k_diffusion(
        system, solver=solver, lin_atol=1e-10, lin_rtol=1e-10, atol_vector=1e-12
    )

    edges = np.concatenate([[0], np.cumsum(lengths)]) - (water_length + fuel_length / 2)
    centers = (edges[1:] + edges[:-1]) / 2
    k_analytic, flux_analytic = fuel_in_water(uranium, water, fuel_length / 2, water_length, (diff_fuel, diff_water), centers)
    flux = flux / np.sum(flux)
    flux_analytic = flux_analytic / np.sum(flux_analytic)
    relative_error = np.abs((flux - flux_analytic) / flux_analytic)
    assert k == pytest.approx(k_analytic, abs=1e-10)
    assert np.max(relative_error) < 1e-6
