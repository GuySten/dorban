"""
This module contains the solver of the equation.
"""

from functools import partial
from typing import Callable

import numpy as np
import scipy.sparse.linalg as la

import dorban.finite_differences.fd_cross_sections_operators as fd
from dorban.eigenvalue_solvers.arnoldi_arpack import generalized_arnoldi
from dorban.eigenvalue_solvers.power_iteration import generalized_eigenvalue
from dorban.eigenvalue_solvers.slepc_methods import generalized_eigenvalue_slepc
from dorban.finite_differences.diffusion_operator import diffusion_matrix
from dorban.finite_differences.solve_finite_difference import solve_k_fd
from dorban.nem.solve_nem import NEMSettings, solve_k_nem
from dorban.rdfs.debug_df import debug_discontinuity_factors
from dorban.settings import FDSettings, Settings
from dorban.system import Core


def fixed_source(system: Core, source: np.array, solver: Callable = la.lgmres, **kwargs) -> np.array:
    r"""
    This function solves the fixed source equations


    .. math:: -\nabla D\nabla\phi+\Sigma_a\phi=F\phi+S


    Parameters
    ----------
    system: Core
     object that represents the system
    source: np.array
     array that represents the source. its shape should be (cells,E) and its
     value at (i,e) is the source intensity in the i-th cell at energy group e,
     that is the total source in the cell (the source density times the cell volume), not a density
    solver: Callable
        method to solve Ax=b, the default value is lgmres
    kwargs
        keyword arguments to be passed down to solver

    Returns
    -------
    np.array
        array that represents the flux at the system created by the source
    """
    fission = fd.cmfd_fission(system)
    absorber = diffusion_matrix(system.geometry, system.E, system.current_calc) + fd.cmfd_absorption(system)
    return solver(absorber - fission, source.flatten(), **kwargs)


def flux_with_given_boundary_current(system: Core, k: float, **kwargs) -> np.array:
    """
    computes the flux that gives a known current on the boundary and a given
    multiplication eigenvalue

    Parameters
    ----------
    system: Core
        object that represents the system, the boundary conditions
        should be KnownCurrent for this function to make sense
    k: float
        the known eigenvalue
    kwargs
        keyword arguments to be passed down to the linear solver.

    Returns
    -------
    array that represents the flux
    """
    fission = 1 / k * fd.cmfd_fission(system)
    absorber = diffusion_matrix(system.geometry, system.E, system.current_calc) + fd.cmfd_absorption(system)
    current = fd.boundary_current(system.geometry, system.E)
    flux = la.bicgstab(absorber - fission, current, **kwargs)[0]
    assert np.all(flux > 0), f"The flux wasn't strictly positive! {flux}"
    return flux


def _call_solver(core: Core, settings: Settings) -> tuple[float, np.array]:
    if isinstance(settings, NEMSettings):
        if settings.adjoint:
            raise NotImplementedError("Currently adjoint computations aren't implemented in NEM")
        return solve_k_nem(core, settings)
    if settings.solver_name == "power iteration":
        solver = generalized_eigenvalue
    elif settings.solver_name == "arpack":
        solver = generalized_arnoldi
    elif settings.solver_name == "slepc":
        # the unknowns are ordered by cell and then by energy group, which lets SLEPc solve group by group
        solver = partial(generalized_eigenvalue_slepc, groups=core.E)
    else:
        raise ValueError(f"there is no eigenvsolver with the name {settings.solver_name}")
    if isinstance(settings, FDSettings):
        k, flux = solve_k_fd(core, settings, solver)
        _check_fundamental_mode(flux)
        return k, flux
    else:
        raise NotImplementedError("The given settings have no solution implementation as of yet")


def _check_fundamental_mode(flux: np.array, tolerance: float = 1e-3):
    """
    Raise if the flux changes sign. The fundamental mode is the only eigenvector of the diffusion
    equation that does not change sign, so a flux with a significant negative part means the eigensolver
    converged to another eigenvalue. Negative values smaller than tolerance times the largest absolute
    value are accepted, as numerical noise where the flux is small.
    """
    scale = np.max(np.abs(flux))
    if np.max(flux) < -np.min(flux):
        flux = -flux
    if np.min(flux) < -tolerance * scale:
        raise RuntimeError(
            "The eigensolver did not converge to the fundamental mode: the flux changes sign "
            f"(smallest value {np.min(flux) / scale:.3g} of the largest). Try another solver_name."
        )


def _normalize_to_neutron_source(core: Core, k: float, flux: np.array) -> np.array:
    r"""
    Function used to normalize the flux to neutron source. In a k eigenvalue problem it means that flux is normalized
    such that :math:`F\phi=k`.

    Parameters
    ----------
    core: Core
     The core whose flux and eigenvalue were computed.
    k: float
     The k multiplication eigenvalue.
    flux: np.array
     The not necessarily normalized flux.

    Returns
    -------
    np.array
     The normalized flux
    """
    nufission_rate = np.dot(
        np.hstack([core.geometry.volumes[cell] * iso.nusigmaf for cell, iso in enumerate(core.isotopes)]), flux
    )
    return flux * k / nufission_rate


def solve_k(core: Core, settings: Settings) -> tuple[float, np.array]:
    r"""
    function to solve the k eigenvalue.
    :math:`-\nabla D\nabla\phi+\Sigma_a\phi=\frac{1}{k}F\phi`

    Parameters
    ----------
    core: Core
        The core
    settings: Settings
        The settings to be used in solving the eigenvalue problem

    Returns
    -------
    Tuple[float,np.array]
        tuple of the k and the flux
    """
    if settings.debug_discontinuity:
        bad_indices = debug_discontinuity_factors(core)
        if bad_indices:
            cell, face, neighbor = bad_indices[0]
            raise ValueError(
                f"There is a problem with the discontinuity factors between cell number {cell} and cell number {neighbor} across face number {face}."
            )
    k, flux = _call_solver(core, settings)
    return k, _normalize_to_neutron_source(core, k, flux)
