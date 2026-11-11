"""
This module contains the outer functions to call the NEM solver
"""

import logging
from typing import Optional, Tuple

import numpy as np

from dorban.nem.cmfd_iterator import GeneratorNemCMFD, GeneratorNemCMFDSource
from dorban.settings import Settings
from dorban.system import Core, mesh_refinement

logger = logging.getLogger(__name__)


class NEMSettings(Settings):
    """
    settings for a NEM calculation
    """

    def __init__(
        self,
        split=None,
        cmfd_iter=0,
        initial_flux: Optional[np.array] = None,
        initial_k: float = 1,
        k_tol: float = 1e-6,
        flux_rtol: float = 1e-5,
        flux_atol: float = 1e-5,
        lin_solve_rtol: float = 1e-10,
        lin_solve_atol: float = 1e-10,
        max_iter=np.inf,
        save_file: str = None,
        coupling=None,
    ):
        super(NEMSettings, self).__init__(
            False,
            initial_flux,
            initial_k,
            k_tol,
            flux_rtol,
            flux_atol,
            lin_solve_rtol,
            lin_solve_atol,
            max_iter,
            save_file,
        )
        self.cmfd_iter = cmfd_iter
        self.split = split
        self.initial_coupling = coupling


def solve_k_nem(system: Core, settings: NEMSettings) -> Tuple[float, np.array]:
    """
    function to solve the diffusion k problem using NEM and return the multiplication
    factor and the average flux at each node, to get all the computed information
    flux reconstruction should be used.

    Parameters
    ----------
    system: Core
     The core to be solved.
    settings: NEMSettings
     The settings to be used in the solution, the settings determine the split of the cells, the numerical tolerance
     constraints and the initial guess for the flux and eigenvalue.

    Returns
    -------
    Tuple[float, np.array]
     Tuple of the first eigenvalue and the average flux at each node.
    """
    core = mesh_refinement(system, settings.split) if settings.split else system
    if settings.initial_coupling is not None:
        core.current_calc.coupling = settings.initial_coupling
    solver = GeneratorNemCMFD(
        core,
        settings.cmfd_iter,
        settings.initial_k,
        settings.initial_flux,
        settings.k_tol,
        settings.flux_rtol,
        settings.flux_atol,
        settings.lin_solve_rtol,
        settings.lin_solve_atol,
        settings.max_iter,
    )
    for k, fine_flux in solver:
        logger.info(f"k number {solver.iter} is {k}")
        continue
    if settings.split is None:
        return k, fine_flux
    assemblies = system.geometry.array_refine(np.arange(system.geometry.cells), settings.split)
    volumes = system.geometry.array_refine(system.geometry.volumes, settings.split)
    normalized_fine_flux = fine_flux * np.repeat(core.geometry.volumes / volumes, core.E)
    assemblies_with_energy = np.vstack([assemblies * core.E + e for e in range(core.E)]).flatten(order="F").astype(int)
    return k, np.bincount(assemblies_with_energy, normalized_fine_flux)


def solve_source_nem(system: Core, settings: NEMSettings, source: np.array) -> np.array:
    """
    Function to solve the diffusion source problem using NEM and return the average flux at each node,
    to get all the computed information flux reconstruction should be used.

    Parameters
    ----------
    system: Core
     The core to be solved.
    settings: NEMSettings
     The settings to be used in the solution, the settings determine the split of the cells, the numerical tolerance
     constraints and the initial guess for the flux and eigenvalue.
    source: np.array
     The source is assumed to be constant within each node, energy group. The values of the source array
     are organized by energy group and the by cell, it is a one dimensional array of size cells x energy groups.

    Returns
    -------
    np.array
     The average flux induced by the source at each node.
    """
    core = mesh_refinement(system, settings.split) if settings.split else system
    if settings.initial_coupling is not None:
        core.current_calc.coupling = settings.initial_coupling
    refined_source = system.geometry.array_refine(source, settings.split) if settings.split else source
    solver = GeneratorNemCMFDSource(
        core,
        refined_source,
        settings.cmfd_iter,
        settings.initial_k,
        settings.initial_flux,
        settings.k_tol,
        settings.flux_rtol,
        settings.flux_atol,
        settings.lin_solve_rtol,
        settings.lin_solve_atol,
        settings.max_iter,
    )
    for _, fine_flux in solver:
        continue
    if settings.split is None:
        return fine_flux
    assemblies = system.geometry.array_refine(np.arange(system.geometry.cells), settings.split)
    volumes = system.geometry.array_refine(system.geometry.volumes, settings.split)
    normalized_fine_flux = fine_flux * np.repeat(core.geometry.volumes / volumes, core.E)
    assemblies_with_energy = np.vstack([assemblies * core.E + e for e in range(core.E)]).flatten(order="F").astype(int)
    return np.bincount(assemblies_with_energy, normalized_fine_flux)
