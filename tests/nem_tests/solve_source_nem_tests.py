"""
tests for the NEM source solver against a fine-mesh finite differences reference
"""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from dorban import Cartesian, Core, CurrentCalculatorFD, Fissionable, Isotope, Void
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.solve_nem import NEMSettings, solve_source_nem
from dorban.solve_equation import fixed_source

E = 2
MATERIAL_DATA = dict(
    scatter=np.array([[0, 0], [0.02, 0]]),
    absorb=np.array([0.01, 0.08]),
    diffusion=np.array([1.5, 0.4]),
)
FUEL = Fissionable("Fuel", nusigmaf=np.array([0.004, 0.06]), chi=np.array([1, 0]), **MATERIAL_DATA)


def fine_mesh_reference(geometry, material, source_density, refinement=12):
    """Finite differences on a refined mesh, volume averaged back onto the cells of geometry."""
    split = geometry.uniform_split(refinement)
    fine_geometry = geometry.refine_mesh(split)
    materials = [material] * fine_geometry.cells
    core = Core(materials, E, fine_geometry, CurrentCalculatorFD.from_isotopes(materials))
    fine_source = source_density * np.repeat(fine_geometry.volumes, E)
    fine_flux = fixed_source(core, fine_source, solver=lambda A, b: (spsolve(A.tocsc(), b), 0))[0]
    owner = geometry.array_refine(np.arange(geometry.cells), split).astype(int)
    flux = np.zeros(geometry.cells * E)
    for e in range(E):
        np.add.at(flux, owner * E + e, fine_flux[e::E] * fine_geometry.volumes)
    return flux / np.repeat(geometry.volumes, E)


@pytest.mark.parametrize("material", [FUEL], ids=["fissionable"])
def test_solve_source_nem_agrees_with_fine_mesh(material):
    geometry = Cartesian([np.full(6, 3.0), np.full(6, 3.0)], [Void()] * 4)
    materials = [material] * geometry.cells
    core = Core(materials, E, geometry, CMFDCurrentCalculator.from_isotopes(materials, 2))
    # the source is the total source in each cell, a density of 1 per unit volume
    source = np.repeat(geometry.volumes, E)
    flux = solve_source_nem(core, NEMSettings(max_iter=100), source)
    reference = fine_mesh_reference(geometry, material, 1.0)
    assert np.max(np.abs(flux - reference) / reference) < 0.07
