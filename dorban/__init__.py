"""
DORBAN - Diffusion Of Reactors By Analyzing Nodes

DORBAN is a python code used for modeling nuclear reactors and solving the various
forms of the diffusion approximation of the reactor transport equation.

DORBAN allows users to model reactors with varying geometries including Cartesian and Triangular
geometries, and allows solving different types of problems, including the direct and
adjoint flux problems as well as source problems, given both as a source inside the
core and as boundary conditions.

The code is built to be modular as evident for example by the implementation of geometries.
A Geometry is implemented by a set of cells each having a list of neighboring cells as wells as
various geometric properties like volume, diameter and surface area.

DORBAN supports various method for application of the diffusion approximation as well
as discontinuity factors and directional diffusion coefficients.

DORBAN uses SLEPc in order to solve eigenvalue problems and via SLEPc it supports many
advanced numerical linear algebraic algorithms.


"""

from numba.np.unsafe.ndarray import *

from dorban.finite_differences.finite_difference_current_calculator import (
    CurrentCalculatorFD as CurrentCalculatorFD,
    DiscontinuityCurrentCalculator as DiscontinuityCurrentCalculator,
)
from dorban.geometry import (
    Boundary as Boundary,
    Cartesian as Cartesian,
    Cylindrical as Cylindrical,
    FiniteGeometry as FiniteGeometry,
    Reflector as Reflector,
    TriangularTessellation as TriangularTessellation,
    Void as Void,
)
from dorban.materials import Fissionable as Fissionable, Isotope as Isotope
from dorban.solve_equation import solve_k as solve_k
from dorban.system import Core as Core, mesh_refinement as mesh_refinement
