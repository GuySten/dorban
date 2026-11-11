"""
tests for the face arrays of the geometries, which the finite difference and NEM matrices are built from: the
vectorized face_distances and face_areas of each geometry against its distance_to_face and surface_area, face by face
"""

import numpy as np
import pytest

from dorban.finite_differences.diffusion_operator import diffusion_matrix
from dorban.finite_differences.finite_difference_current_calculator import (
    CurrentCalculatorFD,
    DirectionalCurrentCalculator,
)
from dorban.geometry.boundary_conditions import ExtrapolationLength, Reflector, Void, ZeroFlux
from dorban.geometry.cartesian import Cartesian
from dorban.geometry.cylindrical import Cylindrical
from dorban.geometry.geometry import BOUNDARY, NO_FACE, FiniteGeometry
from dorban.geometry.spherical import Spherical
from dorban.geometry.triangular_tessellation import TriangularTessellation, hex2triangles

RNG = np.random.default_rng(0)


def _geometries():
    yield "Cartesian 1D", Cartesian([RNG.uniform(0.5, 2, 5)], [Reflector(), Void()])
    yield (
        "Cartesian 2D with black absorber",
        Cartesian(
            [RNG.uniform(0.5, 2, 5), RNG.uniform(0.5, 2, 4)], [Reflector(), ZeroFlux(), Void(), Reflector()], [0, 7, 8]
        ),
    )
    cube = Cartesian([RNG.uniform(0.5, 2, n) for n in (3, 4, 2)], [Reflector(), ExtrapolationLength()] * 3, [5])
    yield "Cartesian 3D", cube
    yield "PolyBox", cube.to_polybox()
    yield (
        "Cylindrical R-Z",
        Cylindrical([RNG.uniform(0.5, 2, 4), RNG.uniform(0.5, 2, 3)], [Reflector(), Void(), ZeroFlux(), Void()], [3]),
    )
    yield "Cylindrical R", Cylindrical([RNG.uniform(0.5, 2, 4)], [Reflector(), Void()])
    yield "Triangular 2D", TriangularTessellation(6, 3.0, hex2triangles([[Void()] * 6]))
    yield (
        "Triangular 3D",
        TriangularTessellation(6, 3.0, hex2triangles([[Void()] * 6]), [2.0, 3.0], boundary_conditions=(Void(), Void())),
    )
    yield "Spherical", Spherical(RNG.uniform(0.5, 2, 4), Void(), 3)


GEOMETRIES = dict(_geometries())


@pytest.mark.parametrize("name", GEOMETRIES)
def test_face_arrays_match_the_face_by_face_values(name):
    geometry = GEOMETRIES[name]
    neighbors = geometry.face_neighbors()
    by_face = FiniteGeometry.face_distances(geometry), FiniteGeometry.face_areas(geometry)
    has_face = neighbors != NO_FACE
    assert np.allclose(geometry.face_distances()[has_face], by_face[0][has_face])
    assert np.allclose(geometry.face_areas()[has_face], by_face[1][has_face])
    for cell, row in enumerate(geometry.neighbors):
        for face, neighbor in enumerate(row):
            assert neighbors[cell, face] == (neighbor if isinstance(neighbor, (int, np.integer)) else BOUNDARY)


def test_spherical_center_cell_has_one_face():
    neighbors = GEOMETRIES["Spherical"].face_neighbors()
    assert neighbors[0, 1] == NO_FACE and np.all(neighbors[1:] != NO_FACE)


@pytest.mark.parametrize("dim", [2, 3])
def test_isotropic_directional_diffusion_is_the_scalar_one(dim):
    # reflective boundaries: the vacuum boundary conditions take scalar diffusion coefficients
    geometry = Cartesian([RNG.uniform(0.5, 2, n) for n in (4, 3, 2)[:dim]], [Reflector()] * (2 * dim))
    dc = RNG.uniform(0.5, 2, (geometry.cells, 2))
    tensor = dc[:, :, None, None] * np.eye(dim)
    scalar = diffusion_matrix(geometry, 2, CurrentCalculatorFD(dc))
    directional = diffusion_matrix(geometry, 2, DirectionalCurrentCalculator(tensor))
    assert abs(scalar - directional).max() < 1e-12
