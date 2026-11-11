"""
tests for the triangular tessellation class
"""

import numpy as np
from pytest import fixture

from dorban.geometry.boundary_conditions import Void
from dorban.geometry.triangular_tessellation import (
    Triangle,
    TriangularTessellation,
    hex2triangles,
)


@fixture
def example_triangular():
    plane = hex2triangles([[Void()] * 6])
    return TriangularTessellation(
        6, 3.0, plane, [2, 2], boundary_conditions=(Void(), Void())
    )


def test_refinment(example_triangular):
    geo = example_triangular
    split = 2, np.array([1, 2])
    geo = geo.refine_mesh(split)
    assert geo.cells == 72
    assert np.all(geo.lengths == [2, 1, 1])


def test_reflection():
    tri = Triangle(3)
    assert tri.reflect_over_face(5, 1) == 2
    assert tri.reflect_over_face(3, 0) == 1
