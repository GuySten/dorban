"""
tests for the construction of cross section data
"""

import numpy as np

from dorban.materials import Isotope


def test_isotope_generation():
    absorb = np.array([1, 2, 3])
    scatter = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kappa = np.array([2, 3, 5])
    mat1 = Isotope("1", scatter, absorb)
    assert (
        np.all(mat1.absorb == absorb)
        and np.all(mat1.scatter == scatter)
        and np.all(mat1.kappa == np.zeros_like(absorb))
    )
    mat2 = Isotope("1", scatter, absorb, kappa=kappa)
    assert np.all(mat2.kappa == kappa)
