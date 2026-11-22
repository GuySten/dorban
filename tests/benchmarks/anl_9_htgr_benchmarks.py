"""
ANL benchmark book, problem 9-1: two dimensional hexagonal HTGR, 4 groups, solved with finite differences on
a triangular mesh. The problem is dorban.examples.anl_9_htgr, shared with docs/source/HTGR.ipynb.
"""

import pytest

from dorban.examples import anl_9_htgr
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k


@pytest.fixture(scope="module")
def solution():
    system = anl_9_htgr.core()
    k, flux = solve_k(system, settings=FDSettings(split=system.geometry.uniform_split(1)))
    return {"k": k, "flux": flux}


def test_k(solution):
    assert solution["k"] == pytest.approx(anl_9_htgr.REFERENCE_K, abs=1e-4)
