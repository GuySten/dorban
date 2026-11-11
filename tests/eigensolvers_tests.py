"""
tests for the various eigensolvers
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

from dorban.eigenvalue_solvers.arnoldi_arpack import generalized_arnoldi
from dorban.eigenvalue_solvers.power_iteration import generalized_eigenvalue
from dorban.eigenvalue_solvers.slepc_methods import (
    generalized_eigenvalue_slepc,
    scipy_to_petcs,
    solve_linear_petsc,
)


def test_generalized_eigenvalue():
    A = np.diag([1.0, 3.0, 53.0, 59.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    M = np.diag([7.0, 6.0, 4.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    k, v = generalized_eigenvalue(A, M, rtol_vector=1e-5, atol_vector=1e-10)
    k1, v1 = generalized_arnoldi(A, M)
    k2, v2 = generalized_eigenvalue_slepc(csr_matrix(A), csr_matrix(M))
    v /= np.sum(v)
    v1 /= np.sum(v1)
    v2 /= np.sum(v2)
    assert np.isclose(k, 13.25)
    assert np.isclose(k1.real, 13.25)
    assert np.isclose(k2, 13.25)
    assert np.allclose(v1, v)
    assert np.allclose(v2, v)


def test_scipy_to_petsc():
    mat = sparse.csr_matrix(np.ones((10, 10)))
    scipy_to_petcs(mat)


def test_solver():
    mat1 = sparse.csr_matrix(np.diag(np.arange(1, 11)))
    mat2 = sparse.csr_matrix(np.diag(np.arange(101, 111)))
    generalized_eigenvalue_slepc(mat1, mat2)
    solve_linear_petsc(mat1, np.ones(10), guess=np.ones(10))
