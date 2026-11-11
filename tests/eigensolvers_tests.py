"""
tests for the various eigensolvers
"""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import csr_matrix

from dorban.eigenvalue_solvers.arnoldi_arpack import generalized_arnoldi, shift_invert_arnoldi
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


def _laplacian_2d(n):
    """5-point Laplacian on an n x n grid; ILU(0) is not an exact factorization of it."""
    tri = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    return sparse.csr_matrix(sparse.kronsum(tri, tri))


def test_solve_linear_petsc_uses_initial_guess():
    # With the exact solution as the guess, the initial residual is already below the
    # tolerance, so the solver must return the guess rather than restart from zero.
    mat = _laplacian_2d(10)
    exact = np.linspace(1, 2, mat.shape[0])
    rhs = mat @ exact
    guess = exact.copy()
    result = solve_linear_petsc(mat, rhs, atol=0, rtol=0.5, guess=guess)
    assert np.allclose(result, exact, rtol=0, atol=1e-12)
    assert np.array_equal(guess, exact)


def test_solve_linear_petsc_recovers_from_bcgsl_breakdown():
    # Every entry is nonzero, so ILU(0) is an exact factorization; BiCGStab(L) then breaks down
    # after one iteration (DIVERGED_DTOL). The solver must not return that unconverged vector.
    rng = np.random.default_rng(0)
    mat = sparse.csr_matrix(np.diag(np.arange(1.0, 9)) + 0.1 * rng.random((8, 8)))
    rhs = np.ones(8)
    result = solve_linear_petsc(mat, rhs)
    assert np.allclose(mat @ result, rhs, rtol=0, atol=1e-10)


def test_solve_linear_petsc_raises_when_not_converged():
    mat = _laplacian_2d(10)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_linear_petsc(mat, np.ones(mat.shape[0]), max_it=1)


@pytest.mark.parametrize(
    "solver", [generalized_eigenvalue, generalized_arnoldi, shift_invert_arnoldi, generalized_eigenvalue_slepc]
)
@pytest.mark.parametrize("negative_guess", [False, True])
def test_eigenvector_sign_is_positive(solver, negative_guess):
    # the dominant eigenvector of a positive matrix is positive (Perron-Frobenius), but an eigensolver returns it
    # with an arbitrary sign; the solvers return it with a positive sum
    rng = np.random.default_rng(1)
    A = csr_matrix(rng.random((30, 30)))
    M = csr_matrix(np.diag(np.arange(1.0, 31.0)) * 3)
    guess = -np.ones(30) if negative_guess else None
    _, v = solver(A, M, v=guess)
    assert np.all(v > 0)
