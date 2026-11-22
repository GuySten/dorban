"""
tests for the various eigensolvers
"""

import numpy as np
import petsc4py.PETSc as petsc
import pytest
import scipy.linalg
from scipy import sparse
from scipy.sparse import csr_matrix

from dorban.eigenvalue_solvers import slepc_methods
from dorban.eigenvalue_solvers.arnoldi_arpack import generalized_arnoldi, shift_invert_arnoldi
from dorban.eigenvalue_solvers.power_iteration import generalized_eigenvalue
from dorban.eigenvalue_solvers.slepc_methods import (
    _petsc_options,
    energy_group_blocks,
    generalized_eigenvalue_slepc,
    linear_solve_options,
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


def _slab_matrices(scatter, cells=200, width=1.0, dim=1):
    """
    Finite differences matrices A (production) and M (loss) of a homogeneous four group slab (dim=1) or cube (dim=3)
    of cells along each axis, with zero flux outside, ordered by cell and then by energy group. scatter[i, j] is the
    scattering from group j to group i.
    """
    absorb = np.array([0.005, 0.01, 0.03, 0.08])
    diffusion = np.array([1.8, 1.2, 0.8, 0.4])
    production = np.outer([1, 0, 0, 0], [0.003, 0.004, 0.03, 0.08])
    laplacian = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(cells, cells)) / width**2
    for _ in range(dim - 1):
        laplacian = sparse.kronsum(laplacian, sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(cells, cells)))
    cells = laplacian.shape[0]
    removal = np.diag(absorb + scatter.sum(axis=0)) - scatter
    M = sparse.kron(laplacian, np.diag(diffusion)) + sparse.kron(sparse.identity(cells), removal)
    A = sparse.kron(sparse.identity(cells), production)
    return A.tocsr(), M.tocsr()


DOWN_SCATTER = np.array([[0, 0, 0, 0], [0.02, 0, 0, 0], [0, 0.015, 0, 0], [0, 0, 0.05, 0]])
UP_SCATTER = DOWN_SCATTER + np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.01], [0, 0, 0, 0]])


@pytest.mark.parametrize(
    "scatter, adjoint, blocks",
    [
        (DOWN_SCATTER, False, [[0], [1], [2], [3]]),
        (DOWN_SCATTER, True, [[3], [2], [1], [0]]),
        (UP_SCATTER, False, [[0], [1], [2, 3]]),
        (UP_SCATTER, True, [[2, 3], [1], [0]]),
    ],
)
def test_energy_group_blocks_follow_the_scattering(scatter, adjoint, blocks):
    # M is block lower triangular in the order of the blocks, and groups coupled by up-scatter are solved together
    _, M = _slab_matrices(scatter, cells=5)
    assert energy_group_blocks(M.T if adjoint else M, 4) == blocks


@pytest.mark.parametrize("scatter", [DOWN_SCATTER, UP_SCATTER], ids=["down-scatter", "up-scatter"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_group_by_group_solve_finds_the_eigenvalue(scatter, adjoint):
    # a three dimensional problem of 18 x 18 x 18 cells, 23,328 unknowns, large enough to be solved group by group with
    # algebraic multigrid by default; the reference is a direct solve
    A, M = _slab_matrices(scatter, cells=18, dim=3)
    if adjoint:
        A, M = A.T.tocsr(), M.T.tocsr()
    options = linear_solve_options("krylovschur", None, None, 4, 1e-12, M)
    assert (options["st_pc_type"], options["st_fieldsplit_0_pc_type"]) == ("fieldsplit", "gamg")
    direct, _ = generalized_eigenvalue_slepc(A, M, groups=4, lin_solver_name="preonly", pc_name="lu")
    k, _ = generalized_eigenvalue_slepc(A, M, groups=4)
    assert k == pytest.approx(direct, rel=1e-9)


def test_slepc_finds_the_eigenvalue_when_only_a_preconditioner_is_given():
    # pc_name alone used to leave SLEPc's default Krylov method, preonly, which applies the preconditioner once
    # instead of solving: k came out 0.00043 instead of 0.0106. It must be used with GMRES.
    A, M = _slab_matrices(DOWN_SCATTER)
    exact = np.max(np.abs(scipy.linalg.eigvals(A.toarray(), M.toarray())))
    k, _ = generalized_eigenvalue_slepc(A, M, groups=4, pc_name="jacobi")
    assert k == pytest.approx(exact, rel=1e-9)


def test_generalized_davidson_rejects_a_linear_solver():
    # Generalized Davidson only applies the preconditioner, a linear solver would be ignored
    A, M = _slab_matrices(DOWN_SCATTER, cells=5)
    with pytest.raises(ValueError, match="gd"):
        generalized_eigenvalue_slepc(A, M, solver_name="gd", lin_solver_name="gmres")


def test_petsc_options_are_restored():
    database = petsc.Options()
    database["test_dorban_kept"] = "user value"
    with pytest.raises(RuntimeError):
        with _petsc_options({"kept": "ours", "added": "ours"}, prefix="test_dorban_"):
            assert database.getString("test_dorban_kept") == "ours"
            raise RuntimeError
    assert database.getString("test_dorban_kept") == "user value"
    assert not database.hasName("test_dorban_added")
    database.delValue("test_dorban_kept")


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


@pytest.mark.parametrize("groups", [None, 1])
def test_slepc_solves_a_tiny_problem(groups):
    # the setup of algebraic multigrid broke down on this 2 x 2 problem (from a NEM first guess): the short GMRES with
    # which it estimates an eigenvalue broke down exactly. It is solved directly, as small problems are by default
    A = csr_matrix(np.diag([1.5, 1.5]))
    M = csr_matrix(np.array([[2.25, -1.0], [-1.0, 4.596]]))
    k, v = generalized_eigenvalue_slepc(A, M, groups=groups)
    expected = np.max(np.linalg.eigvals(np.linalg.solve(M.toarray(), A.toarray())))
    assert k == pytest.approx(expected, rel=1e-10)
    assert np.allclose(A @ v, k * (M @ v))


def test_slepc_solves_directly_when_the_default_linear_solver_fails(monkeypatch):
    # a default linear solver that fails to set up, whatever the problem: PETSc has no such preconditioner
    monkeypatch.setattr(slepc_methods, "linear_solve_options", lambda *args: {"st_pc_type": "no_such_preconditioner"})
    A, M = _slab_matrices(DOWN_SCATTER, cells=20)
    exact = np.max(np.abs(scipy.linalg.eigvals(A.toarray(), M.toarray())))
    with pytest.warns(RuntimeWarning, match="directly"):
        k, _ = generalized_eigenvalue_slepc(A, M, groups=4)
    assert k == pytest.approx(exact, rel=1e-9)


def test_slepc_raises_when_the_linear_solver_given_fails():
    # the linear solver or preconditioner the user gave is not replaced
    A, M = _slab_matrices(DOWN_SCATTER, cells=20)
    with pytest.raises((petsc.Error, SystemError)):
        generalized_eigenvalue_slepc(A, M, groups=4, pc_name="no_such_preconditioner")
