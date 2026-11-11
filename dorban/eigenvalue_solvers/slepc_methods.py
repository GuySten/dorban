from typing import Tuple

import numpy as np
import petsc4py.PETSc as petsc
import scipy.sparse as sparse
import slepc4py.SLEPc as slepc

_OPTIONS_PREFIX = "dorban_eps_"
# below this number of unknowns the linear problems are solved directly: algebraic multigrid gains nothing there and
# its setup can break down on very small problems
_DIRECT_SOLVE_SIZE = 20000


def _inner_solver_options(solver_name, lin_solver_name, pc_name, groups, lin_rtol, size) -> dict:
    """
    PETSc options for the linear solves inside the eigenvalue solver: the solves with M for Krylov-Schur,
    the correction equation for Jacobi-Davidson.
    """
    if solver_name == "gd":
        lin_solver_name = "preonly"
    if lin_solver_name is not None or pc_name is not None:
        options = {}
        if lin_solver_name is not None:
            options["st_ksp_type"] = lin_solver_name
        if pc_name is not None:
            options["st_pc_type"] = pc_name
        return options
    if size < _DIRECT_SOLVE_SIZE:
        return {"st_ksp_type": "preonly", "st_pc_type": "lu"}
    if groups is not None and groups > 1:
        # With only down-scatter, M is block lower triangular by energy group and solving one group after the
        # other is exact. Each group is a diffusion operator, solved with multigrid. With up-scatter this is a
        # preconditioner, and the outer flexible GMRES makes the solve exact.
        options = {
            "st_ksp_type": "fgmres",
            "st_pc_type": "fieldsplit",
            "st_pc_fieldsplit_type": "multiplicative",
            "st_pc_fieldsplit_block_size": str(groups),
        }
        for group in range(groups):
            options[f"st_fieldsplit_{group}_ksp_type"] = "gmres"
            options[f"st_fieldsplit_{group}_pc_type"] = "gamg"
            options[f"st_fieldsplit_{group}_ksp_rtol"] = str(lin_rtol / 100)
        return options
    return {"st_ksp_type": "gmres", "st_pc_type": "gamg"}


def generalized_eigenvalue_slepc(
    A: sparse.csr_matrix,
    M: sparse.csr_matrix,
    v: np.array = None,
    rtol_vector: float = 1e-10,
    lin_rtol=1e-12,
    max_iter=np.inf,
    solver_name="krylovschur",
    lin_solver_name=None,
    pc_name=None,
    groups=None,
    **kwargs,
) -> Tuple[float, np.array]:
    r"""
    interface to use SLEPc solvers for the eigenvalue problem Av=kMv

    Parameters
    ----------
    A: sparse.csr_matrix
     sparse matrix in csr format
    M: sparse.csr_matrix
     Invertible sparse matrix in csr format
    v: Optional[np.array]
     Initial guess for the eigenvector.
    rtol_vector: float
     Relative convergence tolerance for the eigenvector in :math:`l^\infty` norm
    lin_rtol: float
     Relative tolerance for the solution of :math:`Mx=b` when applying the :math:`M^{-1}A`
    max_iter: Optional[int]
     The maximal number of iterations allowed. If None, np.inf is used.
    solver_name: str
     the name of the EPS to use, for all possible option see the slepc manual.
     The default, krylovschur, converges to the eigenvalues of largest magnitude.
     jd (Jacobi-Davidson) is faster on some problems but can converge to another eigenvalue.
    lin_solver_name: str
     the name of KSP to use for solving the linear problems
     during the application of the eigenvalue solution
     algorithm, for a full list see the petsc manual,
     the useful ones are bcgsl, gmres, lgmres and preonly.
     If neither lin_solver_name nor pc_name is given, a default is chosen: a direct solve (LU) for fewer than
     20000 unknowns, otherwise see groups.
    pc_name: str
     the name of the preconditioner to use during the KSP application,
     if None then the default of the chosen KSP is used.
     For a full list see the petsc manual, some of them are
     jacobi, bjacobi, ilu, gamg and lu.
     If the KSP is set to preonly then it only applies the preconditioner
     this way direct solvers can be used, for example with pc_name=lu
    groups: Optional[int]
     The number of energy groups, when the unknowns are ordered by cell and then by energy group.
     If given (and neither lin_solver_name nor pc_name is), problems of 20000 unknowns or more are solved one energy
     group after the other, each with algebraic multigrid. Otherwise GMRES with algebraic multigrid is used.

    Returns
    -------
    Tuple[float, np.array]
     the maximal eigenvalue and its eigenvector
    """
    max_iter = None if max_iter == np.inf else max_iter
    options = _inner_solver_options(solver_name, lin_solver_name, pc_name, groups, lin_rtol, A.shape[0])
    database = petsc.Options()
    for key, value in options.items():
        database[_OPTIONS_PREFIX + key] = value
    solver = slepc.EPS()
    solver.create()
    sA = scipy_to_petcs(A)
    sM = scipy_to_petcs(M)
    try:
        if groups is not None and groups > 1:
            sM.setBlockSize(groups)
        solver.setOperators(sA, sM)
        solver.setType(solver_name)
        solver.setWhichEigenpairs(slepc.EPS.Which.LARGEST_MAGNITUDE)
        solver.setTolerances(rtol_vector, max_iter)
        if v is not None:
            solver.setInitialSpace(petsc.Vec().createWithArray(v))
        solver.getST().getKSP().setTolerances(lin_rtol, None)
        solver.setOptionsPrefix(_OPTIONS_PREFIX)
        solver.setFromOptions()
        solver.solve()
        if solver.getConverged() < 1:
            raise RuntimeError(
                f"SLEPc did not converge to an eigenvalue, converged reason {solver.getConvergedReason()}"
            )
    finally:
        for key in options:
            database.delValue(_OPTIONS_PREFIX + key)
    vr, vi = sA.createVecs()
    k = solver.getEigenpair(0, vr, vi)
    vector = vr.getArray().copy()
    for petsc_object in (vr, vi, sA, sM, solver):
        petsc_object.destroy()
    assert k == k.real
    # the sign of an eigenvector is arbitrary: return it with a positive sum
    return k.real, vector if np.sum(vector) >= 0 else -vector


def scipy_to_petcs(A: sparse.spmatrix):
    """
    function to create a petsc sparse matrix from a scipy sparse matrix
    """
    if A.format != "csr":
        A = A.tocsr()
    return petsc.Mat().createAIJ(size=A.shape, **{A.format: (A.indptr, A.indices, A.data)})


def _ksp_solve(sA, b, atol, rtol, solver_type, guess, max_it):
    """Solve sA x = b with one PETSc KSP; return the solution and PETSc's converged reason."""
    ksp = petsc.KSP().create()
    ksp.setOperators(sA, sA)
    ksp.setTolerances(rtol, atol, max_it=max_it)
    ksp.setType(solver_type)
    if guess is not None and np.any(guess):
        # PETSc starts from the contents of x, so the solution vector holds a copy of the guess.
        x = petsc.Vec().createWithArray(np.array(guess, dtype=petsc.ScalarType))
        ksp.setInitialGuessNonzero(True)
    else:
        x = petsc.Vec().create()
        x.setSizes(len(b))
        x.setFromOptions()
    vector = petsc.Vec().createWithArray(b)
    ksp.setFromOptions()
    ksp.solve(vector, x)
    reason = ksp.getConvergedReason()
    array = x.array.copy()
    ksp.destroy()
    vector.destroy()
    x.destroy()
    return array, reason


def solve_linear_petsc(
    A: sparse.spmatrix, b: np.array, atol=1e-14, rtol=1e-12, solver_type="bcgsl", guess=None, max_it=None
):
    """
    Solve A x = b with PETSc and return x.

    If the solver does not converge, the solve is repeated once with GMRES. BiCGStab(L), the default,
    breaks down when the preconditioner is an exact factorization, as ILU(0) is for small problems.
    If GMRES does not converge either, a RuntimeError is raised rather than returning an unconverged vector.
    """
    sA = scipy_to_petcs(A)
    try:
        array, reason = _ksp_solve(sA, b, atol, rtol, solver_type, guess, max_it)
        if reason < 0 and solver_type != "gmres":
            array, reason = _ksp_solve(sA, b, atol, rtol, "gmres", guess, max_it)
    finally:
        sA.destroy()
    if reason < 0:
        names = {v: k for k, v in vars(petsc.KSP.ConvergedReason).items() if isinstance(v, int)}
        raise RuntimeError(f"PETSc linear solve did not converge: {names.get(reason, reason)}")
    return array
