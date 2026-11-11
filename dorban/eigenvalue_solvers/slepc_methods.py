from typing import Tuple

import numpy as np
import petsc4py.PETSc as petsc
import scipy.sparse as sparse
import slepc4py.SLEPc as slepc


def generalized_eigenvalue_slepc(
    A: sparse.csr_matrix,
    M: sparse.csr_matrix,
    v: np.array = None,
    rtol_vector: float = 1e-10,
    lin_rtol=1e-12,
    max_iter=np.inf,
    solver_name="jd",
    lin_solver_name="bcgsl",
    pc_name=None,
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
     the name of the EPS to use, for all possible option see
     the slepc manual, the useful ones are krylovschur and jd
    lin_solver_name: str
     the name of KSP to use for solving the linear problems
     during the application of the eigenvalue solution
     algorithm, for a full list see the petsc manual,
     the useful ones are bcgsl,lgmres and preonly
    pc_name: str
     the name of the preconditioner to use during the KSP application,
     if None then the default of the chosen KSP is used.
     For a full list see the petsc manual, some of them are
     jacobi, bjacobi, ilu and lu.
     If the KSP is set to preonly then it only applies the preconditioner
     this way direct solvers can be used, for example with pc_name=lu

    Returns
    -------
    Tuple[float, np.array]
     the maximal eigenvalue and its eigenvector
    """
    max_iter = None if max_iter == np.inf else max_iter
    if solver_name == "gd":
        lin_solver_name = "preonly"
    solver = slepc.EPS()
    solver.create()
    sA = scipy_to_petcs(A)
    sM = scipy_to_petcs(M)
    solver.setOperators(sA, sM)
    solver.setType(solver_name)
    solver.setTolerances(rtol_vector, max_iter)
    if v is not None:
        solver.setInitialSpace(petsc.Vec().createWithArray(v))
    ksp = solver.getST().getKSP()
    ksp.setType(lin_solver_name)
    ksp.setTolerances(lin_rtol, None)
    solver.solve()
    vr, vi = sA.getVecs()
    k = solver.getEigenpair(0, vr, vi)
    sA.destroy()
    sM.destroy()
    ksp.destroy()
    solver.destroy()
    assert k == k.real
    return k.real, vr.getArray()


def scipy_to_petcs(A: sparse.spmatrix):
    """
    function to create a petsc sparse matrix from a scipy sparse matrix
    """
    if A.format != "csr":
        A = A.tocsr()
    return petsc.Mat().createAIJ(size=A.shape, **{A.format: (A.indptr, A.indices, A.data)})


def solve_linear_petsc(A: sparse.spmatrix, b: np.array, atol=1e-14, rtol=1e-12, solver_type="bcgsl", guess=None):
    sA = scipy_to_petcs(A)
    ksp = petsc.KSP().create()
    ksp.setOperators(sA, sA)
    ksp.setTolerances(rtol, atol)
    if np.any(guess):
        ksp.setInitialGuessNonzero(petsc.Vec().createWithArray(guess))
    ksp.setType(solver_type)
    x = petsc.Vec().create()
    x.setSizes(len(b))
    x.setFromOptions()
    vector = petsc.Vec().createWithArray(b)
    ksp.setFromOptions()
    ksp.solve(vector, x)
    array = x.array
    sA.destroy()
    ksp.destroy()
    vector.destroy()
    x.destroy()
    return array
