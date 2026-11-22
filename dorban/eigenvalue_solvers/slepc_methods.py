"""
Eigenvalue and linear solvers from SLEPc and PETSc.

The linear solves inside the eigenvalue solver are configured through PETSc's options database, under the prefix
``_OPTIONS_PREFIX``. The names of the options are PETSc's and SLEPc's:

``st_``
    the spectral transformation of SLEPc, which owns the linear problems solved with M
``ksp``
    the Krylov subspace method that solves them (gmres, fgmres, or preonly: apply the preconditioner once)
``pc``
    the preconditioner (lu is a direct solve, gamg algebraic multigrid, fieldsplit splits the unknowns)
``fieldsplit_<i>_``
    the solver of the i-th part of a fieldsplit preconditioner
"""

import warnings
from contextlib import contextmanager

import numpy as np
import petsc4py.PETSc as petsc
import scipy.sparse as sparse
import slepc4py.SLEPc as slepc

_OPTIONS_PREFIX = "dorban_eps_"
# below this number of unknowns the linear problems are solved directly, which is faster there than algebraic
# multigrid. Measured in three dimensions with 2, 4 and 7 groups, group by group multigrid was faster from about
# 15,000 unknowns (7 groups: 0.12 s against 0.39 s), the two were within 0.05 s at 7,000 unknowns with 2 and 4 groups.
_DIRECT_SOLVE_SIZE = 10000
# problems whose cells have at most this many neighbors, two dimensional ones, are solved directly at any size
_DIRECT_SOLVE_NEIGHBORS = 4


@contextmanager
def _petsc_options(options: dict, prefix: str = _OPTIONS_PREFIX):
    """
    Set options in PETSc's global options database for the duration of the block, then restore what was there.
    """
    database = petsc.Options()
    previous = {}
    try:
        for key, value in options.items():
            name = prefix + key
            previous[name] = database.getString(name) if database.hasName(name) else None
            database[name] = value
        yield
    finally:
        for name, value in previous.items():
            database.delValue(name)
            if value is not None:
                database[name] = value


def spatial_neighbors(M: sparse.spmatrix, groups: int | None) -> int:
    """
    The largest number of neighbors of a cell in the matrix M, ordered by cell and then by energy group: the number
    of off diagonal entries of a row in the same energy group. It is at most 2 in one dimension, 4 in two (3 for
    triangles) and 6 in three (5 for triangular prisms). Every energy group has the same spatial coupling (neutrons
    change group where they are), so the first group is counted.
    """
    groups = groups or 1
    M = M.tocoo()
    first_group = (M.row % groups == 0) & (M.col % groups == 0) & (M.row != M.col) & (M.data != 0)
    counts = np.bincount(M.row[first_group] // groups, minlength=M.shape[0] // groups)
    return int(counts.max()) if counts.size else 0


def energy_group_blocks(M: sparse.spmatrix, groups: int) -> list[list[int]]:
    """
    The blocks of energy groups of M, in the order in which to solve them.

    M is the matrix of the whole problem, in space and energy: its unknowns are the flux of each cell in each energy
    group, numbered by cell and then by energy group (unknown ``cell * groups + group``), so its shape is
    (cells * groups, cells * groups). The energy coupling is read from the entries between groups of the same cell.

    Group j is coupled to group i in M when neutrons scatter from j to i. Groups that are coupled in both
    directions, through up-scatter, are put in one block and solved together. The blocks are ordered so that each
    block only receives neutrons from the blocks before it: then M is block lower triangular in this order, and
    solving one block after the other is exact. With only down-scatter every group is a block, from the fastest to
    the slowest; in the adjoint problem the order is reversed.
    """
    M = M.tocoo()
    same_cell = (M.row // groups == M.col // groups) & (M.row != M.col) & (M.data != 0)
    reaches = np.eye(groups, dtype=bool)  # reaches[j, i]: neutrons from group j reach group i
    reaches[M.col[same_cell] % groups, M.row[same_cell] % groups] = True
    for k in range(groups):  # transitive closure
        reaches |= np.outer(reaches[:, k], reaches[k, :])
    blocks = {tuple(np.flatnonzero(reaches[g] & reaches[:, g]).tolist()) for g in range(groups)}
    # a block that reaches another is reached by fewer groups, so this is an order of solution; blocks that do not
    # reach each other are ordered by their groups, to always give the same order
    return [list(block) for block in sorted(blocks, key=lambda block: (reaches[:, block[0]].sum(), block))]


def _explicit_options(eps_type: str, ksp_type: str | None, pc_type: str | None) -> dict:
    """The linear solver and preconditioner the user asked for."""
    if eps_type == "gd":
        # Generalized Davidson only applies the preconditioner, it does not solve linear problems
        if ksp_type not in (None, "preonly"):
            raise ValueError(
                f"solver_name='gd' only applies the preconditioner, lin_solver_name={ksp_type!r} is invalid"
            )
        ksp_type = "preonly"
    elif ksp_type is None and pc_type is not None:
        # SLEPc would apply the preconditioner alone, a far from exact solve unless it is a direct one
        ksp_type = "gmres"
    options = {"st_ksp_type": ksp_type}
    if pc_type is not None:
        options["st_pc_type"] = pc_type
    return options


def _fieldsplit_options(blocks: list[list[int]], groups: int, outer_ksp_type: str) -> dict:
    """A multiplicative fieldsplit (block Gauss-Seidel) over the blocks of energy groups, in their order."""
    options = {
        "st_ksp_type": outer_ksp_type,
        "st_pc_type": "fieldsplit",
        "st_pc_fieldsplit_type": "multiplicative",
        "st_pc_fieldsplit_block_size": str(groups),
    }
    for i, block in enumerate(blocks):
        options[f"st_pc_fieldsplit_{i}_fields"] = ",".join(map(str, block))
    return options


def _direct_solve_options(blocks: list[list[int]], groups: int) -> dict:
    """
    LU factorization of each block of energy groups, solved one block after the other in the order of
    energy_group_blocks. Each block is solved exactly, so this is an exact solve (preonly), without an outer iteration.
    The factors of the blocks have 1.5 to 2 times fewer nonzeros than those of all of M together, and the eigenvalue
    solve was 1.6 to 6 times faster on the problems measured (ANL 2-A1, 7 groups, 114,000 unknowns: 2.35 s -> 1.36 s).
    """
    if len(blocks) == 1:
        return {"st_ksp_type": "preonly", "st_pc_type": "lu"}
    options = _fieldsplit_options(blocks, groups, "preonly")
    for i in range(len(blocks)):
        options[f"st_fieldsplit_{i}_ksp_type"] = "preonly"
        options[f"st_fieldsplit_{i}_pc_type"] = "lu"
    return options


def _group_by_group_options(blocks: list[list[int]], groups: int, lin_rtol: float) -> dict:
    """
    Solve the blocks of energy groups one after the other (multiplicative fieldsplit, block Gauss-Seidel), each with
    GMRES and algebraic multigrid. In the order of energy_group_blocks this is exact, but each block is solved
    iteratively, so the outer solve is flexible GMRES, and the blocks are solved more accurately than the outer solve.
    """
    options = _fieldsplit_options(blocks, groups, "fgmres")
    for i in range(len(blocks)):
        options[f"st_fieldsplit_{i}_ksp_type"] = "gmres"
        options[f"st_fieldsplit_{i}_pc_type"] = "gamg"
        options[f"st_fieldsplit_{i}_ksp_rtol"] = str(lin_rtol / 100)
    return options


def linear_solve_options(
    eps_type: str, ksp_type: str | None, pc_type: str | None, groups: int | None, lin_rtol: float, M: sparse.spmatrix
) -> dict:
    """
    PETSc options for the linear problems solved inside the eigenvalue solver: the solves with M for Krylov-Schur,
    the correction equation for Jacobi-Davidson. The first that applies of:

    1. the linear solver or preconditioner the user gave (ksp_type, pc_type),
    2. a direct solve, block by block (_direct_solve_options), for fewer than _DIRECT_SOLVE_SIZE unknowns, or when no
       cell has more than _DIRECT_SOLVE_NEIGHBORS neighbors (two dimensional problems),
    3. group by group (_group_by_group_options), when there are several energy groups and they are not all coupled
       by up-scatter,
    4. GMRES with algebraic multigrid, on all the energy groups at once.

    M is the matrix of the linear problems, numbered by cell and then by energy group (see energy_group_blocks).

    A direct solve suits small problems and two dimensional ones, where the fill of the factors stays moderate: on two
    dimensional problems of 80,000 to 456,000 unknowns it was 2 to 7 times faster than group by group multigrid
    (ANL 2-A1, 456,000 unknowns: 13.6 s against 27.3 s), except on a 7 group problem with a block of 3 groups coupled by
    up-scatter, where it took 1.6 s against 0.9 s. In three dimensions the fill grows fast (7 groups, 109,000
    unknowns: 64 s against 1.1 s), so three dimensional problems are solved iteratively above _DIRECT_SOLVE_SIZE.
    """
    if eps_type == "gd" or ksp_type is not None or pc_type is not None:
        return _explicit_options(eps_type, ksp_type, pc_type)
    groups = groups or 1
    blocks = energy_group_blocks(M, groups) if groups > 1 else [[0]]
    if M.shape[0] < _DIRECT_SOLVE_SIZE or spatial_neighbors(M, groups) <= _DIRECT_SOLVE_NEIGHBORS:
        return _direct_solve_options(blocks, groups)
    if len(blocks) > 1:
        return _group_by_group_options(blocks, groups, lin_rtol)
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
) -> tuple[float, np.array]:
    r"""
    interface to use SLEPc solvers for the eigenvalue problem Av=kMv

    Parameters
    ----------
    A: sparse.csr_matrix
     sparse matrix in csr format
    M: sparse.csr_matrix
     Invertible sparse matrix in csr format
    v: np.array | None
     Initial guess for the eigenvector.
    rtol_vector: float
     Relative convergence tolerance for the eigenvector in :math:`l^\infty` norm
    lin_rtol: float
     Relative tolerance for the solution of :math:`Mx=b` when applying the :math:`M^{-1}A`
    max_iter: int | None
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
     If neither lin_solver_name nor pc_name is given, a default is chosen: a direct solve (LU, one block of energy
     groups at a time) for fewer than 10000 unknowns or when no cell has more than 4 neighbors (two dimensional
     problems), otherwise see groups.
     With solver_name='gd' it must be None or preonly: Generalized Davidson only applies the preconditioner.
    pc_name: str
     the name of the preconditioner to use during the KSP application,
     if None then the default of the chosen KSP is used.
     For a full list see the petsc manual, some of them are
     jacobi, bjacobi, ilu, gamg and lu.
     If it is given without lin_solver_name, the KSP is gmres.
     If the KSP is set to preonly then it only applies the preconditioner
     this way direct solvers can be used, for example with lin_solver_name=preonly and pc_name=lu
    groups: int | None
     The number of energy groups, when the unknowns are ordered by cell and then by energy group. It describes M
     and does not choose a solver: lin_solver_name and pc_name, when given, take precedence over the defaults it
     enables. Those solve the energy groups one after the other, with the groups coupled by up-scatter together
     (see energy_group_blocks): directly, or for three dimensional problems of 10000 unknowns or more with
     algebraic multigrid. It is also needed to count the neighbors of the cells. Without groups, all the groups are
     solved together, directly or with GMRES and algebraic multigrid.

    Returns
    -------
    tuple[float, np.array]
     the maximal eigenvalue and its eigenvector
    """
    max_iter = None if max_iter == np.inf else max_iter
    options = linear_solve_options(solver_name, lin_solver_name, pc_name, groups, lin_rtol, M)
    try:
        k, vector = _eigenpair(A, M, v, rtol_vector, lin_rtol, max_iter, solver_name, groups, options)
    except (petsc.Error, SystemError) as error:
        # petsc4py raises a PETSc error from a solve as a SystemError caused by it
        petsc_error = error if isinstance(error, petsc.Error) else error.__cause__
        chosen_by_user = solver_name == "gd" or lin_solver_name is not None or pc_name is not None
        direct = _direct_solve_options(energy_group_blocks(M, groups) if groups and groups > 1 else [[0]], groups or 1)
        if not isinstance(petsc_error, petsc.Error) or chosen_by_user or options == direct:
            raise
        # The setup of algebraic multigrid can break down: on a 2 x 2 problem, the short GMRES with which it
        # estimates an eigenvalue to smooth its interpolation broke down exactly (PETSc error 91, "tt == 0.0").
        warnings.warn(
            f"The default linear solver of the eigenvalue problem failed (PETSc error {petsc_error.ierr}), "
            "solving the linear problems directly instead",
            RuntimeWarning,
        )
        k, vector = _eigenpair(A, M, v, rtol_vector, lin_rtol, max_iter, solver_name, groups, direct)
    # the sign of an eigenvector is arbitrary: return it with a positive sum
    return k, vector if np.sum(vector) >= 0 else -vector


def _eigenpair(
    A: sparse.spmatrix,
    M: sparse.spmatrix,
    v: np.ndarray | None,
    rtol_vector: float,
    lin_rtol: float,
    max_iter: int | None,
    solver_name: str,
    groups: int | None,
    options: dict,
) -> tuple[float, np.array]:
    """The dominant eigenpair of Av=kMv, with the given PETSc options for the linear solves."""
    sA = scipy_to_petcs(A)
    sM = scipy_to_petcs(M)
    if groups is not None and groups > 1:
        sM.setBlockSize(groups)
    solver = slepc.EPS().create()
    try:
        solver.setOperators(sA, sM)
        solver.setType(solver_name)
        solver.setWhichEigenpairs(slepc.EPS.Which.LARGEST_MAGNITUDE)
        solver.setTolerances(rtol_vector, max_iter)
        if v is not None:
            solver.setInitialSpace(petsc.Vec().createWithArray(v))
        solver.getST().getKSP().setTolerances(lin_rtol, None)
        solver.setOptionsPrefix(_OPTIONS_PREFIX)
        # the options of the linear solves are read when the solve sets up its solvers
        with _petsc_options(options):
            solver.setFromOptions()
            solver.solve()
        if solver.getConverged() < 1:
            raise RuntimeError(
                f"SLEPc did not converge to an eigenvalue, converged reason {solver.getConvergedReason()}"
            )
        vr, vi = sA.createVecs()
        k = solver.getEigenpair(0, vr, vi)
        vector = vr.getArray().copy()
        vr.destroy()
        vi.destroy()
    finally:
        for petsc_object in (sA, sM, solver):
            petsc_object.destroy()
    assert k == k.real
    return k.real, vector


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
