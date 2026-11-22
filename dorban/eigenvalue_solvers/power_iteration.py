r"""
This module contains an implementation of the power iteration algorithms for the
solution of the generalized eigenvaluelue problem :math:`Av=kMv`.
"""

from typing import Generator, Optional, Union

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as la
from more_itertools import last


def generalized_eigenvalue(
    A: Union[sparse.spmatrix, np.array, la.LinearOperator],
    M: Union[sparse.spmatrix, np.array, la.LinearOperator],
    k: float = 1,
    v: Optional[np.array] = None,
    **kwargs,
) -> tuple[float, np.array]:
    r"""
    solves the generalized eigenvalue problem :math:`Av=kMv`.
    Finds the largest eigenvalue satisfying :math:`Av=kMv` and its eigenvector

    Parameters
    ----------
    A: Union[sparse.spmatrix, np.array, la.LinearOperator]
     Linear operator
    M: Union[sparse.spmatrix, np.array, la.LinearOperator]
     Invertible linear operator matrix
    k: float
     guess for the top eigenvalue, default is 1.
    v: Optional[np.array]
     guess for the eigenvector, if None then a flat guess is used.
    kwargs
     keyword arguments to be passed down to the generator

    Returns
    -------
    tuple[float, np.array]
     tuple whose first value is the eigenvalue and whose second value is the
     eigenvector
    """
    k, v = last(source_iteration_generator(A, M, k0=k, v=v, **kwargs))
    # the sign of an eigenvector is arbitrary: return it with a positive sum
    return float(k), v if np.sum(v) >= 0 else -v


def source_iteration_generator(
    A: Union[sparse.spmatrix, np.array],
    M: Union[sparse.spmatrix, np.array],
    k0: float = 1.0,
    v: Optional[np.array] = None,
    atol_eigenvalue: float = 1e-5,
    rtol_vector: float = 1e-4,
    atol_vector: float = 1e-5,
    lin_rtol: float = 1e-8,
    lin_atol: float = 1e-8,
    max_iter: Optional[int] = None,
) -> Generator[tuple[float, np.array], None, None]:
    r"""
    Generator that yields iterations for the generalized eigenvalue problem
    :math:`Av=kMv`.
    Each iteration is like applying :math:`M^-1A` to a vector.
    This way, after enough iterations we get the largest eigenvalue and its
    eigenvector. This is often referred to as a "Power Iteration".

    Parameters
    ----------
    A: Union[sparse.spmatrix, np.array]
     sparse or dense matrix
    M: Union[sparse.spmatrix, np.array]
     Invertible sparse matrix or dense matrix
    k0: float
     Initial guess for the largest in magnitude eigenvalue. Default is 1.
    v: Optional[np.array]
     Initial guess for the eigenvector. If None, then a flat guess is used.
    atol_eigenvalue: float
     Absolute convergence tolerance for the eigenvalue
    rtol_vector: float
     Relative convergence tolerance for the eigenvector in :math:`l^\infty` norm
    atol_vector: float
     Absolute convergence tolerance for the eigenvector in :math:`l^\infty` norm
    lin_rtol: float
     Relative tolerance for the solution of :math:`Mx=b` when applying the :math:`M^{-1}A`
    lin_atol: float
     Absolute tolerance for the solution of :math:`Mx=b` when applying the :math:`M^{-1}A`
    max_iter: Optional[int]
     The maximal number of iterations allowed. If None, np.inf is used.

    """
    max_iter = max_iter if max_iter is not None else np.inf
    v = v if v is not None else np.ones(A.shape[0])
    iterations = 0
    k, u = k0, v
    while (iterations == 0) or (
        (iterations < max_iter)
        and not _should_stop(k, k0, u, v, ktol=atol_eigenvalue, vatol=atol_vector, vrtol=rtol_vector)
    ):
        k0, v = k, u
        u = la.lgmres(M, A @ v, v * k0, atol=lin_atol, rtol=lin_rtol)[0]
        k = np.linalg.norm(A @ u)
        u /= k
        iterations += 1
        yield k, u


def _should_stop(k, k0, u, v, ktol, vatol, vrtol):
    return np.isclose(k, k0, rtol=ktol) and np.allclose(u, v, atol=vatol, rtol=vrtol)
