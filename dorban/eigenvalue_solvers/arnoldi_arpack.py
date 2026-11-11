"""
This is an eigenvalue solver that uses the eigs function of scipy
"""

from typing import Tuple, Union

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as la


def generalized_arnoldi(
    A: Union[sparse.spmatrix, np.array, la.LinearOperator],
    M: Union[sparse.spmatrix, np.array, la.LinearOperator],
    v: np.array = None,
    rtol_vector=1e-4,
    lin_atol=1e-8,
    lin_rtol=1e-8,
    max_iter=np.inf,
    **kwargs,
) -> Tuple[float, np.array]:
    r"""
    this is an eigenvalue solver based on the arnoldi algorithm, it uses lgmres
    to solve linear equations.
    the problem to be solved is :math:`Av=kMv`.

    Parameters
    ----------
    v: np.array
     guess for the eigenvector
    rtol_vector: float
     relative tolerance for the eigenvector
    lin_atol: float
     absolute tolerances for the linear solver
    lin_rtol: float
     relative tolerances for the linear solver
    max_iter: Union[int,np.inf]
     the maximal number of iterations allowed

    Returns
    -------
    Tuple[float, np.array]
     the maximal eigenvalue and its eigenvector
    """
    max_iter = None if max_iter == np.inf else max_iter

    def Minv(vec: np.array):
        return la.lgmres(M, vec, atol=lin_atol, rtol=lin_rtol)[0]

    inverse = la.LinearOperator(M.shape, Minv)

    # noinspection PyTypeChecker
    k, flux = la.eigs(A, 1, M, v0=v, tol=1e-10, Minv=inverse, maxiter=max_iter)
    k = k[0].real
    flux = np.real(flux).flatten()
    # the sign of an eigenvector is arbitrary: return it with a positive sum
    return k, flux if np.sum(flux) >= 0 else -flux


def shift_invert_arnoldi(
    A: Union[sparse.spmatrix, np.array, la.LinearOperator],
    M: Union[sparse.spmatrix, np.array, la.LinearOperator],
    k: float = 1,
    v: np.array = None,
    rtol_vector=1e-5,
    lin_atol=1e-8,
    lin_rtol=1e-8,
    max_iter=np.inf,
    **kwargs,
) -> (complex, np.array):
    r"""
    this is an eigenvalue solver based on the shifted arnoldi algorithm,
    it uses lgmres to solve linear equations.
    the problem to be solved is :math:`Av=kMv`.

    Parameters
    ----------
    v: np.array
     guess for the eigenvector
    rtol_vector: float
     relative tolerance for the eigenvector
    lin_atol: float
     absolute tolerances for the linear solver
    lin_rtol: float
     relative tolerances for the linear solver
    max_iter: Union[int,np.inf]
     the maximal number of iterations allowed

    Returns
    -------
    Tuple[float, np.array]
     the maximal eigenvalue and its eigenvector
    """
    max_iter = None if max_iter == np.inf else max_iter

    def OPinv(vec: np.array):
        return la.lgmres(A - k * M, vec, atol=lin_atol, rtol=lin_rtol)[0]

    # noinspection PyTypeChecker
    k, flux = la.eigs(
        A, 1, M, v0=v, tol=rtol_vector, sigma=k, maxiter=max_iter, OPinv=la.LinearOperator(M.shape, OPinv)
    )
    k = k[0].real
    flux = flux.astype(float).flatten()
    # the sign of an eigenvector is arbitrary: return it with a positive sum
    return k, flux if np.sum(flux) >= 0 else -flux
