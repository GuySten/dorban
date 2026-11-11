from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np


@dataclass
class Settings:
    r"""
    Settings for a diffusion calculation

    Attributes
    ----------
    adjoint: Boolean
        whether to solve the direct or the adjoint equation. Defaults to False (direct equation).
    initial_flux: Optional[np.array]
        The guess for the flux
    initial_k: float
        The guess for the k eigenvalue
    k_tol: float
        The tolerance for the eigenvalue
    flux_rtol: float
        The relative tolerance for the flux
    flux_atol: float
        The absolute tolerances for the flux
    lin_solve_rtol: float
        Many Eigenvalue solvers need to solve :math:'Mx=b' type problems during
        their execution. This determines the relative tolerance for the solution
        of such problems.
    lin_solve_atol: float
        Many Eigenvalue solvers need to solve :math:'Mx=b' type problems during
        their execution. This determines the absolute tolerance for the solution
        of such problems.
    max_iter: int
        The maximal number of iterations allowed in the power iteration
        solution
    save_file: Optional[str]
        Path to the file in which to save the flux
    solver_name: Literal["slepc","arpack","power iteration"]
        Which eigensolver to use. The options are:
        "power iteration", "arpack" and "slepc"
    debug_discontinuity : bool
     Flag that determines if to run the debug function of the discontinuity factors that checks that they are reasonable.
     If its True and the discontinuity factors are not reasonable an error will be raised. Default to True.
    """

    adjoint: bool = False
    initial_flux: Optional[np.array] = field(default=None, repr=False)
    initial_k: float = 1.0
    k_tol: float = 1e-5
    flux_rtol: float = 1e-5
    flux_atol: float = 1e-5
    lin_solve_rtol: float = 1e-8
    lin_solve_atol: float = 1e-8
    max_iter: int = np.inf
    save_file: Optional[str] = None
    solver_name: Literal["slepc", "arpack", "power iteration"] = "slepc"
    debug_discontinuity: bool = True

    @property
    def kwargs(self):
        return {
            "max_iter": self.max_iter,
            "lin_atol": self.lin_solve_atol,
            "lin_rtol": self.lin_solve_rtol,
            "atol_vector": self.flux_atol,
            "rtol_vector": self.flux_rtol,
            "atol_eigenvalue": self.k_tol,
            "k": self.initial_k,
            "v": self.initial_flux,
        }


@dataclass
class FDSettings(Settings):
    """
    Settings for a finite difference calculation

    Attributes
    ----------
    split:
     Information about how to split the geometry (?).
     Must be set to something that isn't None.

    Raises
    ------
    ValueError if split is not set.

    See Also
    --------
    Settings - The superclass for this settings class.

    """

    split: Any = field(default=None)

    def __post_init__(self):
        if self.split is None:
            raise ValueError("The split argument must be set (not to None)in FDSettings, but it was None")
