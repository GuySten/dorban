"""
This class contains the CMFD iterator used for the solution of the NEM coupled
CMFD equations.
"""

import logging
import warnings
from typing import Iterable

import numba
import numpy as np

from dorban.eigenvalue_solvers.slepc_methods import generalized_eigenvalue_slepc, solve_linear_petsc
from dorban.finite_differences import cmfd_absorption, cmfd_fission
from dorban.geometry.boundary_conditions import Boundary, Reflector, Void
from dorban.materials import CrossSectionData
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.compiled_diffusion_matrix import diffusion_matrix
from dorban.nem.leakage import leakage
from dorban.nem.nem_coupling_rhs import nem_boundary_rhs, nem_rhs
from dorban.nem.nem_matrices import nem_boundary_mat, nem_mat
from dorban.nem.utils import (
    boundary_coupling_coefficient,
    cell_indices_in_two_cell_system,
    cells_axi_energy_slice,
    cells_axi_slice,
    compute_coupling_coefficients,
    current_from_poly_exp,
    energy_slice,
    other_axi,
)
from dorban.system import Core
from dorban.utils import op_face

logger = logging.getLogger(__name__)


def boundary_conditions_numbering(boundary: Boundary) -> int:
    """
    Function that assigns a negative number to any supported boundary condition type.
    Currently only reflective and void boundary conditions are supported.
    Void boundary condition is assigned the number -1.
    Reflector boundary condition is assigned the number -2.

    Parameters
    ----------
    boundary: Boundary
     The boundary condition

    Returns
    -------
    int
    """
    if isinstance(boundary, Void):
        return -1
    if isinstance(boundary, Reflector):
        return -2
    raise NotImplementedError(f"The boundary condition of type {boundary} isn't implemented in the NEM kernel.")


def change_neighbors(neighbors: Iterable[Iterable[int | Boundary]]) -> np.array:
    """
    Function that creates an array representing the neighbor relations of cells, boundary conditions object are
    replaced by negative numbers that represent them. Numbers of neighbors remain unchanged.

    Parameters
    ----------
    neighbors: Iterable[Iterable[int | Boundary]]
     Iterable that at place i has the neighbors of cell number i.

    Returns
    -------
    np.array
    """
    return np.array(
        [
            [n if not isinstance(n, Boundary) else boundary_conditions_numbering(n) for n in cell_neighbors]
            for cell_neighbors in neighbors
        ]
    )


def change_isotopes(isotopes: Iterable[CrossSectionData], E: int) -> np.array:
    r"""
    Function that replaces a collection of CrossSectionData object with an array that contains the cross sections.

    Parameters
    ----------
    isotopes : Iterable[CrossSectionData]
     The collection of CrossSectionData objects.
    E: int
     The number of energy groups.

    Returns
    -------
    np.array
     Array which contains for each cell the cross section :math:`\Sigma_{total}. \Sigma_a, \Sigma_s, \nu\Sigma_f` and
     :math:`\chi`.
    """
    return np.array([mat.data[: E + 4, :] for mat in isotopes])


class GeneratorNemCMFD:
    r"""
    Generator class that solves the k eigenvalue equation in the NEM discretization using CMFD acceleration.
    At each iteration the k-eigenvalue, the average flux at each cell and the coupling coefficients between cells
    are updated. At the first iteration a standard finite differences k-eigenvalue problem is solved and a guess for
    the eigenvalue and average flux are obtained. At the first iteration the coupling coefficients are zero.

    At each consecutive iteration the NEM equations are solved between every two adjacent cells and the currents between
    them are computed, then the coupling coefficients are updated using:

        .. math:: \hat{D}=\left(J_{NEM}-\frac{D_1D_2(F_1\phi_1-F_2\phi_2)}{F_2D_1l_2+F_1D_2l_1}\right)
                  \cdot(\phi_1+\phi_2)^{-1}

    Where :math:`D_1,D_2` are the diffusion coefficients of the cells,
    :math:`\phi_1,\phi_2` are the fluxes at the centers of the cells,
    :math:`l_1,l_2` are the distances between the centers of the cells to the boundary and
    :math:`F_1,F_2` are the discontinuity factors of the cells on the side of their common face.
    :math:`\hat{D}` is the coupling coefficient between the cells.
    :math:`J_{NEM}` is the current between the cells according to the two cell NEM computation

    After updating the coupling coefficients an outer iteration of the CMFD solver is preformed. This iteration is
    solving a linear problem of the type :math:`Ax=b` where :math:`b` is the fission source of the previous iteration and
    :math:`A` is the diffusion + absorption - minus scattering matrix, the diffusion matrix is constructed using
    the updated coupling coefficients and the modified Fick's Law :math:`J=-D\nabla\phi+\hat{D}\phi`.

    The whole algorithm is accelerated using wielandt shift. Meaning the eigenvalue is shifted before preforming the
    outer iteration, ths shift is taken to be 0.04 as recommended in the PARCS manual for LWR computations.

    Parameters
    ----------
    core: Core
        Core data. Note that only :class:`~.CMFDCurrentCalculator` are supported.
    cmfd_max_iter:int
     The maximal number allowed of cmfd iterations without coefficients update. Default is 0,
     meaning after each iteration the coupling coefficients are updated.
    k: float
     guess for the top eigenvalue, if it is None a finite differences eigenvalue problem is solved at the first iteration.
    v: np.array
     guess for the average flux at each cell, if it is None a finite differences eigenvalue problem is solved at the first iteration.
    tol_value: float
     Absolute convergence tolerance for the eigenvalue.
    rtol_vector: float
     Relative convergence tolerance for the eigenvector.
    atol_vector: float
     Absolute convergence tolerance for the eigenvector.
    lin_rtol: float
     Relative convergence for the linear solver.
    lin_atol: float
     Absolute convergence for the linear solver.
    max-iter: float
     Maximal number of CMFD iteration allowed.

    """

    def __init__(
        self,
        core: Core,
        cmfd_max_iter=0,
        k: float = None,
        v: np.array = None,
        tol_value: float = 1e-10,
        rtol_vector: float = 1e-4,
        atol_vector: float = 1e-5,
        lin_rtol: float = 1e-8,
        lin_atol=1e-8,
        max_iter=np.inf,
    ):
        self.core = core
        self.dim = core.geometry.dim
        self.current_calc: CMFDCurrentCalculator = core.current_calc  # type: ignore
        self.distances = np.array(
            [
                [core.geometry.distance_to_face(cell, face) for face in range(2 * self.dim)]
                for cell in range(core.geometry.cells)
            ]
        )
        self.surface_areas = np.array(
            [
                [core.geometry.surface_area(cell, face) for face in range(2 * self.dim)]
                for cell in range(core.geometry.cells)
            ]
        )
        self.neighbors = change_neighbors(core.geometry.neighbors)
        self.volumes = core.geometry.volumes
        self.materials_data = change_isotopes(core.isotopes, core.E)
        self.cmfd_max_iter = cmfd_max_iter
        self.max_iter = max_iter
        self.cmfd_iter = 0
        self.iter = 0
        self.lin_atol = lin_atol
        self.lin_rtol = lin_rtol
        self.k = k
        self.v = v
        self.tol_value = tol_value
        self.atol_vector = atol_vector
        self.rtol_vector = rtol_vector
        self.setup()

    def setup(self):
        """Method to construct the initial matrices."""
        self.F = cmfd_fission(self.core)
        self.A = cmfd_absorption(self.core)
        self.M = self.A + diffusion_matrix(
            self.neighbors, self.distances, self.surface_areas, self.materials_data.shape[2], self.current_calc
        )
        self.dc = self.current_calc.dc
        self.df = self.current_calc.df

    def __next__(self):
        if self.v is None:
            self.set_first_guess()
            k, u = self.k, self.v
        else:
            k, u = self.outer_iteration()
        if not self.should_stop(k, u):
            self.k = k
            self.v = u
            if self.cmfd_iter > self.cmfd_max_iter:
                self.update_coupling()
                self.M = self.A + diffusion_matrix(
                    self.neighbors, self.distances, self.surface_areas, self.materials_data.shape[2], self.current_calc
                )
                self.cmfd_iter = 0
            return self.k, self.v
        else:
            if self.iter == 1:
                self.iter += 1
                return k, u
            raise StopIteration

    def set_first_guess(self):
        self.k, self.v = generalized_eigenvalue_slepc(self.F, self.M, groups=self.materials_data.shape[2])

    def outer_iteration(self) -> tuple[float, np.array]:
        """
        Method to preform an outer iteration accelerated by wielandt_shift.
        First the eigenvalue is shifted, then an outer iteration is performed and then the eigenvalue is shifted back.
        """
        k_old = self.k + 0.04  # PARCS constant for LWR
        u = solve_linear_petsc(
            self.M - 1 / k_old * self.F, self.F @ self.v, guess=self.v * self.k, rtol=self.lin_rtol, atol=self.lin_atol
        )
        k_new = np.sum(self.F @ u)
        u /= k_new
        k = 1 / (1 / k_new + 1 / k_old)
        self.iter += 1
        self.cmfd_iter += 1
        return k, u

    def update_coupling(self) -> None:
        self.current_calc.coupling = update_coupling_coefficients(
            self.dim,
            self.materials_data,
            self.neighbors,
            self.distances,
            self.surface_areas,
            self.dc,
            self.df,
            self.current_calc.coupling,
            self.v,
            self.k,
            np.zeros_like(self.v),
        )

    def should_stop(self, k, u) -> bool:
        """
        Method that decides when to stop the iteration.
        The iteration is stopped if the tolerances are achieved or if the flux becomes negative.
        In the second case a warning is raised.
        The CMFD algorithm is unstable if the flux is negative in any cell,
        so the iteration is stopped before it loses stability.
        """
        if self.iter <= 2:
            return False
        if np.any(u / u[0] < 0):
            warnings.warn(
                f"Negative flux encountered stopping iteration. "
                f"convergence in k is {abs(k - self.k)}, tolerance is {self.tol_value}, "
                f"convergence in flux is {np.linalg.norm(self.F @ u - k * self.M @ u)}, "
                f"tolerance is {k * self.rtol_vector * np.linalg.norm(u)}"
            )
            logger.info(
                f"Stopped after {self.iter} iteration because"
                f"encountered negative flux."
                f"convergence in k is {abs(k - self.k)}, tolerance is {self.tol_value}, "
                f"convergence in flux is {np.linalg.norm(self.F @ u - k * self.M @ u)}, "
                f"tolerance is {k * self.rtol_vector * np.linalg.norm(u)}"
            )
            return True
        to_stop = (
            abs(k - self.k) < self.tol_value
            and np.linalg.norm(self.F @ u - k * self.M @ u) < k * self.rtol_vector * np.linalg.norm(u)
            and not np.all(self.current_calc.coupling == 0)
        ) or self.iter > self.max_iter
        if to_stop:
            logger.info(
                f"Stopped after {self.iter} iteration because"
                f"converged"
                f"convergence in k is {abs(k - self.k)}, tolerance is {self.tol_value}, "
                f"convergence in flux is {np.linalg.norm(self.F @ u - k * self.M @ u)}, "
                f"tolerance is {k * self.rtol_vector * np.linalg.norm(u)}"
            )
        return to_stop

    def __iter__(self):
        return self


class GeneratorNemCMFDSource(GeneratorNemCMFD):
    r"""
    Generator class that solves the source equation in the NEM discretization using CMFD acceleration.
    At each iteration the average flux at each cell and the coupling coefficients between cells
    are updated. At the first iteration a standard finite differences source problem is solved and a guess for
    the average flux are obtained. At the first iteration the coupling coefficients are zero.

    At each consecutive iteration the NEM equations are solved between every two adjacent cells and the current between
    them are computed, next the coupling coefficients are updated using:

        .. math:: \hat{D}=(J_{NEM}-\frac{D_1D_2(F_1\phi_1-F_2\phi_2)}{F_2D_1l_2+F_1D_2l_1})\cdot(\phi_1+\phi_2)^{-1}

    Where :math:`D_1,D_2` are the diffusion coefficients of the cells,
    :math:`\phi_1,\phi_2` are the fluxes at the centers of the cells,
    :math:`l_1,l_2` are the distances between the centers of the cells to the boundary and
    :math:`F_1,F_2` are the discontinuity factors of the cells on the side of their common face.
    :math:`\hat{D}` is the coupling coefficient between the cells.
    :math:`J_{NEM}` is the current between the cells according to the two cell NEM computation

    After updating the coupling coefficients an outer iteration of the CMFD solver is preformed.

    The whole algorithm is accelerated using wielandt shift. Meaning the eigenvalue is shifted before preforming the
    outer iteration, ths shift is taken to be 0.04 as recommended in the PARCS manual for LWR computations.

    Parameters
    ----------
    core: Core
    source: np.array
     The source of the problem: the total source in each cell and energy group (the source density times the
     cell volume), not a density, organized by cell and then by energy group.
    cmfd_max_iter: int
     the maximal number allowed of CMFD iterations without coefficients update. Default is 0,
     meaning after each iteration the coupling coefficients are updated.
    v: np.array
     guess for the average flux at each cell, if it is None
     a finite differences eigenvalue problem is solved at the first iteration.
    tol_value: float
     Absolute convergence tolerance for the eigenvalue.
    rtol_vector: float
     Relative convergence tolerance for the eigenvector.
    atol_vector: float
     Absolute convergence tolerance for the eigenvector.
    lin_rtol: float
     Relative convergence for the linear solver.
    lin_atol: float
     Absolute convergence for the linear solver.
    max-iter: float
     Maximal number of CMFD iteration allowed.
    """

    def __init__(
        self,
        core: Core,
        source: np.array,
        cmfd_max_iter=0,
        k: float = 1,
        v: np.array = None,
        tol_value: float = 1e-10,
        rtol_vector: float = 1e-4,
        atol_vector: float = 1e-5,
        lin_rtol: float = 1e-8,
        lin_atol=1e-8,
        max_iter=np.inf,
    ):
        super().__init__(core, cmfd_max_iter, k, v, tol_value, rtol_vector, atol_vector, lin_rtol, lin_atol, max_iter)
        self.source = source

    def set_first_guess(self):
        self.k = 1
        self.v = solve_linear_petsc(self.M - self.F, self.source)

    def outer_iteration(self) -> tuple[float, np.array]:
        u = solve_linear_petsc(
            self.M, self.F @ self.v + self.source, guess=self.v, rtol=self.lin_rtol, atol=self.lin_atol
        )
        self.iter += 1
        self.cmfd_iter += 1
        return 1, u

    def update_coupling(self):
        # self.source is the total source in each cell, as in the volume integrated CMFD equations,
        # while the two-cell NEM equations balance it against reaction rates per unit volume.
        source_density = self.source / np.repeat(self.volumes, self.materials_data.shape[2])
        self.current_calc.coupling = update_coupling_coefficients(
            self.dim,
            self.materials_data,
            self.neighbors,
            self.distances,
            self.surface_areas,
            self.dc,
            self.df,
            self.current_calc.coupling,
            self.v,
            self.k,
            source_density,
        )

    def should_stop(self, k, u):
        """
        Method that decides when to stop the iteration.
        The iteration is stopped if the tolerances are achieved or if the flux becomes negative.
        In the second case a warning is raised.
        The CMFD algorithm is unstable if the flux is negative in any cell,
        so the iteration is stopped before it loses stability.
        The tolerances are achieved when the CMFD equations are solved and the flux changed by less than the
        relative tolerance since the previous iteration. The CMFD residual alone is not enough: without fission
        the CMFD equations are solved exactly at every iteration, before the coupling coefficients have converged.
        """
        if self.iter <= 2:
            return False
        if np.any(u / u[0] < 0):
            warnings.warn(
                f"Negative flux encountered stopping iteration. "
                f"convergence in flux is {np.linalg.norm(self.source + self.F @ u - self.M @ u)}, "
                f"tolerance is {self.rtol_vector * np.linalg.norm(u)}"
            )
            return True
        return (
            np.linalg.norm(self.source + self.F @ u - self.M @ u) < self.rtol_vector * np.linalg.norm(u)
            and np.linalg.norm(u - self.v) < self.rtol_vector * np.linalg.norm(u)
            and not np.all(self.current_calc.coupling == 0)
        ) or self.iter > self.max_iter


@numba.njit(cache=True)
def update_coupling_coefficients(
    dim: int,
    mat_data: np.array,
    neighbors_array: np.array,
    distances: np.array,
    surface_areas: np.array,
    dc: np.array,
    df: np.array,
    coupling: np.array,
    v: np.array,
    k: float,
    source: np.array,
):
    """
    Function that updates nodal coupling coefficients. It edits the coupling array in place.
    The computation is separated to a computation for the coupling coefficient between two cells
    and a computation of the coupling coefficient near void boundary condition.
    The coupling coefficient near reflective boundary conditions is 0.

    Parameters
    ----------
    dim: int
     The dimension of the system.
    mat_data: np.array
     Array which contains the cross-sections of each material
    neighbors_array: np.array
     Array of shape cells x faces which contains the neighbors of each cell,
     negative values represent boundary conditions.
    distances: np.array
     Array of shape cells x faces which contains the distances between the center of each cell to its faces.
    surface_areas: np.array
     Array of shape cells x faces which contains the surface area of each face.
    dc: np.array
     The array of diffusion coefficients.
    df: np.array
     The array of discontinuity factors
    coupling: np.array
     The array of coupling coefficients.
    v: np.array
     The average flux at each cell.
    k: float
     The top eigenvalue, in a source problems is 1.
    source: np.array
     The source, in an eigenvalue problem is zero.
    """
    E = mat_data.shape[2]
    ns = other_axi
    leak = leakage(neighbors_array, distances, surface_areas, dc, df, coupling, E, v, dim)
    for cell, neighbors in enumerate(neighbors_array):
        for face, neighbor in enumerate(neighbors):
            axis = face // 2
            t_axi = ns(axis, dim)
            left = neighbors[op_face(face)]
            l_left, leakage_left = length_and_leakage(distances, leak, cell, left, axis, t_axi)
            l_right, leakage_right = length_and_leakage(distances, leak, cell, neighbor, axis, t_axi)
            if (neighbor >= 0) and (face % 2 == 1):
                coupling[cell, face] = _coupling_between_two_cells(
                    E,
                    v,
                    distances,
                    cell,
                    neighbor,
                    face,
                    dc,
                    mat_data,
                    df,
                    neighbors_array,
                    leak,
                    axis,
                    t_axi,
                    k,
                    source,
                    l_right,
                    l_left,
                    leakage_right,
                    leakage_left,
                )
                coupling[neighbor, op_face(face)] = -coupling[cell, face]
            elif neighbor == -1:
                coupling[cell, face] = _coupling_between_cell_and_void(
                    E,
                    v,
                    distances,
                    cell,
                    face,
                    dc,
                    mat_data,
                    leak,
                    axis,
                    t_axi,
                    k,
                    l_right,
                    l_left,
                    leakage_right,
                    leakage_left,
                    source,
                )
    return coupling


@numba.njit(cache=True)
def length_and_leakage(
    distances: np.array, leak: np.array, cell: int, neighbor: int, axis: int, t_axi: tuple[int, int]
) -> tuple[float, np.array]:
    """
    Extracts the length and leakage of a neighbor of a cell.
    This is used for the quadratic leakage approximation which assumes that the leakage across the boundary of each cell
    is a quadratic polynomial whose coefficients are the leaks from the neighboring cells.
    In the case that a neighbor is a boundary, which is indicated by a negative neighbor the approximation is made that
    the length of the "fake" neighbor cell is the same as the length of main cell and the leakage is opposite
    of the leakage from the main cell.
    This is done according to a recommendation which appears in a paper By Kord Smith.

    Parameters
    ----------
    distances: np.array
     Array of shape cells x faces which contains the distances between the center of each cell to its faces.
    leak: np.array
     array of shape cells x dim x E which contains the leakage from each cell in each direction.
    cell: int
     The number of the main cell.
    neighbor: int
     The number of the neighbor.
    axis: int
     The number of the axis connecting the cell to its neighbor.
    t_axi: tuple[int,...]
     The numbers of the axi which are different than axis.

    Returns
    -------
    tuple[float, np.array]
     The length of the neighbor cell and the leakage from it.
    """
    x, sign = (neighbor, 1) if not neighbor < 0 else (cell, -1)
    length = 2 * distances[x, 2 * axis]
    _leakage = sign * cells_axi_energy_slice(leak, x, t_axi)
    return length, _leakage


@numba.njit(cache=True)
def _coupling_between_two_cells(
    E,
    v,
    distances,
    cell,
    neighbor,
    face,
    dc,
    mat_data,
    df,
    neighbors_array,
    leak,
    axis,
    t_axi,
    k,
    source,
    l_right,
    l_left,
    leakage_right,
    leakage_left,
):
    """Function that computes the coupling coefficient between two cells"""
    mat = nem_mat(
        E,
        2 * distances[cell, face],
        2 * distances[neighbor, face],
        dc[cell],
        dc[neighbor],
        _absorb(mat_data[cell]),
        _absorb(mat_data[neighbor]),
        _nufission(mat_data[cell]) / k,
        _nufission(mat_data[neighbor]) / k,
        df[cell, face],
        df[neighbor, op_face(face)],
    )
    righter = neighbors_array[neighbor][face]
    l_righter, leakage_righter = length_and_leakage(distances, leak, neighbor, righter, axis, t_axi)
    rhs = nem_rhs(
        E,
        v[energy_slice(cell, E)],
        v[energy_slice(neighbor, E)],
        cells_axi_energy_slice(leak, cell, t_axi),
        _absorb(mat_data[cell]),
        _absorb(mat_data[neighbor]),
        _nufission(mat_data[cell]) / k,
        _nufission(mat_data[neighbor]) / k,
        leakage_right,
        leakage_left,
        2 * cells_axi_slice(distances, cell, 2 * t_axi),
        df[cell, face],
        df[neighbor, op_face(face)],
        leakage_righter,
        2 * distances[cell, 2 * axis],
        l_right,
        l_left,
        l_righter,
        source[energy_slice(cell, E)],
        source[energy_slice(neighbor, E)],
    )
    poly_exp = np.linalg.solve(mat, rhs)
    current = current_from_poly_exp(
        E, poly_exp[cell_indices_in_two_cell_system(E)], dc[cell], 2 * distances[cell, 2 * axis], face
    )
    return compute_coupling_coefficients(
        current,
        v[energy_slice(cell, E)],
        v[energy_slice(neighbor, E)],
        dc[cell],
        dc[neighbor],
        2 * distances[cell, 2 * axis],
        2 * distances[neighbor, 2 * axis],
        df[cell, 2 * axis + 1],
        df[neighbor, 2 * axis],
    )


@numba.njit(cache=True)
def _coupling_between_cell_and_void(
    E,
    v,
    distances,
    cell,
    face,
    dc,
    mat_data,
    leak,
    axis,
    t_axi,
    k,
    l_right,
    l_left,
    leakage_right,
    leakage_left,
    source,
):
    """
    Function that computes the coupling coefficient between a cell and void boundary condition.
    Void boundary condition is represented by the number -1.
    """
    D_over_l = np.diag(dc[cell]) / distances[cell, face] / 2
    void_boundary_equation = np.hstack(
        (
            3 * D_over_l + np.diag(np.full((E,), 1 / 4)),
            D_over_l / 5,
            D_over_l + np.diag(np.full((E,), 1 / 4)),
            D_over_l / 2,
        )
    )
    mat = nem_boundary_mat(
        E,
        2 * distances[cell, face],
        dc[cell],
        _absorb(mat_data[cell]),
        _nufission(mat_data[cell]) / k,
        void_boundary_equation,
    )
    flux = v[energy_slice(cell, E)]
    rhs = nem_boundary_rhs(
        E,
        flux,
        cells_axi_energy_slice(leak, cell, t_axi),
        _absorb(mat_data[cell]),
        _nufission(mat_data[cell]) / k,
        leakage_right,
        leakage_left,
        2 * cells_axi_slice(distances, cell, 2 * t_axi),
        2 * distances[cell, 2 * axis],
        l_right,
        l_left,
        source[energy_slice(cell, E)],
        -flux / 2,
    )
    poly_exp = np.linalg.solve(mat, rhs)
    current = current_from_poly_exp(E, poly_exp, dc[cell], 2 * distances[cell, 2 * axis], 1)
    return boundary_coupling_coefficient(current, v[energy_slice(cell, E)], np.full(E, 1 / 2))


@numba.njit(cache=True)
def _nufission(data: np.array):
    E = data.shape[1]
    if np.all(data[E + 3] == 0):
        return np.zeros((E, E))
    return np.outer(data[E + 3] / np.sum(data[E + 3]), data[E + 2])


@numba.njit(cache=True)
def _absorb(data: np.array):
    E = data.shape[1]
    return np.diag(data[E + 1]) - data[:E, :]
