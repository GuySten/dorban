"""
This module contains the CMFD accelerated solution of the nem formulation of
the diffusion equation
"""

from typing import Optional, Sequence

import numba
import numpy as np

from dorban.geometry.geometry import FiniteGeometry, faces_between_original_cells
from dorban.system import IsotopeData


class CMFDCurrentCalculator:
    r"""
    This is a class for computing the current between two cells using the CMFD
    method.
    The current is computed according to the following equation


    .. math:: J=\hat{D}\cdot(\phi_1+\phi_2)+\frac{D_1D_2(F_1\phi_1-F_2\phi_2)}{F_2D_1l_2+F_1D_2l_1}

    Where :math:`D_1,D_2` are the diffusion coefficients of the cells,
    :math:`\phi_1,\phi_2` are the fluxes at the centers of the cells,
    :math:`l_1,l_2` are the distances between the centers of the cells to the boundary and
    :math:`F_1,F_2` are the discontinuity factors of the cells on the side of their common face.
    :math:`\hat{D}` is the coupling coefficient between the cells.


    Parameters
    ----------
    dc: np.array
     the diffusion coefficients as a cellsxE np.array
    dim: int
     the dimension of the system.
    coupling: np.array
     the coupling coefficients as a cellsx(number of faces of each cell)xE
     np.array. Default is 0 everywhere.
    df: np.array
     the discontinuity factors as a  cellsx(number of faces of each cell)xE
     np.array. Default is 1 everywhere.

    """

    def __init__(self, dc: np.array, dim: int, coupling: Optional[np.array] = None, df: Optional[np.array] = None):
        self.dc = dc
        self.dim = dim
        try:
            self.E = dc.shape[1]
        except IndexError:  # might happen if there is only 1 energy group
            self.E = 1
        self.coupling = coupling if coupling is not None else np.zeros((dc.shape[0], 2 * dim, self.E))
        self.df = df if df is not None else np.ones_like(self.coupling)

    def dc_on_face(self, cell: int, face: int, geometry: FiniteGeometry) -> np.array:
        """The diffusion coefficients of the cell across the face."""
        return self.dc[cell]

    def mesh_refine(
        self,
        geo: FiniteGeometry,
        split: Sequence,
        refgeo: FiniteGeometry,
    ) -> "CMFDCurrentCalculator":
        """
        creates and returns a new CurrentCalculator which can calculate the
        current for the refined system.

        Parameters
        ----------
        geo: FiniteGeometry
         the geometry of the system
        split: Sequence
         split data whose structure is determined by the geometry of
         the system
        refgeo: FiniteGeometry
         the refined geometry

        Returns
        -------
        CMFDCurrentCalculator
         new current calculator compatible with the refined geometry
        """

        # the faces between the original cells keep their discontinuity factors and coupling coefficients
        assemblies, across = faces_between_original_cells(geo, split, refgeo)
        across = across[:, :, None]
        df = np.where(across, self.df[assemblies], 1.0)
        coupling = np.where(across, self.coupling[assemblies], 0.0)
        return self.__class__(geo.array_refine(self.dc, split, axis=0), self.dim, coupling, df)

    @classmethod
    def from_isotopes(cls, isotopes: IsotopeData, dim: int, transport: bool = False) -> "CMFDCurrentCalculator":
        """
        method to construct a CurrentCalculator from a sequence of :class:`CrossSectionData
        <dorban.materials.CrossSectionData>` objects each having a diffusion cross section or a transport cross
        section. Using this method will result in a CMFD current calculator with default discontinuity factors and
        coupling coefficients.

        Parameters
        ----------
        isotopes: IsotopeData
         Sequences of the :class:`~dorban.materials.CrossSectionData` to use to construct the current calculator
        dim: int
         the dimension of the system.
        transport: bool
         Determines whether to use a diffusion cross section or a transport cross section,
         If False the diffusion cross section is used and if True then the transport one is used.
         False by default.

        Returns
        -------
        CMFDCurrentCalculator
        """
        if transport:
            dc = np.vstack([1 / (3 * isotope.transport) for isotope in isotopes])
        else:
            dc = np.vstack([isotope.diffusion for isotope in isotopes])
        return cls(dc, dim)


@numba.njit(cache=True)
def _compute(cell, E, neighbor, D1, D2, l1, l2, df1, df2, surface_area, coupling_coefficient):
    row_indices = np.hstack((np.arange(cell * E, (cell + 1) * E), np.arange(cell * E, (cell + 1) * E)))
    col_indices = np.hstack((np.arange(neighbor * E, (neighbor + 1) * E), np.arange(cell * E, (cell + 1) * E)))
    return row_indices, col_indices, _values(D1, D2, l1, l2, df1, df2, surface_area, coupling_coefficient)


@numba.njit(cache=True)
def _values(D1, D2, l1, l2, df1, df2, surface_area, coupling_coefficient):
    diff = D1 * D2 / (l2 * df2 * D1 + l1 * df1 * D2)
    values = surface_area * diff
    correction = surface_area * coupling_coefficient
    return np.hstack((correction - df2 * values, correction + df1 * values))
