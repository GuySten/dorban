"""
This module contains the finite differences current calculators
"""

from typing import Sequence

import numpy as np

from dorban.current_calculator import CurrentCalculator
from dorban.geometry.boundary_conditions import Boundary
from dorban.geometry.geometry import FiniteGeometry
from dorban.system import IsotopeData


class CurrentCalculatorFD(CurrentCalculator):
    r"""
    This is a class for computing the current between two cells in the
    finite differences model without discontinuity factors.
    The current is computed according to the following equation


    .. math:: J=\frac{D_1D_2(\phi_1-\phi_2)}{D_1l_2+D_2l_1}

    Where :math:`D_1,D_2` are the diffusion coefficients of the cells,
    :math:`\phi_1,\phi_2` are the fluxes at the centers of the cells and
    :math:`l_1,l_2` are the distances between the centers of the cells to the boundary.

    Parameters
    ----------
    dc: np.array
     The diffusion coefficients as cellsxE np.array
    """

    def __init__(self, dc: np.array):
        self.dc = dc
        self.E = dc.shape[1]

    def dc_on_face(self, cell: int, face: int, geometry: FiniteGeometry) -> float:
        """
        returns the diffusion coefficient across a given face

        Parameters
        ----------
        cell: int
         the index of the cell
        face: int
         the index of the face
        geometry: FiniteGeometry
         the geometry of the system
        """
        return self.dc[cell]

    def face_coefficients(self, cells, others, faces, opposite, distances, other_distances, areas, geometry):
        """
        The coefficients of the currents through many faces at once, see
        :meth:`dorban.current_calculator.CurrentCalculator.face_coefficients`.
        """
        D1, D2 = self.dcs_on_faces(cells, faces, geometry), self.dcs_on_faces(others, opposite, geometry)
        values = areas[:, None] * D1 * D2 / (D2 * distances[:, None] + D1 * other_distances[:, None])
        return values, values

    def dcs_on_faces(self, cells: np.array, faces: np.array, geometry: FiniteGeometry) -> np.array:
        """The diffusion coefficients of the given cells across the given faces, an array of shape faces x E."""
        return self.dc[cells]

    def mesh_refine(self, geo: FiniteGeometry, split: Sequence, refgeo) -> "CurrentCalculatorFD":
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
        CurrentCalculatorFD
         new current calculator compatible with the refined geometry
        """
        return self.__class__(geo.array_refine(self.dc, split, axis=0))

    @classmethod
    def from_isotopes(cls, isotopes: IsotopeData, transport: bool = False) -> "CurrentCalculatorFD":
        """
        method to construct a CurrentCalculator from a sequence of :class:`CrossSectionData <dorban.materials.CrossSectionData>`
        objects each having a diffusion cross-section or a transport cross-section.

        Parameters
        ----------
        isotopes: IsotopeData
         Sequences of the  :class:`CrossSectionData <dorban.materials.CrossSectionData>`
         to use to construct the current calculator.
        transport: bool
         Determines whatever to use a diffusion cross section or a transport cross section,
         If False the diffusion cross section is used and if True then the transport one is used.
         False by default.

        Returns
        -------
        CurrentCalculatorFD
        """
        return cls(collect_diffusion_coefficients(isotopes, transport))


class DiscontinuityCurrentCalculator(CurrentCalculator):
    r"""
    This is a class that computes the current between two cells using the
    discontinuity factors method in the finite differences model.
    The current is computed according to the following equation


    .. math:: J=\frac{D_1D_2(F_1\phi_1-F_2\phi_2)}{F_2D_1l_2+F_1D_2l_1}

    Where :math:`D_1,D_2` are the diffusion coefficients of the cells,
    :math:`\phi_1,\phi_2` are the fluxes at the centers of the cells,
    :math:`l_1,l_2` are the distances between the centers of the cells to the boundary and
    :math:`F_1,F_2` are the discontinuity factors of the cells on the side of their common face.

    Parameters
    ----------
    dc:np.array
     the diffusion coefficients as a cellsxE np.array
    df:np.array
     the discontinuity factors as a  cellsx(number of faces of each cell)xE
     np.array.

     Notice that although the discontinuity factors are defined across both sides
     of each face only the ratio between them is used.

    """

    def __init__(self, dc: np.array, df: np.array):
        self.dc = dc
        self.df = df
        self.E = dc.shape[1]

    def dc_on_face(self, cell: int, face: int, geometry: FiniteGeometry) -> float:
        """
        returns the diffusion coefficient across a given face

        Parameters
        ----------
        cell: int
         the index of the cell
        face: int
         the index of the face
        geometry: FiniteGeometry
         the geometry of the system
        """
        return self.dc[cell]

    def face_coefficients(self, cells, others, faces, opposite, distances, other_distances, areas, geometry):
        """
        The coefficients of the currents through many faces at once, see
        :meth:`dorban.current_calculator.CurrentCalculator.face_coefficients`.
        """
        D1, D2 = self.dc[cells], self.dc[others]
        df1, df2 = self.df[cells, faces], self.df[others, opposite]
        values = areas[:, None] * D1 * D2 / (other_distances[:, None] * df2 * D1 + distances[:, None] * df1 * D2)
        return df1 * values, df2 * values

    def mesh_refine(
        self, geometry: FiniteGeometry, split: Sequence, refgeo: FiniteGeometry
    ) -> "DiscontinuityCurrentCalculator":
        """
        creates and returns a new CurrentCalculator which can calculate the
        current for the refined system.

        Parameters
        ----------
        geometry: FiniteGeometry
         the geometry of the system
        split: Sequence
         split data whose structure is determined by the geometry of
         the system
        refgeo: FiniteGeometry
         the refined geometry

        Returns
        -------
        DiscontinuityCurrentCalculator
         new current calculator compatible with the refined geometry
        """
        assemblies = geometry.array_refine(np.arange(0, stop=geometry.cells), split).astype(int)
        df = np.array(
            [
                [
                    self.df[assemblies[cell]][face]
                    if ((not isinstance(neighbor, Boundary)) and assemblies[cell] != assemblies[neighbor])
                    else np.ones_like(self.df[assemblies[cell]][face])
                    for face, neighbor in enumerate(neighbors)
                ]
                for cell, neighbors in enumerate(refgeo.neighbors)
            ]
        )
        return self.__class__(geometry.array_refine(self.dc, split, axis=0), df)

    @classmethod
    def from_isotopes(
        cls, isotopes: IsotopeData, face_num: int, transport: bool = False
    ) -> "DiscontinuityCurrentCalculator":
        """
        method to construct a CurrentCalculator from a sequence of :class:`CrossSectionData <dorban.materials.CrossSectionData>`
        objects each having a diffusion cross section or a transport cross section.

        Parameters
        ----------
        isotopes: IsotopeData
         Sequences of the  :class:`CrossSectionData <dorban.materials.CrossSectionData>` to use to construct the current calculator
        face_num: int
         The number of faces each cell has.
        transport: bool
         Determines whatever to use a diffusion cross section or a transport cross section,
         If False the diffusion cross section is used and if True then the transport one is used.
         False by default.

        Returns
        -------
        DiscontinuityCurrentCalculator
        """
        dc = collect_diffusion_coefficients(isotopes, transport)
        E = isotopes[0].data.shape[1]
        df = np.zeros((len(isotopes), face_num, E))
        for cell, isotope in enumerate(isotopes):
            df[cell] = np.array([list(isotope.adf)] * face_num)
        return cls(dc, df)


def collect_diffusion_coefficients(isotopes: IsotopeData, transport: bool) -> np.array:
    """
    function to construct an array of diffusion coefficients from a sequence of :class:`CrossSectionData <dorban.materials.CrossSectionData>`
    objects each having a diffusion cross-section or a transport cross-section.

    Parameters
    ----------
    isotopes: IsotopeData
     Sequences of the  :class:`CrossSectionData <dorban.materials.CrossSectionData>`
     to use to construct the current calculator.
    transport: bool
     Determines whatever to use a diffusion cross section or a transport cross section,
     If False the diffusion cross section is used and if True then the transport one is used.
     False by default.

    Returns
    -------
    np.array
     array of diffusion coefficients
    """
    relevant_data = (lambda isotope: 1 / (3 * isotope.transport)) if transport else (lambda isotope: isotope.diffusion)
    return np.vstack([relevant_data(isotope) for isotope in isotopes])


class DirectionalCurrentCalculator(CurrentCalculatorFD):
    r"""
    This is a current calculator that uses directional diffusion coefficients
    for treating situation in which the diffusion rate is very dependent on the
    direction of the neutrons' movement, for example in the presence of large in core voids.

    The current is computed according to the following equation


    .. math:: J=\frac{D'_1D'_2(\phi_1-\phi_2)}{D'_1l_2+D'_2l_1}

              D'_1=(D_1a_1^{\rightarrow})\cdot a_1^{\rightarrow}

              D'_2=(D_2a_2^{\rightarrow})\cdot a_2^{\rightarrow}


    Where :math:`D_1,D_2` are the diffusion coefficients of the cells,
    :math:`\phi_1,\phi_2` are the fluxes at the centers of the cells,
    :math:`l_1,l_2` are the distances between the centers of the cells to the boundary
    and :math:`a_1^{\rightarrow},a_2^{\rightarrow}` are the unit vectors
    pointing from the center of the cells to the center of the face between them.


    Parameters
    ----------
    dc:np.array
     np array of dimensions (cells,E,dim,dim)
     For each cell i and energy group e, diffusion[i][e] is the directional
     diffusion coefficient at this cell and energy group.
    """

    def __int__(self, dc: np.array):
        super().__init__(dc)

    def dcs_on_faces(self, cells: np.array, faces: np.array, geometry: FiniteGeometry) -> np.array:
        """The directional diffusion coefficients of the given cells across the given faces, see :meth:`dc_on_face`."""
        directions = np.array([geometry.direction(face) for face in range(int(np.max(faces, initial=0)) + 1)])
        direction = directions[faces]
        return np.einsum("neij,nj,ni->ne", self.dc[cells], direction, direction)

    def dc_on_face(self, cell: int, face: int, geometry: FiniteGeometry) -> float:
        direction = geometry.direction(face)
        return float(np.dot(self.dc[cell] @ direction, direction))
