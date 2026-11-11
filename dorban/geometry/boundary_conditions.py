"""
This module contains classes representing different boundary conditions
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from dorban.geometry.geometry import FiniteGeometry


class Boundary(metaclass=ABCMeta):
    """
    Abstract class representing boundary conditions
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_boundary_coefficient(
        self, geometry, E, current_calc, cell: int, face: int
    ) -> Tuple[np.array, np.array, np.array]:
        """
        computes the coefficients of the diffusion matrix that represent
        the current between a cell in the boundary.

        Parameters
        ----------
        geometry: Geometry
         the geometry which this is a boundary of.
        E: int
         the number of energy groups in which the computation is performed
        current_calc: CurrentCalculator
         the :class:`CurrentCalculator <dorban.current_calc.CurrentCalculator>` used to compute the currents.
        cell: int
         the number of the cell near the boundary
        face: int
         the number of the face between the cell and the boundary

        Returns
        -------
        Tuple[np.array, np.array, np.array]
            tuple of the row indices in the diffusion matrix in which the coefficents
            appear, the column indices in the diffusion matrix in which the coefficents
            appear, the values of the diffusion matrix
        """
        raise NotImplementedError


class Reflector(Boundary):
    r"""
    class representing reflective boundary condition. The equation it enforces
    is :math: `J=0` on the boundary.
    """

    def __init__(self):
        super().__init__()

    def compute_boundary_coefficient(
        self, geometry, E, current_calc, cell: int, face: int
    ) -> Tuple[np.array, np.array, np.array]:
        return np.zeros(0), np.zeros(0), np.zeros(0)


def _albedo_boundary_coefficient(albedo, geometry, E, current_calc, cell: int, face: int):
    r"""
    Coefficient of the outgoing current through a face where :math:`J=\alpha\phi_s`, with :math:`\phi_s` the flux
    at the face. The flux at the face is eliminated with the finite difference current between the center of the
    cell and the face, :math:`J=D(\phi-\phi_s)/l`, which gives :math:`J=\phi/(1/\alpha+l/D)` where :math:`l` is the
    distance from the center of the cell to the face.
    """
    distance = geometry.distance_to_face(cell, face)
    return (
        np.arange(cell * E, (cell + 1) * E),
        np.arange(cell * E, (cell + 1) * E),
        geometry.surface_area(cell, face) / (1 / albedo + distance / current_calc.dc_on_face(cell, face, geometry)),
    )


class ExtrapolationLength(Boundary):
    r"""
    class representing void boundary condition using the extrapolated length
    approximation with the 0.4692 constant :math:`J=-0.4692\phi` at the boundary
    """

    constant = 0.4692

    def __init__(self):
        super().__init__()

    def compute_boundary_coefficient(
        self, geometry, E, current_calc, cell: int, face: int
    ) -> Tuple[np.array, np.array, np.array]:
        return _albedo_boundary_coefficient(self.constant, geometry, E, current_calc, cell, face)


class Void(Boundary):
    r"""
    class representing void boundary condition based on the P1 approximation.
    The equation on the current is :math:`J=-\frac{1}{2}\phi` at the boundary
    """

    def __init__(self):
        super().__init__()

    def compute_boundary_coefficient(
        self, geometry, E, current_calc, cell: int, face: int
    ) -> Tuple[np.array, np.array, np.array]:
        return _albedo_boundary_coefficient(0.5, geometry, E, current_calc, cell, face)


class ZeroFlux(Boundary):
    r"""
    class representing zero flux at the boundary. The equation it enforces in
    the finite differences model is :math:`J=-\frac{D}{l}\phi`
    where :math:`D` is the diffusion coefficent on the boundary
    and :math:`l` is the distance between the center of the cell closest to the
    boundary and the boundary itself.
    """

    def __init__(self):
        super().__init__()

    def compute_boundary_coefficient(
        self, geometry, E, current_calc, cell: int, face: int
    ) -> Tuple[np.array, np.array, np.array]:
        s = geometry.surface_area(cell, face)
        distance = geometry.distance_to_face(cell, face)
        return (
            np.arange(cell * E, (cell + 1) * E),
            np.arange(cell * E, (cell + 1) * E),
            np.full(E, s * current_calc.dc_on_face(cell, face, geometry) / distance),
        )


class BluePortal(Boundary):
    """
    This class is used in order to glue two cells together in a periodic
    boundary condition.

    See Also
    --------
    :class:`OrangePortal <dorban.geometry.boundary_conditions.OrangePortal>`
    :func:`~dorban.geometry.boundary_conditions.glue_orange_blue`
    """

    def __init__(self):
        super().__init__()

    def compute_boundary_coefficient(self, geometry, E, current_calc, cell: int, face: int) -> float:
        pass


class OrangePortal(Boundary):
    """
    This class is used in order to glue two cells together in a periodic
    boundary condition.

    See Also
    --------
    :class:`BluePortal <dorban.geometry.boundary_conditions.BluePortal>`
    :func:`~dorban.geometry.boundary_conditions.glue_orange_blue`
    """

    def __init__(self):
        super().__init__()

    def compute_boundary_coefficient(self, geometry, E, current_calc, cell: int, face: int) -> float:
        pass


class CurrentCondition(Boundary):
    """
    This type of boundary condition is a condition on the current, incoming,
    outgoing or combined. It should implement a boundary_current method that
    gives the part of known current.
    """

    def __init__(self, current: np.array):
        """

        Parameters
        ----------
        current: np.array
            the current through the boundary, divided by the area of
            the boundary
        """
        super(CurrentCondition, self).__init__()
        self.current = current

    def compute_boundary_coefficient(
        self, geometry, E, current_calc, cell: int, face: int
    ) -> Tuple[np.array, np.array, np.array]:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    @abstractmethod
    def boundary_current(self, geometry, E: int, current_calc, cell: int, face: int) -> Tuple[np.array, np.array]:
        r"""
        gives the indices and the values of the known part of the current through the boundary, the current entering
        the system through the face times its area. It is the source term this boundary adds to the balance of the
        cell.

        Parameters
        ----------
        geometry: FiniteGeometry
         the geometry of the system
        E:int
         the number of energy groups
        current_calc: CurrentCalculator
         the current calculator of the system, which gives the diffusion coefficients
        cell:int
         the number of the cell whose boundary this is
        face:int
         the number of the face of the cell on the boundary

        Returns
        -------
        Tuple[np.array, np.array]
            tuple of arrays whose first argument gives the indices of the current
            and the second the values of the current
        """
        pass


class KnownCurrent(CurrentCondition):
    """
    This class is used in order to find the flux that yields a given current on
    the boundary.
    From the point of view of the diffusion matrix it is the same like reflective
    boundary condition, but it also allows finding the flux which yields a given
    current on the boundary.
    The current is the net current entering the system per unit area, positive inwards.
    """

    def boundary_current(self, geometry, E: int, current_calc, cell: int, face: int) -> Tuple[np.array, np.array]:
        return np.arange(cell * E, (cell + 1) * E), geometry.surface_area(cell, face) * self.current


class IncomingCurrent(CurrentCondition):
    r"""
    This class is used in order to solve a fixed source problem when the source
    is outside of the system.
    The current is the incoming partial current :math:`J^-` per unit area, positive inwards, as for
    :class:`KnownCurrent`.

    With the Marshak condition :math:`J^-=\phi_s/4-J/2` at the face, :math:`J` being the outgoing net current and
    :math:`\phi_s` the flux at the face, and the finite difference current between the center of the cell and the
    face :math:`J=D(\phi-\phi_s)/l`, the outgoing net current is

    .. math:: J=\frac{\phi}{2+l/D}-\frac{2J^-}{1+l/(2D)}

    The first term is the coefficient of :class:`Void`, the second the known current entering the cell.
    """

    def compute_boundary_coefficient(
        self, geometry, E, current_calc, cell: int, face: int
    ) -> Tuple[np.array, np.array, np.array]:
        return _albedo_boundary_coefficient(0.5, geometry, E, current_calc, cell, face)

    def boundary_current(self, geometry, E: int, current_calc, cell: int, face: int) -> Tuple[np.array, np.array]:
        distance = geometry.distance_to_face(cell, face)
        entering = 2 * self.current / (1 + distance / (2 * current_calc.dc_on_face(cell, face, geometry)))
        return np.arange(cell * E, (cell + 1) * E), geometry.surface_area(cell, face) * entering


def glue_orange_blue(geometry: "FiniteGeometry", orange_indices: Sequence[int], blue_indices: Sequence[int]):
    r"""
    Function that glues the :class:`OrangePortal <dorban.geometry.boundary_conditions.OrangePortal>`
    with the :class:`OrangePortal <dorban.geometry.boundary_conditions.BluePortal>`.
    The gluing is done in place and edits the neighbors property of the geometry object.
    The gluing is done according to the order of cell indices in the supplied orange_indices
    and blue_indices sequences.

    Warnings
    --------
    The gluing process might not commute with refinement, to be safe the gluing
    should be preformed after the refinement process.

    Parameters
    ----------
    geometry: FiniteGeometry
     A geometry which has an equal amount of BluePortal and Orange Portal boundary conditions
    orange_indices: Sequence[int]
     A sequence of the indices of cells with OrangePortal boundary condition.
    blue_indices: Sequence[int]
     A sequence of the indices of cells with BluePortal boundary condition.

    """
    neighbors = geometry.neighbors
    for index, cell in enumerate(blue_indices):
        neighbor = orange_indices[index]
        neighbors[cell] = [n if not isinstance(n, BluePortal) else neighbor for n in neighbors[cell]]
        neighbors[neighbor] = [n if not isinstance(n, OrangePortal) else cell for n in neighbors[neighbor]]
