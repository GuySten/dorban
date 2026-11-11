"""
Contains the Cylindrical geometry class, the R-Z geometry of a cylinder that is symmetric around its axis, or the R
geometry of an infinitely long cylinder.
"""

from typing import Optional, Sequence

import numpy as np

from dorban.geometry.boundary_conditions import Boundary, Reflector
from dorban.geometry.cartesian import Cartesian


class Cylindrical(Cartesian):
    r"""
    Class that represents a cylinder that is symmetric around its axis, in R-Z (two dimensions) or R (one dimension,
    an infinitely long cylinder per unit of height).

    The cells are rings: the first axis is the radius r, starting at the axis r = 0, and the second is the height z.
    The cells are numbered and refined as in :class:`~dorban.geometry.cartesian.Cartesian`, with x replaced by r and y
    by z, and parts of the rectangle can be filled with black absorber in the same way.
    Only the volumes and the areas of the faces differ: a cell between the radii :math:`r_0<r_1` of height
    :math:`h` has the volume :math:`\pi(r_1^2-r_0^2)h`, its radial faces the areas :math:`2\pi r_0h` and
    :math:`2\pi r_1h` and its axial faces the area :math:`\pi(r_1^2-r_0^2)`.
    The distance from the center of the cell to its faces is half its length, as in the finite difference box
    scheme with the mesh points at the middle of the mesh intervals.

    Parameters
    ----------
    lengths: Sequence[np.array]
     sequence of the arrays of the lengths of the cells along r and, in R-Z, along z.
    boundary_condition: Sequence[Boundary]
     the boundary conditions at r = 0, at the outer radius and, in R-Z, at the bottom and at the top.
     The face at r = 0 has no area, so the boundary condition there must be a Reflector.
    black_absorber: Optional[Sequence[int]]
     the numbers of the cells of the rectangle that are filled with black absorber,
     see :class:`~dorban.geometry.cartesian.Cartesian`.
    """

    def __init__(
        self,
        lengths: Sequence[np.array],
        boundary_condition: Sequence[Boundary],
        black_absorber: Optional[Sequence[int]] = None,
    ):
        if len(lengths) not in (1, 2):
            raise ValueError("A Cylindrical geometry has one (R) or two (R-Z) dimensions")
        if not isinstance(boundary_condition[0], Reflector):
            raise ValueError("The boundary condition at r = 0 must be a Reflector")
        self.radii = np.hstack(([0.0], np.cumsum(lengths[0])))
        super().__init__(lengths, boundary_condition, black_absorber)

    def _radii(self, cell: int) -> tuple[float, float]:
        """The inner and outer radius of the cell."""
        r = self._coordinates(cell)[0]
        return self.radii[r], self.radii[r + 1]

    def _height(self, cell: int) -> float:
        return 1.0 if self.dim == 1 else self.lengths[1][self._coordinates(cell)[1]]

    def _volume(self, cell: int) -> float:
        """
        computes the volume of the cell.

        Parameters
        ----------
        cell: int
         the number of the cell
        """
        inner, outer = self._radii(cell)
        return np.pi * (outer**2 - inner**2) * self._height(cell)

    def surface_area(self, cell: int, face: int) -> float:
        """
        function to compute the surface area of a face of a cell, the faces are numbered in the order
        (inner, outer, bottom, top).

        Parameters
        ----------
        cell: int
         the number of the cell
        face: int
         the number of the face in the cell
        """
        if face >= 2:
            inner, outer = self._radii(cell)
            return np.pi * (outer**2 - inner**2)
        return 2 * np.pi * self._radii(cell)[face] * self._height(cell)

    def to_polybox(self):
        raise NotImplementedError("A Cylindrical geometry has no PolyBox equivalent")
