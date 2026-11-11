"""
Contains the Cartesian Geometry class
"""

from bisect import bisect_right
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dorban.geometry.boundary_conditions import Boundary, Void
from dorban.geometry.geometry import FiniteGeometry
from dorban.geometry.polybox import PolyBox


class Cartesian(FiniteGeometry):
    r"""
    class that represents cartesian geometry.

    Any cartesian geometry can be embedded in a large cube. This class can be
    used to describe a cartesian geometry by defining the large cube and filling
    the difference between the cube and the desired geometry with black absorber.
    The cells of the cube are numbered from left to right then from back to front
    and then from bottom to top, like in the following 2d example::

    4 5 6 7

    0 1 2 3

    The cells of the geometry are numbered by removing the numbers of the black absorbers
    of the cube and using the order of cells of the cube.

    A Cartesian geometry can be transformed to
    :class:`PolyBox <dorban.geometry.polybox.PolyBox>`
    using the to_polybox method.

    Parameters
    ----------
    lengths: Sequence[np.array]
     sequence which contains arrays of the lengths of the cells along each axis.
     The length of this sequence os the dimension of the geometry
    boundary_condition: Sequence[Boundary]
     sequences that contains the type of boundary condition
     on each side (left,right,forward,back,up,down).
     The length of this sequence is twice the dimension of the system.
    black_absorber: Optional[Sequence[int]]
     list that contains the numbers of cells of the cube
     that are filled with black absorber.

    Attributes
    ----------
    x: int
     the number of cells along the x-axis
    y: int
     the number of cells along the y-axis
    z: int
     the number of cells along the z-axis
    cells: int
     the number of the cells, :math:`x\cdot y\cdot z`
    dim: int
     The dimension of the geometry
    boundary: List[int]
     list that contains the numbers of the cells that lay on the boundary.
    neighbors: List[List[int]]
     list that at index `i` has a list of the neighbors of the cell numbered i.

    See Also
    --------
    :class:`PolyBox <dorban.geometry.polybox.PolyBox>`
    """

    def __init__(
        self,
        lengths: Sequence[np.array],
        boundary_condition: Sequence[Boundary],
        black_absorber: Optional[Sequence[int]] = None,
    ):
        self.dim = len(lengths)
        self.lengths = lengths
        self.boundary_conditions = boundary_condition
        self.black_absorber: Sequence[int] = black_absorber or []
        self.x = 1 if self.dim == 0 else len(lengths[0])
        self.y = 1 if self.dim <= 1 else len(lengths[1])
        self.z = 1 if self.dim <= 2 else len(lengths[2])
        self.cells = self.x * self.y * self.z - len(self.black_absorber)
        self.nonblack = np.delete(np.arange(self.x * self.y * self.z), self.black_absorber)
        # coordinates[cell] = (x, y, z) indices of the cell
        self.coordinates = np.stack(np.unravel_index(self.nonblack, (self.x, self.y, self.z), order="F"), axis=1)
        self.boundary = list(np.flatnonzero(self._on_boundary_array()))
        self.neighbors = self._neighbors_calc()
        self.volumes = self._volumes()

    def direction(self, face: int) -> np.array:
        """
        method to compute the direction in space of the unit vector from the
        center of a cell to the face.

        Parameters
        ----------
        face: int
         the number of the face

        Returns
        -------
        np.array
         unit vector that represents the direction to the face
        """
        return _direction(self.dim, face)

    def _reindex(self, cell: int) -> int:
        """
        this function gets a number of a cell assuming there is no black
        absorber in the system and returns the number of the cell in a system
        with black absorber, the indexing skips the cell containing black
        absorber.

        Parameters
        ----------
        cell: int
         the number of cell where the cells are numbered like there is
            no black absorber in the system

        Returns
        -------
        int
         the number of the cell in the new numbering
        """
        return cell - bisect_right(self.black_absorber, cell)

    def _inverse_reindex(self, cell: int) -> int:
        """
        gets a number of a cell in the numbering with black absorber and returns
        the number of the cell in the numbering without black absorber.
        This is the inverse of the reindex method.

        Parameters
        ----------
        cell: int
         the number of the cell in the black absorber indexing

        Returns
        -------
        int
         the number of the cell in the indexing without black absorber
        """
        return self.nonblack[cell]

    def _on_boundary_array(self) -> np.array:
        """Whether each cell is on the boundary of the rectangle (not counting black absorber)."""
        shape = np.array([self.x, self.y, self.z])[: self.dim]
        coordinates = self.coordinates[:, : self.dim]
        return np.any((coordinates == 0) | (coordinates == shape - 1), axis=1)

    def _on_boundary(self, n: int) -> bool:
        """
        checks if the cell whose number is n is on the boundary

        Parameters
        ----------
        n: int
         the number of the cell

        Returns
        -------
        bool
         Whether the cell is on the boundary.
        """
        return bool(self._on_boundary_array()[n])

    def _neighbors_calc(self) -> Sequence[Sequence[int]]:
        """
        function to compute the neighbors of each cell: the number of the neighboring cell, the boundary condition
        on the sides of the rectangle, or a :class:`Void <dorban.geometry.boundary_conditions.Void>` next to
        black absorber.

        Returns
        -------
        Sequence[Sequence[int]]
         list of lists, the list at index i are the neighbors of the cell
         numbered i.
        """
        cube_to_cell = np.full(self.x * self.y * self.z, -1)  # -1 for black absorber
        cube_to_cell[self.nonblack] = np.arange(self.cells)
        shape = (self.x, self.y, self.z)
        strides = (1, self.x, self.x * self.y)
        side = -2  # beyond the sides of the rectangle
        columns = []
        for axis in range(self.dim):
            for step in (-1, 1):
                position = self.coordinates[:, axis] + step
                inside = (position >= 0) & (position < shape[axis])
                neighbor = np.where(inside, self.nonblack + step * strides[axis], 0)
                columns.append(np.where(inside, cube_to_cell[neighbor], side))
        codes = np.stack(columns, axis=1).tolist()
        sides = self.boundary_conditions
        return [
            [n if n >= 0 else (sides[face] if n == side else Void()) for face, n in enumerate(row)] for row in codes
        ]

    def _coordinates(self, cell: int) -> Tuple[int, int, int]:
        """
        computes the x,y,z coordinates of a cell.

        Parameters
        ----------
        cell: int
         the number of the cell

        Returns
        -------
        Tuple[int, int, int]
         (x,y,z) x,y,z being the coordinates of the cell
        """
        return tuple(int(c) for c in self.coordinates[cell])

    def _lengths_cell(self, cell: int) -> Sequence[float]:
        """
        returns the lengths of the edges of the cell

        Parameters
        ----------
        cell:int
         the number of the cell

        Returns
        -------
        Sequence[float]
        """
        return tuple(float(self.lengths[axis][self.coordinates[cell, axis]]) for axis in range(self.dim))

    def _cell_lengths(self) -> np.array:
        """The lengths of the edges of all the cells, an array of shape cells x dim."""
        lengths = [np.asarray(self.lengths[axis], dtype=float)[self.coordinates[:, axis]] for axis in range(self.dim)]
        return np.stack(lengths, axis=1) if lengths else np.ones((self.cells, 0))

    def _volume(self, cell: int) -> float:
        """
        computes the volume of the cell.

        Parameters
        ----------
        cell: int
         the number of the cell
        """
        return self.volumes[cell]

    def _volumes(self) -> np.array:
        """The volumes of all the cells."""
        return np.prod(self._cell_lengths(), axis=1)

    def face_distances(self) -> np.array:
        """
        The distances from the center of each cell to each of its faces, an array of shape cells x faces,
        see :meth:`distance_to_face`.
        """
        return np.repeat(self._cell_lengths() / 2, 2, axis=1)

    def face_areas(self) -> np.array:
        """The areas of the faces of each cell, an array of shape cells x faces, see :meth:`surface_area`."""
        return np.repeat(self.volumes[:, None] / self._cell_lengths(), 2, axis=1)

    def surface_area(self, cell: int, face: int) -> float:
        """
        function to compute the surface area of a face of a cell
        the faces are numbered according to the order
        (left,right,forward,back,up,down)
        face//2 is the number of the axis of the line from the center of the
        cell to the face

        Parameters
        ----------
        cell: int
         the number of the cell
        face: int
         the number of the face in the cell
        """
        return self.volumes[cell] / self.lengths[face // 2][self.coordinates[cell, face // 2]]

    def distance_to_face(self, cell: int, face: int) -> float:
        """
        function computes the distance from the center of the cell to the face
        the faces are numbered according to the order
        (left,right,forward,back,up,down)
        face//2 is the number of the axis of the line from the center of the
        cell to the face

        Parameters
        ----------
        cell:int
         the number of the cell
        face:int
         the number of the face
        """
        return self.lengths[face // 2][self.coordinates[cell, face // 2]] / 2

    def opposite_face(self, face: int) -> int:
        """
        method to compute the number of the face of the cell who is adjacent to
        a given cell along a face

        Parameters
        ----------
        face: int
         the number of the face in a given cell along which
         the two cells are adjacent, a number between 0 and 6

        Returns
        -------
        int
         the number of the face in the other cell, a number between 0 and 6
        """
        return _opposite_face(face)

    def _black_mask(self) -> np.array:
        """Whether each cell of the rectangle is black absorber."""
        mask = np.zeros(self.x * self.y * self.z, dtype=bool)
        mask[list(self.black_absorber)] = True
        return mask

    def _zero_where_black(self, array: np.array):
        r"""
        function that gets an array and returns a new array of length
        :math:`self.x\cdot self.y\cdot self.z` which has zeros where there is black
        absorber
        """
        new_array = np.zeros(tuple([self.x * self.y * self.z] + list(array.shape)[1:]))
        new_array[self.nonblack] = array
        return new_array

    def array_refine_with_black_absorber(self, array: np.array, split: List[Sequence[int]]) -> np.array:
        r"""
        function to refine an array that contains data also about the cells
        that contain black absorber, the length of the array should be
        :math: `self.x \times self.y \times self.z`

        Parameters
        ----------
        array: np.array
         The array that has to be refined
        split: List[Sequence[int]]
         sequence of size the dimension of the system. each of its values
         is a list that specifies to how many parts each cell has to be
         divided in that axis. The length of the cell is equal to the
         number of cells in that axis

        Returns
        -------
        np.array
         new array containing the same information but is suitable to the
         refined geometry
        """
        return _refine_cube(self.x, self.y, self.z, self.dim, array, split)

    def array_refine(self, array: np.array, split: List[Sequence[int]], **kwargs) -> np.array:
        """
        refines an array according to the split parameters. For example this
        function is used in order to refine an array of densities of some
        Isotope

        Parameters
        ----------
        array: np.array
         The array that has to be refined
        split: List[Sequence[int]]
         sequence of size the dimension of the system. each of its values
         is a list that specifies to how many parts each cell has to be
         divided in that axis. The length of the cell is equal to the
         number of cells in that axis

        Returns
        -------
        np.array
         new array containing the same information but is suitable to the
         refined geometry
        """
        array = self._zero_where_black(array)
        refined_with_black = self.array_refine_with_black_absorber(array, split)
        return refined_with_black[self.array_refine_with_black_absorber(~self._black_mask(), split)]

    def refine_mesh(self, split: List[Sequence[int]]) -> "Cartesian":
        """
        This function returns a new Cartesian Geometry with refined mesh
        according to the split data.

        Parameters
        ----------
        split: List[Sequence[int]]
         sequence of size the dimension of the system. each of its values
         is a list that specifies to how many parts each cell has to be
         divided in that axis. The length of the cell is equal to the
         number of cells in that axis

        Returns
        -------
        Cartesian
         new Geometry with a refined mesh
        """
        black_absorber = np.flatnonzero(self.array_refine_with_black_absorber(self._black_mask(), split)).tolist()
        lengths = tuple(map(lambda i: np.repeat(self.lengths[i] / split[i], split[i]), range(self.dim)))
        return self.__class__(lengths, self.boundary_conditions, black_absorber)

    def uniform_split(self, subcells: int) -> Sequence[np.array]:
        """
        Method that creates a split suitable for the geometry, which splits
        each cell into a constant number of cells in each dimension.

        Parameters
        ----------
        subcells: int
         The number of sub cells to split each cell in each dimension.
         Overall the number of cells is multiplied by this parameter to the
         power of the dimension.

        Returns
        -------
        Sequence[np.array]
         Split data to be used with the Cartesian geometry.
        """
        return _uniform_split(subcells, (self.x, self.y, self.z)[: self.dim])

    def to_polybox(self) -> PolyBox:
        """
        creates a PolyBox object that represents the same geometry,
        It is necessary to switch the order of the faces as they are
        ordered differently in PolyBox

        Returns
        -------
        PolyBox
        """
        lengths = [
            [2 * self.distance_to_face(cell, 2 * face) for face in range(self.dim)] for cell in range(self.cells)
        ]
        return PolyBox(self.dim, self.neighbors, lengths)


def _uniform_split(subcells, dimensions):
    """

    Examples
    --------
    >>> _uniform_split(3,(2,3))
    [array([3, 3]), array([3, 3, 3])]
    """
    return [np.full(axis, subcells) for axis in dimensions]


def _opposite_face(face: int) -> int:
    """

    Examples
    --------
    >>> _opposite_face(3)
    2
    """
    return face - (2 * (face % 2) - 1)


def _direction(dim: int, face: int) -> np.array:
    """

    Examples
    --------
    >>> _direction(3,1)
    array([1., 0., 0.])
    """
    d = np.zeros(dim)
    d[face // 2] = 2 * (face % 2) - 1
    return d


def _refine_cube(x: int, y: int, z: int, dim: int, array: np.array, split: List[Sequence[int]]) -> np.array:
    """
    Function used to refine an array along a full cube.

    Parameters
    ----------
    x: int
     the number of cells on the x axis.
    y: int
     the number of cells on the y axis.
    z: int
     the number of cells on the z axis.
    dim: int
     the dimension of the cube
    array:np.array
     the array to refine
    split: List[Sequence[int]]
     The split data,sequence of size the dimension of the system. each of its values
     is a list that specifies to how many parts each cell has to be
     divided in that axis. The length of the cell is equal to the
     number of cells in that axis.


    Returns
    -------
    np.array

    Examples
    --------
    >>> x,y,z=2,2,1
    >>> dim=2
    >>> dc=np.array([[1,2],[3,4],[5,6],[7,8]])
    >>> split=[[2,2],[2,2]]
    >>> _refine_cube(x,y,z,dim,dc,split)
    array([[1, 2],
           [1, 2],
           [3, 4],
           [3, 4],
           [1, 2],
           [1, 2],
           [3, 4],
           [3, 4],
           [5, 6],
           [5, 6],
           [7, 8],
           [7, 8],
           [5, 6],
           [5, 6],
           [7, 8],
           [7, 8]])
    >>> x,y,z=2,3,4
    >>> dim=3
    >>> dc=np.arange(24)
    >>> split=[[2,2],[1,2,3],[3,2,1,2]]
    >>> _refine_cube(x,y,z,dim,dc,split)
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  2,  2,  3,  3,  4,  4,  5,  5,  4,
            4,  5,  5,  4,  4,  5,  5,  0,  0,  1,  1,  2,  2,  3,  3,  2,  2,
            3,  3,  4,  4,  5,  5,  4,  4,  5,  5,  4,  4,  5,  5,  0,  0,  1,
            1,  2,  2,  3,  3,  2,  2,  3,  3,  4,  4,  5,  5,  4,  4,  5,  5,
            4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  8,  8,  9,  9, 10,
           10, 11, 11, 10, 10, 11, 11, 10, 10, 11, 11,  6,  6,  7,  7,  8,  8,
            9,  9,  8,  8,  9,  9, 10, 10, 11, 11, 10, 10, 11, 11, 10, 10, 11,
           11, 12, 12, 13, 13, 14, 14, 15, 15, 14, 14, 15, 15, 16, 16, 17, 17,
           16, 16, 17, 17, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 20,
           20, 21, 21, 22, 22, 23, 23, 22, 22, 23, 23, 22, 22, 23, 23, 18, 18,
           19, 19, 20, 20, 21, 21, 20, 20, 21, 21, 22, 22, 23, 23, 22, 22, 23,
           23, 22, 22, 23, 23])
    """
    split = split + ([1] * (3 - dim))
    cube = np.reshape(array, tuple([x, y, z][:dim] + list(array.shape)[1:]), order="F")
    for i in range(dim):
        cube = np.repeat(cube, split[i], axis=i)
    reshaped = np.reshape(cube, tuple([np.prod([np.sum(s) for s in split])] + list(array.shape)[1:]), order="F")
    return reshaped
