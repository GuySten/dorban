"""
This module contains the triangular tessellation class.
This type of finite geometry is used in hexagonal and triangular grids,
where the 2d planes are made out of triangles and the cells are prisms along
the third axis.
"""

from bisect import bisect_right
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
from multipledispatch import dispatch

from dorban.geometry.boundary_conditions import Boundary, Void
from dorban.geometry.geometry import FiniteGeometry
from dorban.utils import mink_sum

ThreeD = Tuple[int, Sequence[int]]


class TriangularTessellation(FiniteGeometry):
    """
    This class represent any geometry made out of equilateral triangular prisms
    where the triangular part is in the x,y plane.
    It doesn't assume that the overall shape of the core is simple or even
    known. Instead, everything is based on the given connections between cells.
    The numbering of the cells in the geometry is determined by the numbers of
    the cells in the plane_neighbors parameter and the differences between two
    cells which are exactly above each other is the numbers of cells in a plenary slice.

    Parameters
    ----------
    tri: int
     The number of triangles in each plane.
    tri_edge: float
     The size in cm of the edge of each equilateral triangle
    plane_neighbors: Sequence[Sequence[int]]
     A sequence where each item at index `i` corresponds to
     the indices of the neighbors of cell `i` along the plane (x-y).
    z_lengths: Sequence[float]
     The subdivision of the prisms along the prismatic direction.
    boundary_conditions:
     Tuple of boundary conditions below and above the geometry, used only in 3d

    Attributes
    ----------
    dim: int
     The dimension of the system
    cells: int
     the total number of cells in the system

    """

    opposite_faces_dict = {0: 1, 1: 0, 2: 2, 3: 4, 4: 3}

    def __init__(
        self,
        tri: int,
        tri_edge: float,
        plane_neighbors: Sequence[Sequence[int]],
        z_lengths: Sequence[float] = (1.0,),
        boundary_conditions: Optional[Tuple[Boundary, Boundary]] = None,
    ):
        if len(plane_neighbors) != tri:
            raise ValueError(
                f"The neighbors list should be of length tri={tri} but it was {len(plane_neighbors)} instead"
            )
        self.dim = 2 if len(z_lengths) == 1 else 3
        self.boundary_conditions = boundary_conditions
        self.z = 1 if self.dim <= 2 else len(z_lengths)
        self.tri = tri
        self.tri_edge = tri_edge
        self.cells = self.tri * self.z
        self.plane_neighbors = plane_neighbors
        self._distance_to_face_in_plane = tri_edge / (2 * np.sqrt(3))
        self._triangle_area = np.sqrt(3) / 4 * tri_edge**2
        self.lengths = np.array(z_lengths)
        self._z_areas = self.lengths * tri_edge
        self._z_distances = self.lengths / 2
        self.volumes = np.repeat(self.lengths * self._triangle_area, tri)
        if self.dim == 2:
            self.neighbors = plane_neighbors
        else:
            self.neighbors = [
                [
                    n if isinstance(n, Boundary) else n % self.tri + self.tri * (i // self.tri)
                    for n in plane_neighbors[i % self.tri]
                ]
                + [
                    i - self.tri if i >= self.tri else boundary_conditions[0],
                    i + self.tri if i < self.cells - self.tri else boundary_conditions[1],
                ]
                for i in range(self.cells)
            ]

    def _z_coordinates(self, cell: int) -> int:
        """
        Computes the z coordinates of a cell

        Parameters
        ----------
        cell: int
         The cell number

        Returns
        -------
        int
         The z coordinates of the cell
        """
        return cell // self.tri

    def surface_area(self, cell: int, face: int) -> float:
        r"""
        Function to compute the surface area of a cell's face.

        Parameters
        ----------
        cell: int
         the number of the cell
        face: int
         The number of the face, the faces are numbered like this:
         Assuming that the orientation of the cell in the x-y plane is like
         :math:`\Delta`

         0 - the / face, 1 - the \ face, 2 - the _ face
         3 - the bottom face, 4 - the upper face
        """
        return _surface_area(face, self._triangle_area, self._z_areas, self._z_coordinates(cell))

    def distance_to_face(self, cell: int, face: int) -> float:
        r"""
        Function that computes the distance from the center of the cell to the face

        Parameters
        ----------
        cell: int
         the number of the cell
        face: int
         the number of the face, the faces are numbered like this:
         Assuming that the orientation of the cell in the x-y plane is like
         :math:`\Delta`

         0 - the / face, 1 - the \ face ,2 - the _ face
         3 - the bottom face, 4 - the upper face
        """
        return _distance(face, self._z_distances[self._z_coordinates(cell)], self._distance_to_face_in_plane)

    def opposite_face(self, face: int) -> int:
        """
        Method to compute the number of the face of the cell who is adjacent to
        a given cell along a given face

        Parameters
        ----------
        face: int
         The number of the face in a given cell along which the two cells are
         adjacent. A number between 0 and 4

        Returns
        -------
        int
         The number of the face in the other cell, a number between 0 and 4
        """
        return self.opposite_faces_dict[face]

    def direction(self, face: int) -> np.array:
        """
        Method to compute the direction in space of the unit vector from the
        center of a cell to the face

        Parameters
        ----------
        face: int
         The face number.

        Returns
        -------
        np.array
         A unit vector that represents the direction to the face
        """
        if face in {3, 4}:
            return (1 - 2 * (face % 2)) * np.array([0, 0, 1])
        return np.array([[-1 / 2, np.sqrt(3) / 2], [-np.sqrt(3) / 2, -1 / 2]]) ** (2 - face) @ np.array([0, -1])

    def array_refine(self, array: np.array, split: Union[Tuple[int], ThreeD], **kwargs) -> np.array:
        """
        Refines an array according to the split parameters. For example this
        function is used in order to refine an array of densities of some
        Isotope.

        Parameters
        ----------
        array: np.array
         The array which has to be converted to a geometry with a refined mesh
        split: Union[Tuple[int], Tuple[int, Sequence[int]]]
          - If the dimension is two it's just the number of triangles into which each triangle is divided.
          - If the dimension is three it is a tuple whose first value is the 2d split, and second value is
            a sequence whose length is the number of x-y planes and whose i-th value is the
            amount of planes to which the i-th plane should be divided.

        Returns
        -------
        np.array
         new array containing the same information but is suitable to the
         refined geometry
        """
        return _array_refine(array, split, self.dim, self.tri, **kwargs)

    def single_cell(self, cell: int, boundary_conditions):
        return self.__class__(
            1,
            self.tri_edge,
            [boundary_conditions[:3]],
            self.lengths[self._z_coordinates(cell)],
            boundary_conditions[3:] if len(boundary_conditions) > 3 else None,
        )

    def uniform_split(self, subcells: int) -> Union[Tuple[int], Tuple[int, Sequence[int]]]:
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
        Union[Tuple[int], Tuple[int, Sequence[int]]]
         Split data to be used with the Triangular tesselation geometry.
        """
        return (subcells, [subcells] * self.z) if self.dim == 3 else (subcells,)

    def simple_boundaries(self, dim: int, size: int) -> Sequence[Sequence[int]]:
        triangle = Triangle(size)
        if dim == 2:
            return triangle.boundaries
        return [mink_sum(boundary, size**2 * np.arange(0, self.z)) for boundary in triangle.boundaries] + [
            list(range(size**2)),
            list(range((self.z - 1) * size**2, self.z * size**2)),
        ]

    @dispatch(int)
    def _refine_length_z(self, split: int):
        return self.lengths

    @dispatch(np.ndarray)
    def _refine_length_z(self, split: np.array):
        return np.repeat(self.lengths / split, split)

    def refine_mesh(self, split: Union[Tuple[int], ThreeD]) -> "TriangularTessellation":
        """
        This function returns a new TriangularTessellation Geometry
        with refined mesh according to the split data.

        Parameters
        ----------
        split: Union[Tuple[int], ThreeD]
         The splitting data. If the dimension is two it's just the number of
         triangles to which each triangle is divided.
         If the dimension is 3 it has an additional value which is a sequence
         whose length is the number of x-y planes and whose i-th value is the
         amount of planes to which the i-th plane should be divided

        Returns
        -------
        TriangularTessellation
         new Geometry with a refined mesh
        """
        sub_triangles = split[0] ** 2
        triangles = self.tri * sub_triangles
        triangle_size = self.tri_edge / split[0]
        lengths_z = self._refine_length_z(split[-1])
        plane_neighbors = list(range(triangles))
        triangle = Triangle(split[0])
        for sub_cell, neighbors in enumerate(triangle.neighbors):
            for cell in range(self.tri):
                cell_neighbor = []
                for i in range(3):
                    try:
                        cell_neighbor.append(cell * sub_triangles + neighbors[i])
                    except TypeError:
                        if isinstance(self.neighbors[cell][i], Boundary):
                            cell_neighbor.append(self.neighbors[cell][i])
                        else:
                            cell_neighbor.append(self.neighbors[cell][i] * sub_triangles + neighbors[i][1])
                plane_neighbors[cell * sub_triangles + sub_cell] = cell_neighbor
        return TriangularTessellation(triangles, triangle_size, plane_neighbors, lengths_z, self.boundary_conditions)


def _distance(face, z_distance, distance_plane):
    return z_distance if face in {3, 4} else distance_plane


def _surface_area(face, triangle_area, z_area, cor):
    return triangle_area if face in {3, 4} else z_area[cor]


Neighbor = Union[Tuple[Boundary, int], int]
Tribound = Tuple[Sequence[int], Sequence[int], Sequence[int]]


def _calc_boundaries(tri: int) -> Tribound:
    """
    This function computes the boundary of the triangle. There are 3 types
    of boundary according to the directions: (left, right, straight)

    Returns
    -------
    Tuple[Sequence[int],Sequence[int],Sequence[int]]
     List that contains the numbers of the cells on each boundary

    Examples
    --------
    >>> _calc_boundaries(2)
    ([0, 3], [2, 3], [0, 2])
    """
    bottom_boundary = list(range(0, 2 * tri, 2))
    left_boundary = [2 * tri * i - i**2 for i in range(0, tri)]
    right_boundary = [2 * tri * (i + 1) - (i**2 + 2 * i + 2) for i in range(0, tri)]
    return left_boundary, right_boundary, bottom_boundary


def _array_refine(
    array: np.array, split: Union[Tuple[int], Tuple[int, Sequence[int]]], dim: int, tri: int, **kwargs
) -> np.array:
    """
    Examples
    --------
    >>> array=np.arange(8)
    >>> split=(2,[3,1])
    >>> _array_refine(array,split,3,4)
    array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1,
           1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
           3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7])
    >>> array=np.arange(4)
    >>> split=(2,)
    >>> _array_refine(array,split,2,4)
    array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    >>> array=np.array([[0,1],[2,3],[4,5],[6,7]])
    >>> split=(2,)
    >>> _array_refine(array,split,2,4,axis=0)
    array([[0, 1],
           [0, 1],
           [0, 1],
           [0, 1],
           [2, 3],
           [2, 3],
           [2, 3],
           [2, 3],
           [4, 5],
           [4, 5],
           [4, 5],
           [4, 5],
           [6, 7],
           [6, 7],
           [6, 7],
           [6, 7]])
    >>> array=np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]])
    >>> split=(2,[3,1])
    >>> _array_refine(array,split,3,4,axis=0)
    array([[ 0,  1],
           [ 0,  1],
           [ 0,  1],
           [ 0,  1],
           [ 2,  3],
           [ 2,  3],
           [ 2,  3],
           [ 2,  3],
           [ 4,  5],
           [ 4,  5],
           [ 4,  5],
           [ 4,  5],
           [ 6,  7],
           [ 6,  7],
           [ 6,  7],
           [ 6,  7],
           [ 0,  1],
           [ 0,  1],
           [ 0,  1],
           [ 0,  1],
           [ 2,  3],
           [ 2,  3],
           [ 2,  3],
           [ 2,  3],
           [ 4,  5],
           [ 4,  5],
           [ 4,  5],
           [ 4,  5],
           [ 6,  7],
           [ 6,  7],
           [ 6,  7],
           [ 6,  7],
           [ 0,  1],
           [ 0,  1],
           [ 0,  1],
           [ 0,  1],
           [ 2,  3],
           [ 2,  3],
           [ 2,  3],
           [ 2,  3],
           [ 4,  5],
           [ 4,  5],
           [ 4,  5],
           [ 4,  5],
           [ 6,  7],
           [ 6,  7],
           [ 6,  7],
           [ 6,  7],
           [ 8,  9],
           [ 8,  9],
           [ 8,  9],
           [ 8,  9],
           [10, 11],
           [10, 11],
           [10, 11],
           [10, 11],
           [12, 13],
           [12, 13],
           [12, 13],
           [12, 13],
           [14, 15],
           [14, 15],
           [14, 15],
           [14, 15]])
    """
    sub_triangles = split[0] ** 2
    plane_refine = np.repeat(array, sub_triangles, **kwargs)
    if dim == 2:
        return plane_refine
    return np.concatenate(
        [
            np.concatenate([plane_refine[tri * sub_triangles * i : tri * sub_triangles * (i + 1)]] * sub_cells)
            for i, sub_cells in enumerate(split[1])
        ]
    )


class Triangle:
    """
    class that represents the cells of  a triangle divided to several
    sub triangles, It is used when working with the triangular tesellation
    geometry
    The assumed orientation of the triangle is::
            0
           0 0
          0 0 0
         0 0 0 0
        0 0 0 0 0

    Parameters
    ----------
    triangles: int
     the number of small triangles in each row of
     the large triangle

    Attributes
    ----------
    tri: int
     the number of triangles in each level.
    boundaries: Tuple[Sequence[int], Sequence[int], Sequence[int]]
     tuple that contains arrays with the numbers of the cells at the
     boundary, it is divided to 3 arrays corresponding to 3 possible
     boundary directions. left,right,straight
    boundary: np.array
     list of all possible boundaries
    neighbors: List[List[int]]
     list that at place `i` has the neighbors of the cell numbered `i`
    """

    def __init__(self, triangles):
        self.row = triangles
        self.tri = self.row**2
        self.boundaries: Tuple[Sequence[int], Sequence[int], Sequence[int]] = _calc_boundaries(triangles)
        self.boundary = np.hstack(self.boundaries)
        self.neighbors = self.calc_neighbors()

    def bound_correct(self, inside_neigh: Tuple[int, int, int], cell: int) -> Tuple[Neighbor, Neighbor, Neighbor]:
        """
        Returns a tuple of the neighbors of a given cell

        Parameters
        ----------
        inside_neigh: Tuple[int, int, int]
         the neighbors of the cell if the cell is not on the boundary
        cell: int
         The cell number
        """
        left, right, straight = inside_neigh
        left_boundary = self.boundaries[0]
        right_boundary = self.boundaries[1]
        bottom_boundary = self.boundaries[2]
        if cell in left_boundary:
            left = (Void(), self.reflect_over_face(cell, 0))
        if cell in right_boundary:
            right = (Void(), self.reflect_over_face(cell, 1))
        if cell in bottom_boundary:
            straight = (Void(), cell)
        return left, right, straight

    def inside_neigh(self, tri_index: int) -> Tuple[int, int, int]:
        """
        Computes the neighbors of a cell which is not on the boundary

        Parameters
        ----------
        tri_index: int
         The cell number

        Returns
        -------
        Tuple[int, int, int]
         tuple that contains the numbers of the neighbors of the cell
        """
        left_boundary = self.boundaries[0]
        row = bisect_right(left_boundary, tri_index)
        orientation = (tri_index - row) % 2
        right, left = tri_index + 1, tri_index - 1
        straight = tri_index - (self.row - row + 1) * 2 if orientation else tri_index + (self.row - row) * 2
        return left, right, straight

    def calc_neighbors(self) -> Sequence[Sequence[int]]:
        """
        Function to compute the neighbors of each cell. If a cell is on the boundary,
        its neighbor in this direction will of th form (Boundary, cell) with
        cell being the number of the neighboring cell inside an adjacent
        triangle if such a triangle existed.

        Returns
        -------
        Sequence[Sequence[int]]
         list of lists, the list at index `i` are the neighbors of the cell
         numbered `i`
        """
        return [
            self.bound_correct(self.inside_neigh(i), i) if i in self.boundary else self.inside_neigh(i)
            for i in range(self.tri)
        ]

    def reflect_over_face(self, cell: int, face: Literal[0, 1, 2]) -> int:
        """
        This function computes the number of the cell in a triangle adjoining
        self along a face that we get by reflecting the cell along this face

        Parameters
        ----------
        cell: int
         the number of the cell we want to reflect
        face: int
         the number of the face we reflect along, has to be one of 0,1,2

        Returns
        -------
        int
         The number of the cell in the adjoining triangle
        """
        if face == 2:
            return cell
        left_boundary = self.boundaries[0]
        row = bisect_right(left_boundary, cell)
        place_in_row = cell - left_boundary[row - 1]
        if face == 0:
            row_after_ref = self.row - row + 1 - (1 + place_in_row) // 2
            place_after_reflection = 2 * (self.row - row_after_ref) - place_in_row
        else:  # face == 1
            row_after_ref = self.row - row + 1 - (2 * (self.row - row + 1) - 1 - place_in_row) // 2
            place_after_reflection = 2 * (self.row - row + 1) - 2 - place_in_row
        return left_boundary[row_after_ref - 1] + place_after_reflection


def hex2triangles(neighbors: Sequence[Sequence[int]]) -> Sequence[Sequence[int]]:
    r"""
    Function gets a list of neighbors of hexagonal cells and transforms it to
    a corresponding list of neighbors of triangular cells, by dividing each
    hexagon to 6 triangles::
       _____
     /5\4/3\
     --------
     \0/1\2/
      ------

    Parameters
    ----------
    neighbors: Sequence[Sequence[int]]
     list that at index `i` has a list of the neighbors of the
     hexagon numbered `i`

    Returns
    -------
    Sequence[Sequence[int]]
     list that at index `i` has a list of the neighbors of
     the triangle numbered `i`

    Examples
    --------
    >>> from dorban.utils import triangles_with_vertex
    >>> hex_neighbors = [[1, Void(), Void(), Void(), Void(), Void()],[Void(), Void(), Void(), 0, Void(), Void()]]
    >>> neighbors=hex2triangles(hex_neighbors)
    >>> triangles_with_vertex(1, 0, neighbors)
    {2}
    >>> triangles_with_vertex(1, 2, neighbors)
    {0, 2, 3, 4, 5}
    """

    return [
        _calc_hex_neighbors(neighbor_hex, neighbors, hexagon, index)
        for hexagon, neighbor_hex in enumerate(neighbors)
        for index in range(6)
    ]


def _calc_hex_neighbors(
    neighbors_hex: Sequence[int], neighbors: Sequence[Sequence[int]], hexagon: int, index: Literal[0, 1, 2, 3, 4, 5]
):
    boundary = [i for i, n in enumerate(neighbors_hex) if isinstance(n, Boundary)]
    neighbors_hex = np.array(neighbors_hex)
    neighbors_hex[boundary] = 0
    cell = hexagon * 6
    neigh = [
        [neighbors_hex[0] * 6 + 3, cell + 1, cell + 5],
        [cell, cell + 2, neighbors_hex[1] * 6 + 4],
        [cell + 1, neighbors_hex[2] * 6 + 5, cell + 3],
        [cell + 4, neighbors_hex[3] * 6, cell + 2],
        [cell + 5, cell + 3, neighbors_hex[4] * 6 + 1],
        [neighbors_hex[5] * 6 + 2, cell + 4, cell],
    ]
    outside = (0, 2, 1, 1, 2, 0)
    for b in boundary:
        neigh[b][outside[b]] = neighbors[hexagon][b]
    return neigh[index]
