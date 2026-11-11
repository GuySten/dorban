"""
This module contains the PolyBox geometry. It can be used to model cores with
a Cartesian grids.
This also can be used to model multiplicative systems in non euclidean geometries
"""

from functools import cached_property
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from dorban.geometry.boundary_conditions import Boundary
from dorban.geometry.geometry import FiniteGeometry


class PolyBox(FiniteGeometry):
    """
    This class represents a geometry that is made out of boxes in dimension 1,2 or 3.
    A special case of a polybox geometry is a cartesian geometry,
    but the polybox geometry is more versatile.
    It is local, meaning each cell knows only about its neighbors.
    No global structure is assumed.

    Parameters
    ----------
    dim:int
     the dimension of the system, should be one of 1,2,3
    neighbors:
     sequence that at place `i` has a sequence of the neighbors
     of the cell numbered i, the length of this sequence is twice
     the dimension and the neighbors are ordered according to:
     left, right, back, forward, down, up
    lengths: Sequence[Sequence[float]]
     Sequence of sequences, at place i it contains a sequence of the
     lengths of cell numbered i. Its length is the dimension, and it is ordered
     according to: x,y,z.
    """

    def __init__(
        self, dim: int, neighbors: Sequence[Sequence[Union[int, Boundary]]], lengths: Sequence[Sequence[float]]
    ):
        self.dim = dim
        self.lengths = lengths
        self.neighbors = neighbors
        self.cells = len(neighbors)
        self.volumes = np.prod(self.lengths, axis=1)

    def surface_area(self, cell: int, face: int) -> float:
        """
        computes the surface area of a face of a cell.

        Parameters
        ----------
        cell: int
         the number of the cell
        face: int
         the number of the face according to the order:
         left, right, back, forward, down, up
        """
        return self.volumes[cell] / self.lengths[cell][face // 2]

    def distance_to_face(self, cell: int, face: int) -> float:
        """
        computes the distance from the center of the cell to the face

        Parameters
        ----------
        cell: int
         the number of the cell
        face: int
         the number of the face according to the order:
         left, right, back, forward, down, up
        """
        return self.lengths[cell][face // 2] / 2

    def face_distances(self) -> np.array:
        """The distances from the center of each cell to its faces, an array of shape cells x faces."""
        return np.repeat(np.asarray(self.lengths, dtype=float) / 2, 2, axis=1)

    def face_areas(self) -> np.array:
        """The areas of the faces of each cell, an array of shape cells x faces."""
        return np.repeat(self.volumes[:, None] / np.asarray(self.lengths, dtype=float), 2, axis=1)

    def opposite_face(self, face: int) -> int:
        """
        method to compute the number of the face of the cell who is adjacent to
        a given cell along a face.

        Parameters
        ----------
        face:int
         the number of the face in a given cell along which
         the two cells are adjacent, a number between 0 and 6

        Returns
        -------
        int
         the number of the face in the other cell, a number between 0 and 6
        """
        return face - (2 * (face % 2) - 1)

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
        unit = np.zeros(self.dim)
        unit[face // 2] = 2 * (face % 2) - 1
        return unit

    def array_refine(self, array: np.array, split: Sequence[Sequence[int]], **kwargs) -> np.array:
        """
        refines an array according to the split parameters. For example this
        function is used in order to refine an array of densities of some
        Isotope

        Parameters
        ----------
        array: np.array
         array whose length is the number of cells.
        split: Sequence[Sequence[int]]
         sequence which at place `i` has a sequence of lengths the dimension
         that specifies o how many each axis has to be splatted

        Returns
        -------
        np.array
         new array containing the same information but is suitable to the
         refined geometry
        """
        return np.repeat(array, [np.prod(splitting) for splitting in split], **kwargs)

    def single_cell(self, cell, boundary_conditions):
        return self.__class__(
            self.dim,
            [
                boundary_conditions,
            ],
            [self.lengths[cell]],
        )

    def uniform_split(self, subcells: int) -> Sequence[Sequence[int]]:
        """
        Method that creates a split suitable for the geometry, which splits
        each cell into a constant number of cells in each dimension.

        Parameters
        ----------
        subcells:int
         The number of sub cells to split each cell in each dimension.
         Overall the number of cells is multiplied by this parameter to the
         power of the dimension.

        Returns
        -------
        Sequence[Sequence[int]]
         Split data to be used with the PolyBox geometry.
        """
        return [[subcells] * self.dim] * self.cells

    def simple_boundaries(self, dim, size) -> Sequence[Sequence[int]]:
        box = Box([size] * dim, boundary_conditions=[Boundary] * (2 * dim))
        return box.boundaries

    def _refine_lengths(self, split: Sequence[Sequence[int]]) -> Sequence[Sequence[float]]:
        """
        Function that used in the refine_mesh method, used to refine the lengths
        of the system.

        Parameters
        ----------
        split Sequence[Sequence[int]]
         Sequence of sequences, the sequence at index i is of length
         the dimension of the geometry, and it specifies to how many
         cells in each direction the cell numbered `i` should be split


        Returns
        -------
        Sequence[Sequence[float]]
         The lengths for the refined system
        """
        lengths = [
            [length / splits for length, splits in zip(cell_lengths, cell_splits)]
            for cell_splits, cell_lengths in zip(split, self.lengths)
        ]
        return self.array_refine(lengths, split, axis=0)

    def _refine_neighbors(self, split: Sequence[Sequence[int]]) -> Sequence[Sequence[int]]:
        """
        Function that used in the refine_mesh method, used to refine the neighbors
        of the system.

        Parameters
        ----------
        split Sequence[Sequence[int]]
         Sequence of sequences, the sequence at index i is of length
         the dimension of the geometry, and it specifies to how many
         cells in each direction the cell numbered `i` should be split


        Returns
        -------
        Sequence[Sequence[int]]
         The neighbors for the refined system
        """

        cells_per_cell = np.array([np.prod(splitting) for splitting in split])
        cells = np.sum(cells_per_cell)
        refined_neighbors = list(range(int(cells)))
        up_to_cells = np.hstack([[0], np.cumsum(cells_per_cell)[:-1]])
        for big_cell in range(self.cells):
            box = Box(
                tuple(split[big_cell]),
                [None if isinstance(neigh, Boundary) else split[neigh] for neigh in self.neighbors[big_cell]],
            )
            for small_cell in range(box.cells):
                refined_neighbors[up_to_cells[big_cell] + small_cell] = [
                    neighbor + up_to_cells[big_cell]
                    if not isinstance(neighbor, Tuple)
                    else (
                        big_neighbor
                        if isinstance(big_neighbor := self.neighbors[big_cell][face], Boundary)
                        else neighbor[1] + up_to_cells[big_neighbor]
                    )
                    for face, neighbor in enumerate(box.neighbors[small_cell])
                ]

        return refined_neighbors

    def refine_mesh(self, split: Sequence[Sequence[int]]) -> "PolyBox":
        """
        This function returns a new Geometry with refined mesh according to the
        split data. The split data is unique to each geometry type, and
        it is specified in the geometries refine array property.

        Parameters
        ----------
        split: Sequence[Sequence[int]]
         Sequence of sequences, the sequence at index i is of length
         the dimension of the geometry, and it specifies to how many
         cells in each direction the cell numbered `i` should be split

        Returns
        -------
        PolyBox
         new Geometry with a refined mesh
        """
        return self.__class__(self.dim, self._refine_neighbors(split), self._refine_lengths(split))


Neighbor = Union[int, Boundary, Tuple[Boundary, int]]


class Box:
    """
    This class represents a single box. This class is used for index calculations
    in the refinement of polybox geometry

    Parameters
    ----------
    sizes: Sequence[int]
     a sequence which contains the amount of cells on each axis.
     Its length is the dimension of the Box.
    sizes_of_neighbors:
     sequence of the sizes of the boxes around this box,
     in the same format as sizes
    boundary_conditions:
     the boundary conditions outside the box
    Exactly 1 of sizes of neighbors and boundary conditions should be given
    """

    def __init__(
        self,
        sizes: Sequence[int],
        sizes_of_neighbors: Optional[Sequence[Sequence[int]]] = None,
        boundary_conditions: Optional[Sequence[Boundary]] = None,
    ):
        if (sizes_of_neighbors, boundary_conditions).count(None) != 1:
            raise ValueError(
                "Exactly one of sizes_of_neighbors and "
                "boundary_conditions should be given. "
                "Instead there were"
                f"{[sizes_of_neighbors, boundary_conditions].count(None)}"
            )
        self.dim = len(sizes)
        self.cells = np.prod(sizes)
        self.sizes = sizes
        self.sizes_of_neighbors = sizes_of_neighbors
        self.boundary_conditions = boundary_conditions
        axises = ["x", "y", "z"]
        self.x, self.y, self.z = 1, 1, 1
        for ax, size in zip(axises, sizes):
            self.__setattr__(ax, size)
        # on_face[face][cell]: whether the cell is on that face of the box
        self.on_face = np.zeros((2 * self.dim, self.cells), dtype=bool)
        for face, cells in enumerate(self.boundaries):
            self.on_face[face, cells] = True
        self.neighbors: Sequence[Sequence[Neighbor]] = [
            self.correct_neighbors_of_boundary_cell(self.inside_neighbors(cell), cell) for cell in range(self.cells)
        ]

    def inside_neighbors(self, cell: int) -> Sequence[int]:
        """
        Computes the numbers of the neighbors of a cell not on the boundary

        Parameters
        ----------
        cell: int
         the number of the cell

        Returns
        -------
        Sequence[int]
         Sequence of the numbers of the neighbors of the cell
        """
        parts = ([cell - 1, cell + 1], [cell - self.x, cell + self.x], [cell - self.x * self.y, cell + self.x * self.y])
        return np.hstack(parts[: self.dim])

    @cached_property
    def boundaries(self) -> Sequence[Sequence[int]]:
        """
        computes the boundaries of the box in each direction.
         - The boundaries on the left,right are given by an arithmetic progression
           whose jump is size of the x axis.
         - The boundaries on the back,forward are given by a union of arithmetic
           progression whose jumps are the number of cells in a single x-y plane.
           The number of such arithmetic progression is the number of cells in the
           x axis.
         - The boundaries on the down,up are given by a consecutive numbers.

        Returns
        -------
        Sequence[Sequence[int]]
         list of the boundaries according to the usual order:
         left, right, back, forward, down, up.
         Each boundary is a sequence of the number of the cells on the boundary.
        """
        cells = np.arange(self.cells)
        left = cells[:: self.x]
        right = cells[self.x - 1 :: self.x]
        back = np.add.outer(cells[:: self.x * self.y], cells[: self.x]).flatten()
        forward = np.add.outer(cells[:: self.x * self.y], cells[: self.x]).flatten() + self.x * (self.y - 1)
        down = cells[: self.x * self.y]
        up = cells[self.cells - self.x * self.y :]
        return [left, right, back, forward, down, up][: 2 * self.dim]

    def correct_neighbors_of_boundary_cell(self, neighbors: Sequence[int], cell: int) -> List[Neighbor]:
        """
        corrects the sequence of neighbors of a cell on the boundary

        Parameters
        ----------
        neighbors: Sequence[int]
         the numbers of the neighbors of the cell as if the cell is
         not on a boundary
        cell: int
         the number of the cell

        Returns
        -------
        Sequence[Neighbor]
         corrected sequence of neighbors, if one of the neighbors should be boundary
         then it is the corrected boundary conditions in the case that
         sizes of neighbors is unspecified, and it is a tuple of Boundary and of
         the number of the neighboring cell inside the box outside self.
        """
        if self.sizes_of_neighbors is not None:
            return [
                neigh if not self.on_face[face, cell] else (Boundary, self.neighbor_in_other_box(cell, face))
                for face, neigh in enumerate(neighbors)
            ]
        else:
            return [
                neigh if not self.on_face[face, cell] else self.boundary_conditions[face]
                for face, neigh in enumerate(neighbors)
            ]

    def neighbor_in_other_box(self, cell: int, face: int) -> Union[None, np.ndarray]:
        """
        Computes the number of the neighbors of a cell in another box.
        It is needed in the case that cell is on the boundary.
        The method first converts the number of a cell to x,y,z indices using
        np.unravel_index. Then the method adds to the vector of x,y,z indices the
        unit vector in the direction of the boundary resulting in a x,y,z vector
        representing the cell next to the boundary. Then a ravel_multi_index is
        called with the dimensions of the neighboring cell to convert the x,y,z vector
        back to a numerical index.

        Parameters
        ----------
        cell: int
         the number of the cell
        face: int
         the number of the face according to the usual order: lrbfdu

        Returns
        -------
        Union[None, np.ndarray]
         the number of the neighboring cell in the neighboring box in the case
         that sizes of neighbors is given. None in the case it is not.
        """
        if self.sizes_of_neighbors[face] is None:
            return None
        cor = np.array(np.unravel_index(cell, self.sizes, order="F"), dtype=int)
        return np.ravel_multi_index(
            cor + self.direction(cor, face), self.sizes_of_neighbors[face], mode="wrap", order="F"
        )

    def direction(self, cor: Sequence[int], face: int) -> np.array:
        """
        function used in the computation of the number of a cell in a neighbor
        box, it is the vector in which you need to move from the cell whose
        coordinates are cor to the coordinates of the neighboring cell.

        Parameters
        ----------
        cor: Sequence[int]
         the coordinates of a cell
        face: int
         the face of the cell that is on the boundary of the box
         accordint to the lrbfdu order

        Examples
        --------
        self.direction((2,4),3)
        array([ 0., -4.])
        """
        unit = np.zeros(self.dim, dtype=int)
        unit[face // 2] = -cor[face // 2] if face % 2 else -1
        return unit


def dim3_from_dim2(plane: PolyBox, z_lengths: np.array, boundary_conditions: Tuple[Boundary, Boundary]) -> PolyBox:
    """
    function to help generate a model of a 3 dimensional core using a model of
    a plane section and the data about the lengths of the levels in the z axis
    and the axial boundary conditions.

    Parameters
    ----------
    plane: PolyBox
     the model of the geometry of a plane section.
    z_lengths: np.array
     array of the lengths of the axial levels.
    boundary_conditions: Tuple[Boundary, Boundary]
     the axial boundary conditions

    Returns
    -------
    PolyBox
     3 dimensional model of the core
    """
    # noinspection PyTypeChecker
    z_neighbors = (
        [(boundary_conditions[0], 1)]
        + [(i, i + 2) for i in range(len(z_lengths) - 1)]
        + [(len(z_lengths) - 1, boundary_conditions[1])]
    )
    neighbors = [
        [n if isinstance(n, Boundary) else n + h * plane.cells for n in neigh]
        + [n if isinstance(n, Boundary) else n * plane.cells + cell for n in pair]
        for h, pair in enumerate(z_neighbors)
        for cell, neigh in enumerate(plane.neighbors)
    ]
    lengths = [list(plane_length) + [z_length] for z_length in z_lengths for plane_length in plane.lengths]
    return PolyBox(3, neighbors, lengths)
