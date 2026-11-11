"""This module contains the finite geometry protocol as well as the Point geometry"""

from typing import Protocol, Sequence, Union

import numpy as np

# codes in the array of face_neighbors
BOUNDARY = -1  # the face is on a boundary
NO_FACE = -2  # the cell has fewer faces than others


class FiniteGeometry(Protocol):
    """
    metaclass that represents geometry excluding the infinite geometry.

    The geometry is divided into cells, each cell has a number and a list
    of the numbers of its neighbors,if a cell is near the boundary then some of
    its neighbors are :class:`Boundary <dorban.geometry.boundary_conditions.Boundary>`
    objects.
    A Geometry class must implement the volumes,surface are and distance to face
    methods which are used to compute currents between two cells in the finite -
    differences kernel.

    Each Geometry object also must implement two refinements methods, mesh_refine and
    array_refine. The first is used to create a new geometry object of the same type
    with a refined mesh. The second is used in order to transform data known about
    the original geometry to data suitable to the format of the new geometry.
    The refinment methods are important in order to achive convergence is the spatial
    desacralization.

    Parameters
    ----------
    dimension: int
     the dimension of the geometry
    lengths: Sequence[np.array]
     The lengths of each cell in each dimension.

    Attributes
    ----------
    dim: int
     the dimension of the geometry
    neighbors: Sequence[Sequene[Union[int,Boundary]]]
     Sequence of sequences, the sequences at index i is a sequence of all neighbors
     of the cell numbered i.
    cells: int
     The number of cells of the geometry.
    lengths: Sequence[np.array]
     The lengths of each cell in each dimension.
    volumes: np.array
     array of the volumes of the cells.
    """

    dim: int
    neighbors: Sequence[Sequence[int]]
    cells: int
    lengths: Union[np.array, Sequence[np.array]]
    volumes: np.array

    def surface_area(self, cell: int, face: int) -> float:
        """
        function to compute the surface area of a face of a cell

        Parameters
        ----------
        cell:int
        face:int

        Returns
        -------
        float
         the surface area of a face of a cell
        """
        raise NotImplementedError

    def distance_to_face(self, cell: int, face: int) -> float:
        """
        function to compute the distance from the center of the cell to a given
        face

        Parameters
        ----------
        cell: int
         The index of the specific cell
        face: int
         The index of its specific face

        Returns
        -------
        float
         the distance from the center of the cell to the face
        """
        raise NotImplementedError

    def face_neighbors(self) -> np.array:
        """
        The neighbor across each face of each cell, as an array of shape cells x faces: the number of the neighboring
        cell, BOUNDARY if the face is on a boundary, or NO_FACE where a cell has fewer faces than the others.
        """
        faces = max(len(row) for row in self.neighbors)
        table = np.full((self.cells, faces), NO_FACE)
        for cell, row in enumerate(self.neighbors):
            for face, neighbor in enumerate(row):
                table[cell, face] = BOUNDARY if not isinstance(neighbor, (int, np.integer)) else neighbor
        return table

    def face_distances(self) -> np.array:
        """
        The distance from the center of each cell to each of its faces, as an array of shape cells x faces
        (0 where there is no face). Geometries may override it with a faster computation.
        """
        return self._face_values(self.distance_to_face)

    def face_areas(self) -> np.array:
        """
        The area of each face of each cell, as an array of shape cells x faces (0 where there is no face).
        Geometries may override it with a faster computation.
        """
        return self._face_values(self.surface_area)

    def _face_values(self, value) -> np.array:
        faces = max(len(row) for row in self.neighbors)
        table = np.zeros((self.cells, faces))
        for cell, row in enumerate(self.neighbors):
            for face in range(len(row)):
                table[cell, face] = value(cell, face)
        return table

    def opposite_face(self, face: int) -> int:
        """
        method to compute the number of the face of the cell who is adjacent to
        a given cell along a face

        Parameters
        ----------
        face:int
         the number of the face in a given cell along which
         the two cells are adjacent

        Returns
        -------
        int
         the number of the face in the other cell
        """
        raise NotImplementedError

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
         Unit vector that represents the direction to the face
        """
        raise NotImplementedError

    def array_refine(self, array: np.array, split, **kwargs) -> np.array:
        """
        refines an array according to the split parameters. For example this
        function is used in order to refine an array of densities of some
        isotope.

        Parameters
        ----------
        array:np.array
         An array that has to be refined
        split:
         splitting data. The type may vary for the different members of this protocol.

        Returns
        -------
        np.array
         new array containing the same information but is suitable to the
         refined geometry
        """
        raise NotImplementedError

    def refine_mesh(self, split) -> "FiniteGeometry":
        """
        This function returns a new Geometry with refined mesh according to the
        split data. The split data is unique to each geometry type, and
        it is specified in the geometries refine array property

        Parameters
        ----------
        split:
         the splitting data.  The type may vary for the different members of this protocol.

        Returns
        -------
        FiniteGeometry
         new Geometry with a refined mesh
        """
        raise NotImplementedError

    # required for rdf generation
    def single_cell(self, cell, boundary_condtions):
        """
        method for generating a geometry that contains a single cell.
        """
        raise NotImplementedError

    # required for rdf generation
    def uniform_split(self, subcells: int):
        """
        generates the split data for splitting each axis into the same number
        of cells

        Parameters
        ----------
        subcells:int
         the number of cells to which each axis has to be split
        """
        raise NotImplementedError

    # required for rdf generation
    def simple_boundaries(self, dim, size) -> Sequence[Sequence[int]]:
        """
        returns the boundary of the simplest geometry of this class of the given
        size and dimension. for example form PolyBox geometry this gemetry will
        be a 3D box with sides of length size. This used for calculation of
        surface flux in rdf generation
        """
        raise NotImplementedError


class Point:
    """
    class that represents 0 dimensional geometry
    """

    def __init__(self):
        self.cells = 1
        self.supp_black = False
        super().__init__()
