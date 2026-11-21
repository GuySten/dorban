from bisect import bisect_left
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
from numba import numba

from dorban.geometry.boundary_conditions import Boundary, Void
from dorban.geometry.geometry import FiniteGeometry


def tensor_product(a: np.array, b: np.array) -> np.array:
    """
    If a,b represents vector in the standard basis then the result is the
    representation in the standard  basis of their tensor product.

    Parameters
    ----------
    a: np.array
    b: np.array

    Returns
    -------
    np.array
        the tensor product of a,b
    Examples
    ---------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([7, 11])
    >>> tensor_product(a,b)
    array([ 7, 14, 21, 11, 22, 33])
    """
    return np.outer(b, a).flatten()


def triangular(n: int) -> int:
    """
    computes the n-th triangular number.

    Parameters
    ----------
    n: int

    Returns
    -------
    int
        the n-th triangular number
    """
    return n * (n - 1) // 2


def restore_array(array: np.array, black_absorber: np.array) -> np.array:
    """
    retunrs an array of length len(array)+len(black_absorber)
     who is created from the given array by placing 0 at the
     indices of the black absorber.

    Parameters
    ----------
    array: np.array
        the original array
    black_absorber: np.array
        the indices were 0 should be inserted in the new array

    Returns
    -------
    np.array
        array whose values are the same as the given array at indices which
        there is no black absorber. And 0 where there is black absorber.
    """
    length = len(array) + len(black_absorber)
    fluxiter = iter(array)
    resiter = (0 if i in black_absorber else next(fluxiter) for i in range(length))
    return np.fromiter(resiter, array.dtype, count=length)


@numba.njit(cache=True)
def vertex_in_neighbor(vertex: int, face: int) -> int:
    """
    function that gets an index of a vertex and a face adn returns the index of
    the same vertex in the other triangle that contains face.

    Parameters
    ----------
    vertex: int
    face: int

    Returns
    -------
    int
    """
    if vertex == 2:
        return face
    if face == 2:
        return vertex
    return 3 - vertex - face


def triangles_with_vertex(
    cell: int, vertex: int, neighbors: Sequence[Sequence[int]], found: Optional[set[int]] = None
) -> set:
    r"""
    Finds all the triangles that have a common vertex with the given
    triangle not contained in a given face. The correspondece between vertices
    and face is that each face corresponds to the vertex it doesn't contain.

    Parameters
    ----------
    cell: int
     The index of the given triangle
    vertex: int
     The index of the face. The faces are numbered according to:  / , \ ,_
    neighbors: Sequence[Sequence[int]]
     The sequence of neighbors for each cell of the system
    found: set[int]
     The triangles which are neighbors of the cell that were already
     found. It is used becasue the function calls itself recursivly

    Returns
    -------
    set[int]
        The set of the numbers of the triangles that contain the vertex

    Examples
    ----------
    >>> cell = 0
    >>> face = 2
    >>> boundary = Void()
    >>> neighbors = [[1,2,boundary],[boundary,0,3],[0,boundary,4],[boundary,5,1],[5,boundary,2],[4,3,boundary]]
    >>> triangles_with_vertex(cell, face, neighbors)
    {1, 2, 3, 4, 5}
    """

    ismain = found is None
    found = found or {cell}
    other = np.array([i for i in range(3) if i != vertex])
    dis1 = np.array(neighbors[cell])[other]
    non_boundary_condition = [not isinstance(tri, Boundary) for tri in dis1]
    dis1 = set((v if isinstance(v, int) else v.item() for v in dis1[non_boundary_condition]))
    other = other[non_boundary_condition]
    founded_before = found.copy()
    for triangle, face in zip(dis1, other):
        if triangle not in founded_before:
            found = found.union(
                triangles_with_vertex(triangle, vertex_in_neighbor(vertex, face), neighbors, found.union(dis1))
            )
    if ismain:
        found.remove(cell)
    return found


def intersect_partitions(partitions: Iterable[np.array]) -> np.array:
    """
    Get a sequence of partition of the same segment to disjoint segments and
    Returns the most coarse partition that refines both of them

    Examples
    --------
    >>> intersect_partitions([np.array([1,4,5]),np.array([2,2,6])])
    array([1, 1, 2, 1, 5])
    """
    result = np.diff(np.sort(np.hstack([np.cumsum(partition) for partition in partitions])), prepend=0)
    nonzero = [i for i, x in enumerate(result) if not np.isclose(x, 0)]
    return result[nonzero]


def mink_sum(a: Sequence, b: Sequence) -> np.array:
    """
    Computes the minkowski sum of two sequences. It returns a sequence whose
    elements are all the possible sums of an element from a and an element from b.

    Examples
    --------
    >>> mink_sum([1,2],[3,4])
    array([4, 5, 5, 6])
    """
    return np.array(np.add.outer(a, b).flatten(), dtype=int)


def insert_black_absorber(
    cell: Union[int, Boundary], black: list[int], condition: Boundary = None
) -> Union[int, Boundary]:
    """
    This function gets a cell and a list of the numbers of
    the cells that should be ignored.
    Usually cells has to be ignored because they contain black absorber.
    The function returns the cell in the numbering after ignoring the cells we needed
    to ignore.
    If cell is a Boundary condition then it remains unchanged.

    Parameters
    ----------
    cell: Union[int, Boundary]
     Either a number of a cell or a Boundary condition.
    black: list[int]:
     the list of numbers of cells to ignore
    condition: Boundary
     the boundary condition which the cells near the cells that we
     ignore should have.
    Returns
    --------
    Union[int, Boundary]
        if cell is a Boundary then it is returned unchanged, if it is a number of
        a cell then the number of the cell after ignoring the black cells is
        returned.
    """
    condition = condition or Void()
    black = sorted(black)
    if isinstance(cell, Boundary):
        return cell
    if cell in black:
        return condition
    return cell - bisect_left(black, cell)


@numba.njit(cache=True)
def sparse_columns(size: int, E: int):
    """
    Function to generate the column indices of the values of a sparse matrix
    built from dense matrices of shape ExE on the diagonal

    Examples
    ---------
    >>> size = 4
    >>> E = 2
    >>> sparse_columns(size, E)
    array([0, 1, 0, 1, 2, 3, 2, 3])
    """
    return np.array([(i // (E**2)) * E + i % E for i in range(size * E)])


def normalize(x: np.array):
    """
    Normalize an array
    """
    return x / np.linalg.norm(x, ord=1)


def space_energy_reshape(flux: np.array, cells: int, E: int):
    """
    Reshapes a 1 dim array ordered like the flux, by cells and then by energy to
    a 2 dim array of shape cellsxE. It is the opposite operation to flatten.

    Examples
    -------
    >>> array=np.array([1,2,3,4])
    >>> space_energy_reshape(array,2,2)
    array([[1, 2],
           [3, 4]])
    """
    return np.reshape(flux, (cells, E))


def collapse_flux(geometry: FiniteGeometry, fine_flux: np.array, fine_volumes: np.array, split: Any) -> np.array:
    """
    Function that gets a fine flux obtained by preforming a computation on a refined
    core and collapses the flux back to the mesh of the original core.

    Parameters
    ----------
    geometry: FiniteGeometry
     The geometry to which the flux will be collapsed
    fine_flux: np.array
     The flux obtained from the computation on the refined system
    fine_voluems: np.array
     The volumes of the refined geometry
    split: Any
     the split data used to pass to the refined geometry.

    Returns
    -------
    np.array
     The spatially collapsed flux
    """
    assemblies = geometry.array_refine(np.arange(geometry.cells), split)
    volumes = geometry.array_refine(geometry.volumes, split)
    E = len(fine_flux) // len(assemblies)
    normalized_fine_flux = fine_flux * np.repeat(fine_volumes / volumes, E)
    assemblies_with_energy = np.vstack([assemblies * E + e for e in range(E)]).flatten(order="F").astype(int)
    return np.bincount(assemblies_with_energy, normalized_fine_flux)


@numba.njit(cache=True)
def op_face(face: int) -> int:
    """
    Returns the number of the opposite face to the given face.

    Parameters
    ----------
    face: int

    Returns
    -------
    int
    """
    if face % 2:
        return face - 1
    return face + 1
