"""
This module contains the CurrentCalculator protocol
"""

from typing import Protocol, Sequence, Tuple

import numpy as np

from dorban.geometry import FiniteGeometry


class CurrentCalculator(Protocol):
    """
    Such classes are responsible for calculating the current between 2 points
    in the system. They hold all the needed information like the diffusion
    coefficients and the discontinuity factors.
    """

    dc: np.array

    def face_coefficients(
        self,
        cells: np.array,
        others: np.array,
        faces: np.array,
        opposite: np.array,
        distances: np.array,
        other_distances: np.array,
        areas: np.array,
        geometry: FiniteGeometry,
    ) -> Tuple[np.array, np.array]:
        """
        This method computes the coefficients in the diffusion matrix of the currents through many faces between
        cells at once.

        Parameters
        ----------
        cells, others, faces, opposite: np.array
         for each face, the cell, the neighbor, the number of the face in the cell and in the neighbor
        distances, other_distances: np.array
         the distances from the centers of the cell and of the neighbor to the face
        areas: np.array
         the areas of the faces
        geometry: FiniteGeometry
         the geometry of the system

        Returns
        -------
        Tuple[np.array, np.array]
         arrays of shape faces x E: the current through the face is the first times the flux of the cell minus
         the second times the flux of the neighbor
        """
        raise NotImplementedError

    def mesh_refine(self, geo: FiniteGeometry, split: Sequence, refgeo: FiniteGeometry) -> "CurrentCalculator":
        """
        creates and returns a new CurrentCalculator which can calculate the
        current for the refined system.

        Parameters
        ----------
        geo: FiniteGeometry
            the geometry of the system
        split:
            split data whose structure is determined by the geometry of the system
        refgeo: FiniteGeometry
            the refined geometry
        """
        raise NotImplementedError

    def dc_on_face(self, cell: int, face: int, geometry: FiniteGeometry) -> np.array:
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
        raise NotImplementedError
