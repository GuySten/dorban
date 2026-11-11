"""
This module contains the tools to create and store cross-sections data in a finite
number of energy groups.

The main function offered by this module are the :func:`Fissionable <dorban.materials.Fissionable>` and
:func:`Isotope <dorban.materials.Isotope>` functions.

:func:`Fissionable <dorban.materials.Fissionable>` should be used to create cross section data
of fissionable materials.

:func:`Isotope <dorban.materials.Isotope>` should be used to create cross section data
of non-fissionable materials.
"""

from typing import Dict, Tuple

import numpy as np


class CrossSectionData:
    r"""
    class that holds the values of macroscopic cross sections. It is basically
    a numpy array with a name, the access to cross sections
    is via the names so they look like separate properties. The class implements
    addition and scalar multiplication which are used for cross section interpolation
    and for generation of macroscopic cross sections from microscopic ones.
    It also allows the use of special cross section like data, for example diffusion
    coefficients.

    This class should be created using the :func:`Fissionable <dorban.materials.Fissionable>` and
    :func:`Isotope <dorban.materials.Isotope>` functions.
    """

    def __init__(self, name: str, data: np.array, special_cross_sections: Dict[str, int]):
        """

        Parameters
        ----------
        name: str
            The name of the material whose cross sections are given
        data:np.array
            The data about the cross sections, 2 dim numpy array with E columns
        special_cross_sections: Dict[str,int]
            Dict which maps names of special cross sections to their indices
            in the data

        Attributes
        ----------
        E:int
            the number of energy groups
        """
        self.special_cross_sections = special_cross_sections
        self.data = data
        self.name = name
        self.E = self.data.shape[1]

    @property
    def scatter(self) -> np.array:
        r"""
        2 dim array of the scattering matrix.
        The i,j entry :math:`\Sigma_s^{i,j}` is the cross section for scattering
        from the :math:`j` group to the :math:`i` group.
        Notice that the values on the diagonal are meaningless in the diffusion approximation.
        """
        return self.data[: self.E, :]

    @scatter.setter
    def scatter(self, scatter):
        self.data[: self.E, :] = scatter

    @property
    def absorb(self) -> np.array:
        r"""
        The absorption cross section :math:`\Sigma_a`
        """
        return self.data[self.E]

    @absorb.setter
    def absorb(self, absorb):
        self.data[self.E] = absorb

    @property
    def total(self) -> np.array:
        r"""
        The cross section for any interaction :math:`\Sigma_{total}`
        """
        return self.data[self.E + 1]

    @total.setter
    def total(self, total):
        self.data[self.E + 1] = total

    @property
    def kappa(self) -> np.array:
        r"""
        Data about heat generation :math:`\kappa`
        """
        return self.data[self.E + 4]

    @kappa.setter
    def kappa(self, kappa):
        self.data[self.E + 4] = kappa

    @property
    def fission(self) -> np.array:
        r"""
        The fission cross section :math:`\Sigma_f`
        """
        return self.data[self.E + 5]

    @fission.setter
    def fission(self, fission):
        self.data[self.E + 5] = fission

    @property
    def nusigmaf(self) -> np.array:
        r"""
        The fission cross section multiplied by the number of neutrons
        generated in a fission event :math:`\Sigma_f\cdot \nu`
        """
        return self.data[self.E + 2]

    @nusigmaf.setter
    def nusigmaf(self, nusigmaf):
        self.data[self.E + 2] = nusigmaf

    @property
    def chi(self) -> np.array:
        r"""
        The fission spectrum :math:`\chi`
        """
        return self.data[self.E + 3]

    @chi.setter
    def chi(self, chi):
        self.data[self.E + 3] = chi

    @property
    def transport(self) -> np.array:
        r"""
        The transport cross section, defined as  :math:`\Sigma_{tr}=1/D`
        """
        return self.data[self.special_cross_sections["transport"]]

    @transport.setter
    def transport(self, transport):
        self.data[self.special_cross_sections["transport"]] = transport

    @property
    def diffusion(self):
        r"""
        The diffusion coefficients of the material
        """
        return self.data[self.special_cross_sections["diffusion"]]

    @diffusion.setter
    def diffusion(self, diffusion):
        self.data[self.special_cross_sections["diffusion"]] = diffusion

    @property
    def adf(self):
        """
        sometimes discontinuity factors will be available for each assembly,
        for example when they are computed during the initial homogenization
        process, in this case it might by useful to save them in the corss section
        data of the assembly.
        adf stands for assembly discontinuity factors
        """
        return self.data[self.special_cross_sections["adf"]]

    @adf.setter
    def adf(self, adf):
        self.data[self.special_cross_sections["adf"]] = adf

    @property
    def isfissile(self):
        """
        Checks whatever the fission cross section is zero.
        """
        return not np.all(self.nusigmaf == 0)

    def __add__(self, other):
        if other == 0:
            return self
        name = self.name if self.name == other.name else self.name + " " + other.name
        return CrossSectionData(name, self.data + other.data, self.special_cross_sections)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, density: float):
        if density == 0:
            return 0
        return CrossSectionData(self.name, density * self.data, self.special_cross_sections)

    def __rmul__(self, density: float):
        return self.__mul__(density)

    def __eq__(self, other):
        if other == 0:
            return False
        return self.name == other.name and np.all(self.data == other.data)

    def __str__(self):
        string = f"{self.name}\n absorption: {self.absorb} \n "
        if self.isfissile:
            string += f"nusigmaf: {self.nusigmaf} \n chi: {self.chi}\n"
        string += f"scatter \n {self.scatter}\n"
        return string

    def __hash__(self):
        return hash(str(self))

    def __getitem__(self, item):
        return self.data[self.special_cross_sections[item]]


def _isotope(
    scatter: np.array, absorb: np.array, *, kappa: np.array = None, **special_cross_sections
) -> Tuple[np.array, Dict]:
    if kappa is None:
        kappa = np.zeros_like(absorb)
    E = absorb.shape[0]
    data = np.zeros((E + 6 + len(special_cross_sections) * E, E))
    data[:E, :] = scatter
    data[E] = absorb
    data[E + 1] = absorb + np.sum(scatter, axis=0)
    data[E + 4] = kappa
    special = {}
    count = 6 + E
    for key, value in special_cross_sections.items():
        try:
            data[count] = value
            special[key] = count
            count += 1
        except ValueError:
            data[count : count + E, :] = value
            special[key] = np.arange(count, count + E)
            count += E
    data = data[:count]
    return data, special


def Isotope(name: str, scatter: np.array, absorb: np.array, **kwargs):
    r"""
    Function used to generate the CrossSectionData of a non fissionable material

    Parameters
    ----------
    name: str
        the name of the material
    scatter: np.array
        2 dim array of the scattering matrix.
        The i,j entry :math:`\Sigma_s^{i,j}` is the cross section for scattering
        from the :math:`j` group to the :math:`i` group.
        Notice that the values on the diagonal are meaningless in the diffusion approximation.
    absorb: np.array
        the absorption cross section
    kwargs: np.array
        additional cross sections may be provided using key word arguments.

    Returns
    -------
    CrossSectionData
    """
    data, special = _isotope(scatter, absorb, **kwargs)
    return CrossSectionData(name, data, special)


def Fissionable(
    name: str,
    scatter: np.array,
    absorb: np.array,
    nusigmaf: np.array,
    chi: np.array,
    *,
    fission: np.array = None,
    **kwargs,
) -> CrossSectionData:
    r"""
    Function used to generate the CrossSectionData of a Fissionable material

    Parameters
    ----------
    name: str
        the name of the material
    scatter: np.array
        2 dim array of the scattering matrix.
        The i,j entry :math:`\Sigma_s^{i,j}` is the cross section for scattering
        from the :math:`j` group to the :math:`i` group.
        Notice that the values on the diagonal are meaningless in the diffusion approximation.
    absorb: np.array
        the absorption cross section
    nusigmaf: np.array
        :math:`\nu \cdot \Sigma_f`
    chi: np.array
        :math:`\chi`
    fission: Optional[np.array]
        the fission cross section
    kwargs: np.array
        additional cross sections may be provided using key word arguments.

    Returns
    -------
    CrossSectionData
    """
    if fission is None:
        fission = np.zeros_like(absorb)
    data, special = _isotope(scatter, absorb, **kwargs)
    E = absorb.shape[0]
    data[E + 2] = nusigmaf
    data[E + 3] = chi
    data[E + 5] = fission
    if np.all(data[E + 4] == 0):  # if kappa is not specidifed set it to fission
        data[E + 4] = fission
    return CrossSectionData(name, data, special)
