import numpy as np
import scipy.linalg as la

from dorban import Cartesian, Core, CurrentCalculatorFD, Fissionable, Isotope, Reflector
from dorban.cross_sections_operators import _absorb as absorb
from dorban.cross_sections_operators import _nufission as nufission
from dorban.geometry.boundary_conditions import Boundary, Void
from dorban.nem.cmfd_current_calculator import CMFDCurrentCalculator
from dorban.nem.solve_nem import NEMSettings
from dorban.rdfs.nem_rdf import compute_boundary_fluxes_using_nem, compute_rdfs
from dorban.settings import FDSettings
from dorban.solve_equation import solve_k
from dorban.system import mesh_refinement


def one_dim_colorset_flux_boundary(
    assembly: Core, current: np.array, k: float
) -> np.array:
    """
    Function used to compute the flux on the boundary induced by a given current
    from one side and reflector on the other, used for computing rdf of a 1
    dimensional 2x1 color set problem.
    Parameters
    ----------
    current - the current on the boundary
    k- the k of the color set problem

    Returns
    -------
    the flux on the boundary
    """
    d = assembly.current_calc.dc_on_face(0, 0, assembly.geometry)
    if len(d.shape) == 1:
        d = np.diag(d)
    mat = assembly.isotopes[0]
    a = absorb(mat) - 1 / k * nufission(mat)
    alpha = la.sqrtm(np.linalg.inv(d) @ a)
    length = assembly.geometry.distance_to_face(0, 0) * 2
    v = la.solve(-d @ alpha @ la.sinm(length * alpha), current)
    return la.cosm(length * alpha) @ v


def test_compare_nem_flux_on_boundary_with_analytic_flux_on_boundary_in_2x1_problem():
    ref = Isotope(
        name="ref",
        scatter=np.array(
            [[1.86347243e00, 2.89718656e-02], [8.35668223e-04, 6.30553914e-01]]
        ),
        absorb=np.array([0.04193057, 0.00242824]),
        diffusion=np.array([0.27593463, 1.17322397]),
    )
    fuel = Fissionable(
        name="fuel",
        scatter=np.array([[1.21111292, 0.01692339], [0.00179553, 0.51985346]]),
        absorb=np.array([0.099985, 0.0100711]),
        nusigmaf=np.array([0.13072259, 0.00678641]),
        chi=np.array([0, 1]),
        diffusion=np.array([0.4039059, 1.33992157]),
    )
    current = np.array([0.01044089, 0.129204])
    d = 0.1
    geometry = Cartesian([[d, d]], boundary_condition=[Reflector()] * 2).to_polybox()
    color_set = Core(
        [fuel, ref], 2, geometry, CurrentCalculatorFD.from_isotopes([fuel, ref])
    )
    k = 0.91436
    zero = np.zeros((1, 2))
    flux_1 = one_dim_colorset_flux_boundary(color_set, current, k)
    flux_2 = compute_boundary_fluxes_using_nem(
        current, zero, zero, d, d, d, d, zero, zero, fuel, k
    )[1]
    assert np.all(np.isclose(flux_1, flux_2, atol=1e-2))


def test_nem_rdfs_2x1_problem_regression(file_regression):
    ref = Isotope(
        name="ref",
        scatter=np.array(
            [[1.86347243e00, 2.89718656e-02], [8.35668223e-04, 6.30553914e-01]]
        ),
        absorb=np.array([0.04193057, 0.00242824]),
        diffusion=np.array([0.27593463, 1.17322397]),
    )
    fuel = Fissionable(
        name="fuel",
        scatter=np.array([[1.21111292, 0.01692339], [0.00179553, 0.51985346]]),
        absorb=np.array([0.099985, 0.0100711]),
        nusigmaf=np.array([0.13072259, 0.00678641]),
        chi=np.array([0, 1]),
        diffusion=np.array([0.4039059, 1.33992157]),
    )
    current = np.array([0.01044089, 0.129204])
    d = 0.1
    geometry = Cartesian(
        [[d, d], [1, 1]], boundary_condition=[Reflector()] * 4
    ).to_polybox()
    color_set = Core(
        [fuel, ref, fuel, ref],
        2,
        geometry,
        CurrentCalculatorFD.from_isotopes([fuel, ref, fuel, ref]),
    )
    k = 0.91436
    currents = np.zeros((4, 4, 2))
    currents[0, 1] = current
    currents[1, 0] = current
    currents[2, 1] = current
    currents[3, 0] = current
    rdfs = compute_rdfs(color_set, currents, k)
    file_regression.check(str(rdfs))


def test_rdfs_reproduce_a_fine_reference():
    """Generalized equivalence: a heterogeneous 3x3-node problem solved with finite differences on a fine mesh,
    homogenized per node (flux-volume weighting, node-face currents of the fine solution), is reproduced by NEM on
    the 3x3 nodes with the discontinuity factors of compute_rdfs (k and node fluxes)."""

    E = 2
    fuel = Fissionable("fuel", scatter=np.array([[0.0, 0.0], [0.02, 0.0]]), absorb=np.array([0.010, 0.08]),
                       nusigmaf=np.array([0.007, 0.135]), chi=np.array([1.0, 0.0]), diffusion=np.array([1.4, 0.4]))
    water = Isotope("water", scatter=np.array([[0.0, 0.0], [0.04, 0.0]]), absorb=np.array([0.0005, 0.02]),
                    diffusion=np.array([1.2, 0.15]))
    n_node, n_sub, pitch, split = 3, 3, 1.26, 4
    h = pitch / n_sub
    nf = n_node * n_sub
    mats, node_of_cell = [], []
    for fy in range(nf):
        for fx in range(nf):
            ni, nj = fx // n_sub, fy // n_sub
            centre = (ni == 1 and nj == 1) or (fx % n_sub == 1 and fy % n_sub == 1)
            mats.append(water if centre else fuel)
            node_of_cell.append(nj * n_node + ni)
    fine = Core(mats, E, Cartesian([np.full(nf, h)] * 2, [Reflector()] * 4).to_polybox(), CurrentCalculatorFD.from_isotopes(mats))
    ref = mesh_refinement(fine, fine.geometry.uniform_split(split))
    k_ref, flux = solve_k(ref, FDSettings(split=ref.geometry.uniform_split(1)))
    cells = ref.geometry.cells
    phi = np.asarray(flux).reshape(cells, E)
    hr = h / split
    D = np.vstack([m["diffusion"] for m in ref.isotopes])
    node_of = np.asarray(node_of_cell)[np.arange(cells) // (split * split)]  # refined cells come in blocks
    N = n_node * n_node
    nodes, flux_volume = [], np.zeros((N, E))
    for n in range(N):
        idx = np.nonzero(node_of == n)[0]
        w = phi[idx]
        W = w.sum(axis=0)
        flux_volume[n] = W * hr * hr
        total = sum(ref.isotopes[c].total * w[q] for q, c in enumerate(idx)) / W
        scatter = sum(ref.isotopes[c].scatter * w[q][None, :] for q, c in enumerate(idx)) / W[None, :]
        nusigmaf = sum(ref.isotopes[c].nusigmaf * w[q] for q, c in enumerate(idx)) / W
        diffusion = sum(ref.isotopes[c]["diffusion"] * w[q] for q, c in enumerate(idx)) / W
        kw = dict(scatter=scatter, absorb=total - scatter.sum(axis=0), diffusion=diffusion)
        nodes.append(Fissionable(f"n{n}", nusigmaf=nusigmaf, chi=np.array([1.0, 0.0]), **kw) if nusigmaf.sum() > 0
                     else Isotope(f"n{n}", **kw))
    currents = np.zeros((N, 4, E))  # net current in +x / +y on the faces -x, +x, -y, +y, per unit area
    for c in range(cells):
        for face in (1, 3):
            nb = ref.geometry.neighbors[c][face]
            if isinstance(nb, Boundary) or node_of[nb] == node_of[c]:
                continue
            j = 2 * D[c] * D[nb] * (phi[c] - phi[nb]) / ((D[c] + D[nb]) * hr) * hr / pitch
            currents[node_of[c], face] += j
            currents[node_of[nb], face - 1] += j
    average = flux_volume / pitch**2
    dc = np.vstack([m["diffusion"] for m in nodes])
    geometry = Cartesian([np.full(n_node, pitch)] * 2, [Reflector()] * 4)
    rdfs = compute_rdfs(Core(nodes, E, geometry, CMFDCurrentCalculator(dc, dim=2)), currents, k_ref, average)
    k, nodal = solve_k(Core(nodes, E, Cartesian([np.full(n_node, pitch)] * 2, [Reflector()] * 4),
                            CMFDCurrentCalculator(dc, dim=2, df=rdfs)),
                       NEMSettings(split=geometry.uniform_split(1), k_tol=1e-10, flux_rtol=1e-10, flux_atol=1e-13))
    nodal = np.asarray(nodal).reshape(N, E)
    assert abs(k - k_ref) < 1e-8 * k_ref
    assert np.allclose(nodal / nodal.sum(), average / average.sum(), rtol=1e-7, atol=0)


def test_nem_quarter_core_with_reflective_cuts_equals_the_full_core():
    """A problem symmetric in x and y solved in full and as its quarter with reflective boundaries on the symmetry
    lines: the same k and node fluxes. Across a reflector the quadratic transverse leakage fit takes the mirror image of
    the cell as its neighbor (minus the cell's own leakage gave +560 pcm here)."""

    f1 = Fissionable("f1", scatter=np.array([[0, 0], [0.02, 0]]), absorb=np.array([0.010, 0.08]),
                     nusigmaf=np.array([0.007, 0.135]), chi=np.array([1.0, 0.0]), diffusion=np.array([1.4, 0.4]))
    f2 = Fissionable("f2", scatter=np.array([[0, 0], [0.018, 0]]), absorb=np.array([0.011, 0.10]),
                     nusigmaf=np.array([0.008, 0.16]), chi=np.array([1.0, 0.0]), diffusion=np.array([1.4, 0.4]))
    w = Isotope("w", scatter=np.array([[0, 0], [0.04, 0]]), absorb=np.array([0.0005, 0.02]), diffusion=np.array([1.2, 0.15]))
    rows = [[w, f1, f1, w], [f1, f2, f2, f1], [f1, f2, f2, f1], [w, f1, f1, w]]

    def solve(mats, n, bcs):
        geometry = Cartesian([np.full(n, 10.0)] * 2, bcs)
        dc = np.vstack([m["diffusion"] for m in mats])
        k, flux = solve_k(Core(mats, 2, geometry, CMFDCurrentCalculator(dc, dim=2)),
                          NEMSettings(split=geometry.uniform_split(1), k_tol=1e-11, flux_rtol=1e-11, flux_atol=1e-14))
        return k, np.asarray(flux).reshape(n * n, 2)

    k_full, flux_full = solve([m for row in rows for m in row], 4, [Void()] * 4)
    k_quarter, flux_quarter = solve([rows[2][2], rows[2][3], rows[3][2], rows[3][3]], 2,
                                    [Reflector(), Void(), Reflector(), Void()])
    flux_full = flux_full[[10, 11, 14, 15]]
    assert abs(k_quarter - k_full) < 1e-9
    assert np.allclose(flux_quarter / flux_quarter.sum(), flux_full / flux_full.sum(), rtol=1e-8, atol=0)
