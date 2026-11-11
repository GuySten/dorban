r"""
The NEM two-node and one-node problems that give the coupling coefficients of the CMFD equations, written to fill
preallocated arrays: the coupling update of a cell allocates one matrix and one right hand side, which are reused for
all its faces, so the cells can be computed in parallel without contention on memory allocation.

In each node the flux along the axis between the nodes is :math:`\bar\phi+\sum_{i=1}^4a_iP_i(\xi)` with the NEM
polynomials on :math:`\xi\in[-1/2,1/2]`. The unknowns of the two-node problem are, in blocks of E groups,
:math:`a_1^{(1)},a_2^{(1)},a_1^{(2)},a_2^{(2)},a_3^{(1)},a_4^{(1)},a_3^{(2)},a_4^{(2)}`; its equations are the
balance, first and second moments of each node (blocks 0, 1, 4 for the first, 2, 3, 5 for the second), the
continuity of the current (block 6) and the continuity of the heterogeneous flux (block 7). The one-node problem has
the unknowns :math:`a_1,a_2,a_3,a_4`, the three equations of the node and the boundary condition.
"""

import numba
import numpy as np

from dorban.nem.leakage import leakage_coefficients


@numba.njit(cache=True)
def _fill_node(mat, rows, cols, E, ll, D, absorber, fission):
    """
    The balance and moment equations of a node of width ll: rows are the blocks of its three equations and cols the
    blocks of its unknowns :math:`a_1,a_2,a_3,a_4`.
    """
    for g in range(E):
        d = D[g] / ll / ll
        r0, r1, r2 = rows[0] * E + g, rows[1] * E + g, rows[2] * E + g
        mat[r0, cols[0] * E + g] = -6 * d
        mat[r0, cols[1] * E + g] = -2 / 5 * d
        mat[r1, cols[1] * E + g] = 140 * d
        mat[r2, cols[3] * E + g] = 60 * d
        for h in range(E):
            removal = absorber[g, h] - fission[g, h]
            mat[r1, cols[0] * E + h] += -35 * removal
            mat[r1, cols[1] * E + h] += removal
            mat[r2, cols[2] * E + h] += -10 * removal
            mat[r2, cols[3] * E + h] += removal


@numba.njit(cache=True)
def _neighbor_leakage(leak, cell, neighbor, transverse, g):
    """The transverse leakage of the neighbor, or minus the cell's own across a boundary."""
    return leak[neighbor, transverse, g] if neighbor >= 0 else -leak[cell, transverse, g]


@numba.njit(cache=True)
def _neighbor_length(distances, cell, neighbor, axis):
    """The width of the neighbor along the axis, or the cell's own across a boundary."""
    return 2 * distances[neighbor if neighbor >= 0 else cell, 2 * axis]


@numba.njit(cache=True)
def _add_node_rhs(rhs, rows, E, dim, axis, flux, absorber, fission, source, leak, distances, cell, left, right):
    """
    The right hand side of the three equations of a node (rows: their blocks), with its neighbors left and right
    along the axis for the quadratic transverse leakage.
    """
    ll = 2 * distances[cell, 2 * axis]
    l_left = _neighbor_length(distances, cell, left, axis)
    l_right = _neighbor_length(distances, cell, right, axis)
    for g in range(E):
        value = source[g]
        for h in range(E):
            value -= (absorber[g, h] - fission[g, h]) * flux[h]
        rhs[rows[0] * E + g] = value
        rhs[rows[1] * E + g] = 0.0
        rhs[rows[2] * E + g] = 0.0
    for transverse in range(dim):
        if transverse == axis:
            continue
        length = 2 * distances[cell, 2 * transverse]
        for g in range(E):
            leakage = leak[cell, transverse, g]
            p1, p2 = leakage_coefficients(
                leakage,
                _neighbor_leakage(leak, cell, left, transverse, g),
                _neighbor_leakage(leak, cell, right, transverse, g),
                ll,
                l_left,
                l_right,
            )
            rhs[rows[0] * E + g] -= leakage / length
            rhs[rows[1] * E + g] += 35 * p2 / length
            rhs[rows[2] * E + g] += 10 * p1 / length


@numba.njit(cache=True)
def solve_in_place(mat, rhs, n):
    """Solves mat[:n, :n] x = rhs[:n] by Gaussian elimination with partial pivoting; x replaces rhs."""
    for column in range(n):
        pivot = column
        for row in range(column + 1, n):
            if abs(mat[row, column]) > abs(mat[pivot, column]):
                pivot = row
        if pivot != column:
            for j in range(column, n):
                mat[column, j], mat[pivot, j] = mat[pivot, j], mat[column, j]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        for row in range(column + 1, n):
            factor = mat[row, column] / mat[column, column]
            if factor != 0.0:
                for j in range(column, n):
                    mat[row, j] -= factor * mat[column, j]
                rhs[row] -= factor * rhs[column]
    for row in range(n - 1, -1, -1):
        value = rhs[row]
        for j in range(row + 1, n):
            value -= mat[row, j] * rhs[j]
        rhs[row] = value / mat[row, row]


@numba.njit(cache=True)
def two_node_coupling(
    out, mat, rhs, E, dim, cell, neighbor, face, left, righter, v, dc, df, absorb, fission, source, leak, distances
):
    r"""
    The coupling coefficient between a cell and its neighbor across the face (face // 2 is the axis, the neighbor is
    on the positive side), written to out: :math:`\hat{D}` such that the CMFD current equals the NEM current,

    .. math:: \hat{D}=\left(J_{NEM}-\frac{2D_1D_2(F_1\phi_1-F_2\phi_2)}{F_2D_1l_2+F_1D_2l_1}\right)(\phi_1+\phi_2)^{-1}

    left is the neighbor of the cell on its other side and righter the neighbor of the neighbor beyond it, for the
    transverse leakage. mat (8E x 8E) and rhs (8E) are work arrays.
    """
    axis = face // 2
    opposite = face - 1
    l1, l2 = 2 * distances[cell, face], 2 * distances[neighbor, face]
    flux1, flux2 = v[cell * E : cell * E + E], v[neighbor * E : neighbor * E + E]
    mat[:, :] = 0.0
    _fill_node(mat, (0, 1, 4), (0, 1, 4, 5), E, l1, dc[cell], absorb[cell], fission[cell])
    _fill_node(mat, (2, 3, 5), (2, 3, 6, 7), E, l2, dc[neighbor], absorb[neighbor], fission[neighbor])
    for g in range(E):
        d1, d2 = dc[cell, g] / l1, dc[neighbor, g] / l2
        row = 6 * E + g
        for block, value in enumerate((-3 * d1, -d1 / 5, -3 * d2, -d2 / 5, -d1, -d1 / 2, d2, d2 / 2)):
            mat[row, block * E + g] = value
        dis1, dis2 = df[cell, face, g], df[neighbor, opposite, g]
        row = 7 * E + g
        mat[row, g] = dis1 / 2
        mat[row, 2 * E + g] = -dis2 / 2
        mat[row, 4 * E + g] = dis1 / 2
        mat[row, 6 * E + g] = dis2 / 2
        rhs[6 * E + g] = 0.0
        rhs[row] = dis2 * flux2[g] - dis1 * flux1[g]
    _add_node_rhs(
        rhs, (0, 1, 4), E, dim, axis, flux1, absorb[cell], fission[cell], source[cell * E : cell * E + E], leak,
        distances, cell, left, neighbor,
    )  # fmt: skip
    _add_node_rhs(
        rhs, (2, 3, 5), E, dim, axis, flux2, absorb[neighbor], fission[neighbor],
        source[neighbor * E : neighbor * E + E], leak, distances, neighbor, cell, righter,
    )  # fmt: skip
    solve_in_place(mat, rhs, 8 * E)
    ll, ln = 2 * distances[cell, 2 * axis], 2 * distances[neighbor, 2 * axis]
    for g in range(E):
        # the current through the positive face of the cell
        current = -dc[cell, g] / ll * (3 * rhs[g] + rhs[E + g] / 5 + rhs[4 * E + g] + rhs[5 * E + g] / 2)
        dis1, dis2 = df[cell, 2 * axis + 1, g], df[neighbor, 2 * axis, g]
        D1, D2 = dc[cell, g], dc[neighbor, g]
        finite_difference = 2 * D1 * D2 / (D2 * ll * dis1 + D1 * ln * dis2) * (flux2[g] * dis2 - flux1[g] * dis1)
        out[g] = (finite_difference + current) / (flux2[g] + flux1[g])


@numba.njit(cache=True)
def boundary_coupling(out, mat, rhs, E, dim, cell, face, left, albedo, v, dc, absorb, fission, source, leak, distances):
    r"""
    The coupling coefficient between a cell and a boundary of the form :math:`J=\alpha\phi_s` (Void,
    ExtrapolationLength) or :math:`\phi_s=0` (ZeroFlux, :math:`\alpha=\infty`), written to out. The one-node problem is
    written as if the boundary were on the positive side of the cell; left is the neighbor on the other side. In the
    NEM coordinates the flux at the boundary is :math:`\bar{\phi}+a_1/2+a_2/2` and the current
    :math:`-D/l(a_1+3a_2+a_3/2+a_4/5)`. The coupling coefficient is measured from the current of the CMFD equations
    without coupling: :math:`\alpha\bar\phi` for a finite albedo and the finite differences current
    :math:`D\bar\phi/(l/2)` for zero flux. mat (at least 4E x 4E) and rhs (at least 4E) are work arrays.
    """
    axis = face // 2
    ll = 2 * distances[cell, face]
    flux = v[cell * E : cell * E + E]
    n = 4 * E
    mat[:n, :n] = 0.0
    _fill_node(mat, (0, 1, 2), (0, 1, 2, 3), E, ll, dc[cell], absorb[cell], fission[cell])
    zero_flux = np.isinf(albedo)
    for g in range(E):
        row = 3 * E + g
        if zero_flux:
            mat[row, g] = 0.5
            mat[row, 2 * E + g] = 0.5
            rhs[row] = -flux[g]
        else:
            d = dc[cell, g] / ll
            mat[row, g] = 3 * d + albedo / 2
            mat[row, E + g] = d / 5
            mat[row, 2 * E + g] = d + albedo / 2
            mat[row, 3 * E + g] = d / 2
            rhs[row] = -albedo * flux[g]
    _add_node_rhs(
        rhs, (0, 1, 2), E, dim, axis, flux, absorb[cell], fission[cell], source[cell * E : cell * E + E], leak,
        distances, cell, left, -1,
    )  # fmt: skip
    solve_in_place(mat, rhs, n)
    lc = 2 * distances[cell, 2 * axis]
    for g in range(E):
        current = -dc[cell, g] / lc * (3 * rhs[g] + rhs[E + g] / 5 + rhs[2 * E + g] + rhs[3 * E + g] / 2)
        cmfd = dc[cell, g] / distances[cell, face] if zero_flux else albedo
        out[g] = current / flux[g] - cmfd
