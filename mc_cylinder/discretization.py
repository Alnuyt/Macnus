"""Regular radial bases and spectral discretization operators.

The scalar potentials are expanded as

``q(s) = s**ell * sum_j q_j T_j(2*s**2 - 1)``, ``ell = abs(m)``.

This builds regularity at the cylinder axis into the approximation space and
leaves the endpoint ``s=1`` available for boundary bordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.special import kv, kvp


@dataclass(frozen=True)
class RadialOperatorMatrices:
    """Basis evaluations needed by the compact radial MHD equations."""

    nodes: np.ndarray
    derivatives: tuple[np.ndarray, ...]
    regularized_values: np.ndarray
    regularized_shifted: np.ndarray
    regularized_shifted_squared: np.ndarray
    shifted: np.ndarray
    shifted_squared: np.ndarray
    laplacian: np.ndarray
    shifted_laplacian: np.ndarray
    shifted_laplacian_squared: np.ndarray


@dataclass(frozen=True)
class BulkMHDMatrices:
    """Unbordered collocation pencil ``A x = omega B x``."""

    nodes: np.ndarray
    A: np.ndarray
    B: np.ndarray
    radial: RadialOperatorMatrices


@dataclass(frozen=True)
class BorderedMHDMatrices:
    """MHD pencil after insertion of the five lateral boundary conditions."""

    nodes: np.ndarray
    A: np.ndarray
    B: np.ndarray
    radial: RadialOperatorMatrices
    boundary_rows: tuple[int, ...]
    gauge_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class HydrodynamicMatrices:
    """Inviscid hydrodynamic reference problem in ``(T,P)`` order."""

    nodes: np.ndarray
    A: np.ndarray
    B: np.ndarray
    radial: RadialOperatorMatrices
    boundary_row: int
    gauge_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class ViscousHydrodynamicMatrices:
    """Viscous hydrodynamic pencil with three lateral no-slip rows."""

    nodes: np.ndarray
    A: np.ndarray
    B: np.ndarray
    radial: RadialOperatorMatrices
    boundary_rows: tuple[int, int, int]
    gauge_rows: tuple[int, ...] = ()


def radial_lobatto_nodes(size: int) -> np.ndarray:
    """Return mapped Chebyshev--Lobatto nodes on ``0 <= s <= 1``.

    The Chebyshev coordinate is ``x=2*s**2-1``.  Nodes are returned from the
    wall to the axis, consistently with the usual cosine ordering.
    """

    if size < 2:
        raise ValueError("size must be at least 2")
    j = np.arange(size, dtype=float)
    x = np.cos(np.pi * j / (size - 1))
    return np.sqrt((1.0 + x) / 2.0)


def regular_radial_basis(
    s: np.ndarray, ell: int, modes: int, max_derivative: int = 4
) -> tuple[np.ndarray, ...]:
    """Evaluate a regular cylindrical basis and its radial derivatives.

    Entry ``result[d][i, j]`` is the ``d``-th derivative at ``s[i]`` of
    ``s**ell*T_j(2*s**2-1)``.  The calculation remains in the Chebyshev
    representation, avoiding the catastrophic cancellation caused by
    converting high-degree basis functions to monomials.
    """

    if not isinstance(ell, (int, np.integer)) or ell < 0:
        raise ValueError("ell must be a non-negative integer")
    if modes < 1:
        raise ValueError("modes must be positive")
    if max_derivative < 0 or max_derivative > 4:
        raise ValueError("max_derivative must lie between zero and four")

    points = np.asarray(s, dtype=float)
    if points.ndim != 1 or np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("s must be a one-dimensional array in [0, 1]")

    matrices = [
        np.empty((points.size, modes), dtype=float)
        for _ in range(max_derivative + 1)
    ]
    x = 2.0 * points**2 - 1.0

    def falling_factorial(power: int, order: int) -> float:
        if order > power:
            return 0.0
        value = 1.0
        for factor in range(order):
            value *= power - factor
        return value

    for column in range(modes):
        basis = Chebyshev.basis(column)
        fx = [basis.deriv(order)(x) for order in range(max_derivative + 1)]

        # Successive s derivatives of f(x(s)), x=2*s**2-1.
        chain = [fx[0]]
        if max_derivative >= 1:
            chain.append(4.0 * points * fx[1])
        if max_derivative >= 2:
            chain.append(4.0 * fx[1] + 16.0 * points**2 * fx[2])
        if max_derivative >= 3:
            chain.append(48.0 * points * fx[2] + 64.0 * points**3 * fx[3])
        if max_derivative >= 4:
            chain.append(
                48.0 * fx[2]
                + 384.0 * points**2 * fx[3]
                + 256.0 * points**4 * fx[4]
            )

        # Leibniz rule for s**ell*f(x(s)).  Terms that would contain a
        # negative power of s have zero coefficient and are omitted, giving
        # the analytic axis limits without numerical division by zero.
        for derivative, matrix in enumerate(matrices):
            values = np.zeros_like(points)
            for chain_order in range(derivative + 1):
                power_order = derivative - chain_order
                coefficient = falling_factorial(ell, power_order)
                if coefficient == 0.0:
                    continue
                values += (
                    comb(derivative, chain_order)
                    * coefficient
                    * points ** (ell - power_order)
                    * chain[chain_order]
                )
            matrix[:, column] = values

    return tuple(matrices)


def axisymmetric_gauge_fixed_basis(
    s: np.ndarray, modes: int, max_derivative: int = 4
) -> tuple[np.ndarray, ...]:
    r"""Evaluate an even ``m=0`` basis with the constant gauge removed.

    The columns are

    ``T_j(2*s**2-1) - T_j(-1)``, ``j=1,...,modes``.

    They are even in ``s``, hence their first derivative vanishes at the
    axis, and they vanish at ``s=0``.  The latter condition fixes the additive
    constant in each axial toroidal--poloidal potential without constraining
    any physical field, since the fields depend only on radial derivatives or
    on ``L_0 q``.
    """

    points = np.asarray(s, dtype=float)
    if points.ndim != 1 or np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("s must be a one-dimensional array in [0, 1]")
    if modes < 1:
        raise ValueError("modes must be positive")
    if max_derivative < 0:
        raise ValueError("max_derivative must be non-negative")

    full = regular_radial_basis(
        points, ell=0, modes=modes + 1, max_derivative=max_derivative
    )
    matrices = [matrix[:, 1:].copy() for matrix in full]
    matrices[0] -= np.asarray(
        [(-1.0) ** degree for degree in range(1, modes + 1)]
    )[None, :]
    return tuple(matrices)


def _stable_regularized_operators(
    points: np.ndarray, ell: int, modes: int, k: float,
    *, axisymmetric_gauge_fixed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate I, D and D squared without conversion to monomials."""

    x = 2.0 * np.asarray(points) ** 2 - 1.0
    matrices = tuple(np.empty((x.size, modes)) for _ in range(3))
    x_plus_one = Chebyshev((1.0, 1.0))

    def laplacian_hat(polynomial: Chebyshev) -> Chebyshev:
        return (
            8.0 * (ell + 1) * polynomial.deriv(1)
            + 8.0 * x_plus_one * polynomial.deriv(2)
        )

    for column in range(modes):
        if axisymmetric_gauge_fixed:
            degree = column + 1
            basis = Chebyshev.basis(degree) - Chebyshev(((-1.0) ** degree,))
        else:
            basis = Chebyshev.basis(column)
        shifted = laplacian_hat(basis) - k**2 * basis
        shifted_squared = laplacian_hat(shifted) - k**2 * shifted
        for matrix, polynomial in zip(matrices, (basis, shifted, shifted_squared)):
            matrix[:, column] = polynomial(x)
    return matrices


def _stable_axisymmetric_uncancelled_operators(
    points: np.ndarray, modes: int, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``L_0``, ``D L_0`` and ``D^2 L_0`` in Chebyshev form."""

    x = 2.0 * np.asarray(points) ** 2 - 1.0
    matrices = tuple(np.empty((x.size, modes)) for _ in range(3))
    x_plus_one = Chebyshev((1.0, 1.0))

    def laplacian(polynomial: Chebyshev) -> Chebyshev:
        return 8.0 * polynomial.deriv(1) + 8.0 * x_plus_one * polynomial.deriv(2)

    def shifted(polynomial: Chebyshev) -> Chebyshev:
        return laplacian(polynomial) - k**2 * polynomial

    for column in range(modes):
        lm = laplacian(Chebyshev.basis(column))
        dlm = shifted(lm)
        d2lm = shifted(dlm)
        for matrix, polynomial in zip(matrices, (lm, dlm, d2lm)):
            matrix[:, column] = polynomial(x)
    return matrices


def _stable_wall_functionals(
    ell: int, modes: int, *, axisymmetric_gauge_fixed: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return q, q', q'', q''' and L_m q at s=1 in modal form."""

    q, q1, q2, q3, lm = (np.empty(modes, dtype=complex) for _ in range(5))
    for column in range(modes):
        degree = column + 1 if axisymmetric_gauge_fixed else column
        basis = Chebyshev.basis(degree)
        f0 = basis(1.0)
        if axisymmetric_gauge_fixed:
            f0 -= basis(-1.0)
        f1 = basis.deriv(1)(1.0)
        f2 = basis.deriv(2)(1.0)
        f3 = basis.deriv(3)(1.0)
        q[column] = f0
        q1[column] = ell * f0 + 4.0 * f1
        q2[column] = ell * (ell - 1) * f0 + 4.0 * (2 * ell + 1) * f1 + 16.0 * f2
        q3[column] = (
            ell * (ell - 1) * (ell - 2) * f0
            + 12.0 * ell**2 * f1
            + 48.0 * (ell + 1) * f2
            + 64.0 * f3
        )
        lm[column] = 8.0 * (ell + 1) * f1 + 16.0 * f2
    return q, q1, q2, q3, lm


def radial_operator_matrices(
    s: np.ndarray, m: int, modes: int, k: float
) -> RadialOperatorMatrices:
    """Evaluate ``L_m``, ``D L_m`` and ``D^2 L_m`` on the regular basis.

    Here ``D = L_m-k**2``.  Operators are applied to polynomial coefficients
    before evaluation, so the row at ``s=0`` contains analytic limits and no
    divisions by zero.
    """

    if not isinstance(m, (int, np.integer)):
        raise ValueError("m must be an integer")
    points = np.asarray(s, dtype=float)
    derivatives = regular_radial_basis(
        points, abs(m), modes, max_derivative=4
    )
    stable_I, stable_D, stable_D2 = _stable_regularized_operators(
        points, abs(m), modes, k,
    )
    ell = abs(m)
    radial_power = points[:, None] ** ell
    x = 2.0 * points**2 - 1.0
    laplacian_hat = np.empty((points.size, modes))
    shifted_laplacian_hat = np.empty_like(laplacian_hat)
    shifted_laplacian_squared_hat = np.empty_like(laplacian_hat)
    x_plus_one = Chebyshev((1.0, 1.0))

    def apply_laplacian_hat(polynomial: Chebyshev) -> Chebyshev:
        return (
            8.0 * (ell + 1) * polynomial.deriv(1)
            + 8.0 * x_plus_one * polynomial.deriv(2)
        )

    def apply_shifted_hat(polynomial: Chebyshev) -> Chebyshev:
        return apply_laplacian_hat(polynomial) - k**2 * polynomial

    for column in range(modes):
        basis = Chebyshev.basis(column)
        lm = apply_laplacian_hat(basis)
        dlm = apply_shifted_hat(lm)
        d2lm = apply_shifted_hat(dlm)
        laplacian_hat[:, column] = lm(x)
        shifted_laplacian_hat[:, column] = dlm(x)
        shifted_laplacian_squared_hat[:, column] = d2lm(x)

    return RadialOperatorMatrices(
        nodes=points,
        derivatives=derivatives,
        regularized_values=stable_I,
        regularized_shifted=stable_D,
        regularized_shifted_squared=stable_D2,
        shifted=radial_power * stable_D,
        shifted_squared=radial_power * stable_D2,
        laplacian=radial_power * laplacian_hat,
        shifted_laplacian=radial_power * shifted_laplacian_hat,
        shifted_laplacian_squared=radial_power * shifted_laplacian_squared_hat,
    )


def assemble_mhd_bulk_matrices(
    *, size: int, m: int, k: float, Le: float, E: float, Em: float,
    coriolis: float = 2.0,
) -> BulkMHDMatrices:
    """Assemble the unbordered dense MHD pencil in ``(T,P,G,F)`` order.

    Boundary conditions have deliberately not yet replaced any rows.  This
    function isolates and tests the translation of the compact differential
    equations into matrix blocks.
    """

    nodes = radial_lobatto_nodes(size)
    radial = radial_operator_matrices(nodes, m=m, modes=size, k=k)
    I = radial.regularized_values.astype(complex)
    D = radial.regularized_shifted.astype(complex)
    D2 = radial.regularized_shifted_squared.astype(complex)
    if m == 0:
        # The usual compact equations were obtained by cancelling L_m.  This
        # is not legitimate for m=0 because constants belong to ker(L_0).
        # Retain the uncancelled L_0 factor; four gauge rows are added by the
        # bordered assemblers below.
        I, D, D2 = (
            matrix.astype(complex)
            for matrix in _stable_axisymmetric_uncancelled_operators(
                nodes, size, k
            )
        )
    A = np.zeros((4 * size, 4 * size), dtype=complex)
    B = np.zeros_like(A)
    T, P, G, F = (slice(j * size, (j + 1) * size) for j in range(4))
    r1, r2, r3, r4 = (slice(j * size, (j + 1) * size) for j in range(4))

    A[r1, T] = -E * D
    A[r1, P] = -coriolis * k * I
    A[r1, G] = -(Le**2) * k * I
    B[r1, T] = 1j * I

    A[r2, T] = coriolis * k * I
    A[r2, P] = E * D2
    A[r2, F] = -(Le**2) * k * D
    B[r2, P] = -1j * D

    A[r3, P] = -k * I
    A[r3, F] = -Em * D
    B[r3, F] = 1j * I

    A[r4, T] = k * I
    A[r4, G] = -Em * D
    B[r4, G] = 1j * I

    return BulkMHDMatrices(nodes=nodes, A=A, B=B, radial=radial)


def _border_axisymmetric_gauges(
    A: np.ndarray,
    B: np.ndarray,
    radial: RadialOperatorMatrices,
    size: int,
    blocks: int,
) -> tuple[int, ...]:
    """Replace axis equations by ``q(0)=0`` for each ``m=0`` potential."""

    axis_value = radial.derivatives[0][-1].astype(complex)
    rows = tuple((block + 1) * size - 1 for block in range(blocks))
    for block, row in enumerate(rows):
        A[row, :] = 0.0
        B[row, :] = 0.0
        columns = slice(block * size, (block + 1) * size)
        A[row, columns] = axis_value
    return rows


def assemble_mhd_matrices(
    *,
    size: int,
    m: int,
    k: float,
    Le: float,
    E: float,
    Em: float,
    mechanical_boundary: str = "stress_free",
    coriolis: float = 2.0,
) -> BorderedMHDMatrices:
    """Assemble the reduced MHD problem with mechanical/vacuum wall rows.

    ``mechanical_boundary`` may be ``"inviscid"``, ``"stress_free"`` or
    ``"no_slip"`` and
    refers to the lateral wall at ``s=1``.  The separated axial parity imposes
    impermeability, but not no-slip, at the endcaps.
    Magnetic conditions match the interior field to a decaying exterior
    potential proportional to ``K_abs(m)(k*s)``.
    """

    if k <= 0:
        raise ValueError("k must be positive for the exterior vacuum matching")
    if mechanical_boundary not in {"inviscid", "stress_free", "no_slip"}:
        raise ValueError(
            "mechanical_boundary must be 'inviscid', 'stress_free' or 'no_slip'"
        )
    bulk = assemble_mhd_bulk_matrices(
        size=size, m=m, k=k, Le=Le, E=E, Em=Em, coriolis=coriolis
    )
    A, B = bulk.A.copy(), bulk.B.copy()
    q, q1, q2, q3, lm = _stable_wall_functionals(abs(m), size)
    lm_prime = q3 + q2 - (1 + m**2) * q1 + 2 * m**2 * q
    ratio = kvp(abs(m), k) / kv(abs(m), k)
    T, P, G, F = (slice(j * size, (j + 1) * size) for j in range(4))

    mechanical_rows = (
        (0,) if mechanical_boundary == "inviscid" else (0, size, size + 1)
    )
    magnetic_rows = (2 * size, 3 * size)
    rows = mechanical_rows + magnetic_rows
    for row in rows:
        A[row, :] = 0.0
        B[row, :] = 0.0

    # v_s = k P' + i m T = 0.
    A[mechanical_rows[0], T] = 1j * m * q
    A[mechanical_rows[0], P] = k * q1
    if mechanical_boundary == "stress_free":
        # (d/ds-1) v_phi = 0 and d v_z/ds = 0.
        A[mechanical_rows[1], T] = -q2 + q1
        A[mechanical_rows[1], P] = 1j * k * m * (q1 - 2 * q)
        A[mechanical_rows[2], P] = lm_prime
    elif mechanical_boundary == "no_slip":
        # v_phi = i*k*m*P-T' = 0 and v_z = -L_m P = 0.
        A[mechanical_rows[1], T] = -q1
        A[mechanical_rows[1], P] = 1j * k * m * q
        A[mechanical_rows[2], P] = lm
    # Vacuum continuity after eliminating the exterior amplitude.
    A[magnetic_rows[0], G] = 1j * m * q
    A[magnetic_rows[0], F] = -k * q1 + ratio * lm
    A[magnetic_rows[1], G] = -q1
    A[magnetic_rows[1], F] = -1j * m * k * q + (1j * m / k) * lm

    gauge_rows = ()
    if m == 0:
        gauge_rows = _border_axisymmetric_gauges(A, B, bulk.radial, size, 4)

    return BorderedMHDMatrices(
        nodes=bulk.nodes,
        A=A,
        B=B,
        radial=bulk.radial,
        boundary_rows=rows,
        gauge_rows=gauge_rows,
    )


def assemble_ideal_mhd_matrices(
    *, size: int, m: int, k: float, Le: float
) -> BorderedMHDMatrices:
    """Assemble ideal MHD with only lateral impermeability.

    At E=Em=0 the magnetic equations are algebraic in radius and the viscous
    tangential conditions disappear with the reduced differential order.
    Therefore only v_s(1)=0 borders the bulk four-field pencil.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if Le < 0:
        raise ValueError("Le must be non-negative")
    bulk = assemble_mhd_bulk_matrices(
        size=size, m=m, k=k, Le=Le, E=0.0, Em=0.0
    )
    A, B = bulk.A.copy(), bulk.B.copy()
    values = bulk.radial.derivatives[0][0].astype(complex)
    derivatives = bulk.radial.derivatives[1][0].astype(complex)
    T, P = slice(0, size), slice(size, 2 * size)
    boundary_row = 0
    A[boundary_row, :] = 0.0
    B[boundary_row, :] = 0.0
    A[boundary_row, T] = 1j * m * values
    A[boundary_row, P] = k * derivatives
    gauge_rows = ()
    if m == 0:
        gauge_rows = _border_axisymmetric_gauges(A, B, bulk.radial, size, 4)
    return BorderedMHDMatrices(
        nodes=bulk.nodes,
        A=A,
        B=B,
        radial=bulk.radial,
        boundary_rows=(boundary_row,),
        gauge_rows=gauge_rows,
    )


def assemble_viscous_hydrodynamic_matrices(
    *, size: int, m: int, k: float, E: float
) -> ViscousHydrodynamicMatrices:
    """Assemble the viscous rotating problem with lateral no-slip conditions."""

    if k <= 0:
        raise ValueError("k must be positive")
    if E <= 0:
        raise ValueError("E must be positive for the no-slip viscous problem")
    nodes = radial_lobatto_nodes(size)
    radial = radial_operator_matrices(nodes, m=m, modes=size, k=k)
    x = 2.0 * nodes**2 - 1.0
    ell = abs(m)
    I = np.empty((size, size), dtype=complex)
    D = np.empty_like(I)
    D2 = np.empty_like(I)
    q = np.empty(size, dtype=complex)
    q1 = np.empty(size, dtype=complex)
    lm = np.empty(size, dtype=complex)

    def regularized_laplacian(polynomial: Chebyshev) -> Chebyshev:
        return (
            8.0 * (ell + 1) * polynomial.deriv(1)
            + 8.0 * Chebyshev((1.0, 1.0)) * polynomial.deriv(2)
        )

    # Work directly in the Chebyshev representation. Converting high-degree
    # basis functions to monomials makes the viscous fourth-order block badly
    # conditioned long before the physical boundary layer is resolved.
    for column in range(size):
        basis = Chebyshev.basis(column)
        laplacian_hat = regularized_laplacian(basis)
        shifted_hat = laplacian_hat - k**2 * basis
        shifted_squared_hat = regularized_laplacian(shifted_hat) - k**2 * shifted_hat
        I[:, column] = basis(x)
        D[:, column] = shifted_hat(x)
        D2[:, column] = shifted_squared_hat(x)
        q[column] = basis(1.0)
        q1[column] = ell * basis(1.0) + 4.0 * basis.deriv(1)(1.0)
        lm[column] = laplacian_hat(1.0)

    if m == 0:
        I, D, D2 = (
            matrix.astype(complex)
            for matrix in _stable_axisymmetric_uncancelled_operators(
                nodes, size, k
            )
        )

    A = np.zeros((2 * size, 2 * size), dtype=complex)
    B = np.zeros_like(A)
    T, P = slice(0, size), slice(size, 2 * size)
    r1, r2 = slice(0, size), slice(size, 2 * size)
    A[r1, T] = -E * D
    A[r1, P] = -2.0 * k * I
    B[r1, T] = 1j * I
    A[r2, T] = 2.0 * k * I
    A[r2, P] = E * D2
    B[r2, P] = -1j * D

    rows = (0, size, size + 1)
    for row in rows:
        A[row, :] = 0.0
        B[row, :] = 0.0
    A[rows[0], T] = 1j * m * q
    A[rows[0], P] = k * q1
    A[rows[1], T] = -q1
    A[rows[1], P] = 1j * k * m * q
    A[rows[2], P] = lm
    gauge_rows = ()
    if m == 0:
        gauge_rows = _border_axisymmetric_gauges(A, B, radial, size, 2)
    return ViscousHydrodynamicMatrices(
        nodes=nodes, A=A, B=B, radial=radial,
        boundary_rows=rows, gauge_rows=gauge_rows,
    )


def assemble_inviscid_hydrodynamic_matrices(
    *, size: int, m: int, k: float
) -> HydrodynamicMatrices:
    """Assemble the Greenspan reference problem with only ``v_s=0``.

    The inviscid equations are second order in total after eliminating the
    algebraic toroidal potential.  Consequently regularity at the axis and one
    impermeability condition at the lateral wall close the problem; viscous
    tangential-stress conditions must not be retained when ``E`` is exactly
    zero.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    nodes = radial_lobatto_nodes(size)
    radial = radial_operator_matrices(nodes, m=m, modes=size, k=k)
    I = radial.regularized_values.astype(complex)
    I1 = radial.derivatives[1].astype(complex)
    D = radial.regularized_shifted.astype(complex)
    if m == 0:
        I, D, _ = (
            matrix.astype(complex)
            for matrix in _stable_axisymmetric_uncancelled_operators(
                nodes, size, k
            )
        )
    A = np.zeros((2 * size, 2 * size), dtype=complex)
    B = np.zeros_like(A)
    T, P = slice(0, size), slice(size, 2 * size)
    r1, r2 = slice(0, size), slice(size, 2 * size)
    A[r1, P] = -2.0 * k * I
    B[r1, T] = 1j * I
    A[r2, T] = 2.0 * k * I
    B[r2, P] = -1j * D

    boundary_row = size
    A[boundary_row, :] = 0.0
    B[boundary_row, :] = 0.0
    A[boundary_row, T] = 1j * m * I[0]
    A[boundary_row, P] = k * I1[0]
    gauge_rows = ()
    if m == 0:
        gauge_rows = _border_axisymmetric_gauges(A, B, radial, size, 2)
    return HydrodynamicMatrices(
        nodes=nodes,
        A=A,
        B=B,
        radial=radial,
        boundary_row=boundary_row,
        gauge_rows=gauge_rows,
    )
