"""Explicit symbolic differential operators in cylindrical coordinates.

Vectors are represented by three-component tuples in the physical orthonormal
basis ``(e_s, e_phi, e_z)``.  The basis dependence is included explicitly in
every operator; no Cartesian-coordinate shortcut is used here.
"""

from collections.abc import Sequence

import sympy as sp

Vector = tuple[sp.Expr, sp.Expr, sp.Expr]


def _vector3(vector: Sequence[sp.Expr]) -> Vector:
    """Return a vector as a tuple and reject ambiguous component counts."""

    if len(vector) != 3:
        raise ValueError("a cylindrical vector must have exactly three components")
    return tuple(sp.sympify(component) for component in vector)  # type: ignore[return-value]


def grad_cyl(
    scalar: sp.Expr, s: sp.Symbol, phi: sp.Symbol, z: sp.Symbol
) -> Vector:
    """Gradient in the physical cylindrical basis."""

    scalar = sp.sympify(scalar)
    return (
        sp.diff(scalar, s),
        sp.diff(scalar, phi) / s,
        sp.diff(scalar, z),
    )


def div_cyl(
    vector: Sequence[sp.Expr], s: sp.Symbol, phi: sp.Symbol, z: sp.Symbol
) -> sp.Expr:
    """Divergence in cylindrical coordinates."""

    vector_s, vector_phi, vector_z = _vector3(vector)
    return (
        sp.diff(s * vector_s, s) / s
        + sp.diff(vector_phi, phi) / s
        + sp.diff(vector_z, z)
    )


def curl_cyl(
    vector: Sequence[sp.Expr], s: sp.Symbol, phi: sp.Symbol, z: sp.Symbol
) -> Vector:
    """Curl in the physical cylindrical basis."""

    vector_s, vector_phi, vector_z = _vector3(vector)
    return (
        sp.diff(vector_z, phi) / s - sp.diff(vector_phi, z),
        sp.diff(vector_s, z) - sp.diff(vector_z, s),
        sp.diff(s * vector_phi, s) / s - sp.diff(vector_s, phi) / s,
    )


def laplacian_scalar_cyl(
    scalar: sp.Expr, s: sp.Symbol, phi: sp.Symbol, z: sp.Symbol
) -> sp.Expr:
    """Scalar Laplacian in cylindrical coordinates."""

    scalar = sp.sympify(scalar)
    return (
        sp.diff(s * sp.diff(scalar, s), s) / s
        + sp.diff(scalar, phi, 2) / s**2
        + sp.diff(scalar, z, 2)
    )


def laplacian_vector_cyl(
    vector: Sequence[sp.Expr], s: sp.Symbol, phi: sp.Symbol, z: sp.Symbol
) -> Vector:
    """Vector Laplacian in the physical cylindrical basis.

    The connection terms caused by the position-dependent basis are displayed
    explicitly. Applying the scalar Laplacian componentwise would be wrong.
    """

    vector_s, vector_phi, vector_z = _vector3(vector)
    return (
        laplacian_scalar_cyl(vector_s, s, phi, z)
        - vector_s / s**2
        - 2 * sp.diff(vector_phi, phi) / s**2,
        laplacian_scalar_cyl(vector_phi, s, phi, z)
        - vector_phi / s**2
        + 2 * sp.diff(vector_s, phi) / s**2,
        laplacian_scalar_cyl(vector_z, s, phi, z),
    )


def vector_add(left: Sequence[sp.Expr], right: Sequence[sp.Expr]) -> Vector:
    """Componentwise vector addition."""

    left3, right3 = _vector3(left), _vector3(right)
    return tuple(a + b for a, b in zip(left3, right3))  # type: ignore[return-value]


def vector_sub(left: Sequence[sp.Expr], right: Sequence[sp.Expr]) -> Vector:
    """Componentwise vector subtraction."""

    left3, right3 = _vector3(left), _vector3(right)
    return tuple(a - b for a, b in zip(left3, right3))  # type: ignore[return-value]


def scalar_mul(scalar: sp.Expr, vector: Sequence[sp.Expr]) -> Vector:
    """Multiply a cylindrical vector by a scalar."""

    scalar, vector3 = sp.sympify(scalar), _vector3(vector)
    return tuple(scalar * component for component in vector3)  # type: ignore[return-value]


def dot(left: Sequence[sp.Expr], right: Sequence[sp.Expr]) -> sp.Expr:
    """Euclidean dot product of physical components."""

    left3, right3 = _vector3(left), _vector3(right)
    return sum(a * b for a, b in zip(left3, right3))


def cross(left: Sequence[sp.Expr], right: Sequence[sp.Expr]) -> Vector:
    """Right-handed cross product in the basis (e_s, e_phi, e_z)."""

    a_s, a_phi, a_z = _vector3(left)
    b_s, b_phi, b_z = _vector3(right)
    return (
        a_phi * b_z - a_z * b_phi,
        a_z * b_s - a_s * b_z,
        a_s * b_phi - a_phi * b_s,
    )


def simplify_vector(vector: Sequence[sp.Expr]) -> Vector:
    """Apply SymPy simplification independently to all components."""

    return tuple(sp.simplify(value) for value in _vector3(vector))  # type: ignore[return-value]
