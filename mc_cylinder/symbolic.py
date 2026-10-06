"""Symbolic toroidal-poloidal fields and nondimensional MHD equations."""

from dataclasses import dataclass

import sympy as sp

from .cylindrical import (
    cross,
    curl_cyl,
    laplacian_vector_cyl,
    scalar_mul,
    simplify_vector,
    vector_add,
    vector_sub,
)


def toroidal_poloidal_es(
    toroidal: sp.Expr,
    poloidal: sp.Expr,
    s: sp.Symbol,
    phi: sp.Symbol,
    z: sp.Symbol,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``curl(T*e_s) + curl(curl(P*e_s))``.

    This radial-axis decomposition is solenoidal by construction but does not
    inherit the classical closure properties of a constant-axis decomposition.
    """

    e_s_toroidal = (sp.sympify(toroidal), sp.S.Zero, sp.S.Zero)
    e_s_poloidal = (sp.sympify(poloidal), sp.S.Zero, sp.S.Zero)
    return vector_add(
        curl_cyl(e_s_toroidal, s, phi, z),
        curl_cyl(curl_cyl(e_s_poloidal, s, phi, z), s, phi, z),
    )


def toroidal_poloidal_ez(
    toroidal: sp.Expr,
    poloidal: sp.Expr,
    s: sp.Symbol,
    phi: sp.Symbol,
    z: sp.Symbol,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``curl(T*e_z) + curl(curl(P*e_z))``.

    The distinguished direction ``e_z`` is constant. This is the classical
    cylindrical toroidal-poloidal decomposition used by the final model.
    """

    e_z_toroidal = (sp.S.Zero, sp.S.Zero, sp.sympify(toroidal))
    e_z_poloidal = (sp.S.Zero, sp.S.Zero, sp.sympify(poloidal))
    return vector_add(
        curl_cyl(e_z_toroidal, s, phi, z),
        curl_cyl(curl_cyl(e_z_poloidal, s, phi, z), s, phi, z),
    )


@dataclass(frozen=True)
class RadialMHDEquations:
    """Four radial residuals and the modal fields used to derive them."""

    momentum_curl: sp.Expr
    momentum_double_curl: sp.Expr
    induction_axial: sp.Expr
    induction_curl: sp.Expr
    velocity: tuple[sp.Expr, sp.Expr, sp.Expr]
    magnetic: tuple[sp.Expr, sp.Expr, sp.Expr]

    @property
    def residuals(self) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        return (
            self.momentum_curl,
            self.momentum_double_curl,
            self.induction_axial,
            self.induction_curl,
        )


def _time_harmonic(vector: tuple[sp.Expr, sp.Expr, sp.Expr], omega: sp.Symbol):
    return scalar_mul(-sp.I * omega, vector)


def _axial_derivative(
    vector: tuple[sp.Expr, sp.Expr, sp.Expr], z: sp.Symbol
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    return tuple(sp.diff(component, z) for component in vector)


def _project_modal_factor(
    expression: sp.Expr,
    *,
    phi: sp.Symbol,
    z: sp.Symbol,
    k: sp.Symbol,
    parity: str,
) -> sp.Expr:
    """Remove ``exp(i*m*phi)`` and one sine/cosine axial factor."""

    if parity == "sin":
        projected = expression.subs({phi: 0, z: sp.pi / (2 * k)})
    elif parity == "cos":
        projected = expression.subs({phi: 0, z: 0})
    else:
        raise ValueError("parity must be 'sin' or 'cos'")
    return sp.expand(projected)


def radial_laplacian(
    expression: sp.Expr, s: sp.Symbol, m: sp.Expr
) -> sp.Expr:
    """Return the radial Fourier Laplacian ``L_m`` acting on a scalar."""

    return sp.expand(
        sp.diff(expression, s, 2)
        + sp.diff(expression, s) / s
        - m**2 * expression / s**2
    )


def shifted_radial_laplacian(
    expression: sp.Expr, s: sp.Symbol, m: sp.Expr, k: sp.Expr
) -> sp.Expr:
    """Return ``D_(m,k) = L_m - k**2`` acting on a scalar."""

    return sp.expand(radial_laplacian(expression, s, m) - k**2 * expression)


def compact_radial_mhd_residuals(
    *,
    T: sp.Expr,
    P: sp.Expr,
    G: sp.Expr,
    F: sp.Expr,
    s: sp.Symbol,
    m: sp.Expr,
    k: sp.Expr,
    omega: sp.Expr,
    Le: sp.Expr,
    E: sp.Expr,
    Em: sp.Expr,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return the four MHD residuals in operator-derived compact form.

    The returned expressions are expanded only at the scalar-operator level so
    they can be compared directly with the full cylindrical-vector derivation.
    """

    LT = radial_laplacian(T, s, m)
    LP = radial_laplacian(P, s, m)
    LG = radial_laplacian(G, s, m)
    LF = radial_laplacian(F, s, m)

    return tuple(
        sp.expand(residual)
        for residual in (
            sp.I * omega * LT
            + E * shifted_radial_laplacian(LT, s, m, k)
            + 2 * k * LP
            + Le**2 * k * LG,
            -sp.I * omega * shifted_radial_laplacian(LP, s, m, k)
            - E
            * shifted_radial_laplacian(
                shifted_radial_laplacian(LP, s, m, k), s, m, k
            )
            - 2 * k * LT
            + Le**2 * k * shifted_radial_laplacian(LF, s, m, k),
            sp.I * omega * LF
            + Em * shifted_radial_laplacian(LF, s, m, k)
            + k * LP,
            sp.I * omega * LG
            + Em * shifted_radial_laplacian(LG, s, m, k)
            - k * LT,
        )
    )


def reduced_radial_mhd_residuals(
    *,
    T: sp.Expr,
    P: sp.Expr,
    G: sp.Expr,
    F: sp.Expr,
    s: sp.Symbol,
    m: sp.Expr,
    k: sp.Expr,
    omega: sp.Expr,
    Le: sp.Expr,
    E: sp.Expr,
    Em: sp.Expr,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return the physical radial system after cancelling common ``L_m``.

    On regular Fourier scalars each fully projected residual is ``L_m`` acting
    on the corresponding expression below.  Cancelling this common operator
    removes harmonic-potential nullspaces introduced by the projections.
    """

    DT = shifted_radial_laplacian(T, s, m, k)
    DP = shifted_radial_laplacian(P, s, m, k)
    DG = shifted_radial_laplacian(G, s, m, k)
    DF = shifted_radial_laplacian(F, s, m, k)
    D2P = shifted_radial_laplacian(DP, s, m, k)
    return tuple(
        sp.expand(residual)
        for residual in (
            sp.I * omega * T + E * DT + 2 * k * P + Le**2 * k * G,
            -sp.I * omega * DP - E * D2P - 2 * k * T + Le**2 * k * DF,
            sp.I * omega * F + Em * DF + k * P,
            sp.I * omega * G + Em * DG - k * T,
        )
    )


def derive_radial_mhd_equations(
    *,
    s: sp.Symbol,
    phi: sp.Symbol,
    z: sp.Symbol,
    m: sp.Symbol,
    k: sp.Symbol,
    omega: sp.Symbol,
    Le: sp.Symbol,
    E: sp.Symbol,
    Em: sp.Symbol,
) -> RadialMHDEquations:
    """Derive the four separated radial MHD residuals.

    The convention is ``exp(i*m*phi - i*omega*t)``. With the classical
    ``e_z`` decomposition, impermeable-endcap parity is ``T,F ~ cos(k*z)``
    and ``P,G ~ sin(k*z)``.  This parity does not impose no-slip tangential
    velocity at the endcaps; ``no_slip`` in the radial discretization refers
    only to the lateral wall.
    """

    phase = sp.exp(sp.I * m * phi)
    T = sp.Function("T")(s)
    P = sp.Function("P")(s)
    G = sp.Function("G")(s)
    F = sp.Function("F")(s)

    velocity = toroidal_poloidal_ez(
            T * phase * sp.cos(k * z),
            P * phase * sp.sin(k * z),
            s,
            phi,
            z,
        )
    magnetic = toroidal_poloidal_ez(
            G * phase * sp.sin(k * z),
            F * phase * sp.cos(k * z),
            s,
            phi,
            z,
        )

    momentum = vector_sub(
        vector_add(
            _time_harmonic(velocity, omega),
            scalar_mul(2, cross((0, 0, 1), velocity)),
        ),
        vector_add(
            scalar_mul(Le**2, _axial_derivative(magnetic, z)),
            scalar_mul(E, laplacian_vector_cyl(velocity, s, phi, z)),
        ),
    )
    induction = vector_sub(
        _time_harmonic(magnetic, omega),
        vector_add(
            _axial_derivative(velocity, z),
            scalar_mul(Em, laplacian_vector_cyl(magnetic, s, phi, z)),
        ),
    )

    curl_momentum = curl_cyl(momentum, s, phi, z)
    double_curl_momentum = curl_cyl(curl_momentum, s, phi, z)
    curl_induction = curl_cyl(induction, s, phi, z)

    return RadialMHDEquations(
        momentum_curl=_project_modal_factor(
            curl_momentum[2], phi=phi, z=z, k=k, parity="cos"
        ),
        momentum_double_curl=_project_modal_factor(
            double_curl_momentum[2], phi=phi, z=z, k=k, parity="sin"
        ),
        induction_axial=_project_modal_factor(
            induction[2], phi=phi, z=z, k=k, parity="cos"
        ),
        induction_curl=_project_modal_factor(
            curl_induction[2], phi=phi, z=z, k=k, parity="sin"
        ),
        velocity=velocity,
        magnetic=magnetic,
    )


def split_generalized_eigenvalue(
    expression: sp.Expr, omega: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """Return ``(A, B)`` such that ``expression = 0`` becomes ``A = omega*B``."""

    if sp.diff(expression, omega, 2) != 0:
        raise ValueError("the residual is not linear in omega")
    coefficient = sp.diff(expression, omega)
    remainder = expression.subs(omega, 0)
    if sp.simplify(expression - remainder - omega * coefficient) != 0:
        raise ValueError("failed to separate the eigenvalue coefficient")
    return sp.expand(-remainder), sp.expand(coefficient)
