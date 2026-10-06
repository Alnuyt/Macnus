"""Reconstruction of physical cylindrical fields from spectral potentials."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

def potential_radial_data(
    coefficients: np.ndarray, s: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``q``, ``q'``, ``q/s`` and ``L_m q`` with analytic axis limits."""

    coefficients = np.asarray(coefficients, dtype=complex)
    points = np.asarray(s, dtype=float)
    if coefficients.ndim != 1 or points.ndim != 1:
        raise ValueError("coefficients and s must be one-dimensional")
    ell = abs(m)
    x = 2.0 * points**2 - 1.0
    first = np.polynomial.chebyshev.chebder(coefficients)
    second = np.polynomial.chebyshev.chebder(coefficients, 2)
    f = np.polynomial.chebyshev.chebval(x, coefficients)
    fx = np.polynomial.chebyshev.chebval(x, first)
    fxx = np.polynomial.chebyshev.chebval(x, second)
    q = points**ell * f
    qs = np.zeros_like(q)
    positive = points > 0.0
    qs[positive] = (
        ell * points[positive] ** (ell - 1) * f[positive]
        + 4.0 * points[positive] ** (ell + 1) * fx[positive]
    )
    if ell == 1:
        qs[~positive] = f[~positive]
    quotient = np.zeros_like(q)
    quotient[positive] = q[positive] / points[positive]
    if abs(m) == 1:
        quotient[~positive] = qs[~positive]
    laplacian = points**ell * (
        8.0 * (ell + 1) * fx + 8.0 * (x + 1.0) * fxx
    )
    return q, qs, quotient, laplacian


def compute_field_components(
    solution: np.ndarray,
    s: np.ndarray,
    indices: tuple[np.ndarray, ...],
    m: int,
    k: float,
    omega: complex,
    height: float,
    time: float = 0.0,
    kind: str = "velocity",
) -> tuple[np.ndarray, ...]:
    """Reconstruct velocity or magnetic components on a cylindrical grid."""

    if kind not in {"velocity", "magnetic"}:
        raise ValueError("kind must be 'velocity' or 'magnetic'")
    blocks = tuple(np.asarray(solution)[index] for index in indices)
    first, second = blocks[:2] if kind == "velocity" else blocks[2:]
    _, first_s, first_over_s, _ = potential_radial_data(first, s, m)
    _, second_s, second_over_s, lm_second = potential_radial_data(second, s, m)

    if kind == "velocity":
        radial = k * second_s + 1j * m * first_over_s
        azimuthal = 1j * k * m * second_over_s - first_s
        axial = -lm_second
        axial_factors = ("cos", "cos", "sin")
    else:
        radial = -k * second_s + 1j * m * first_over_s
        azimuthal = -1j * k * m * second_over_s - first_s
        axial = -lm_second
        axial_factors = ("sin", "sin", "cos")

    phi = np.linspace(0.0, 2.0 * np.pi, len(s))
    z = np.linspace(0.0, height, len(s))
    S, PHI, Z = np.meshgrid(s, phi, z, indexing="ij")
    phase = np.exp(1j * (m * PHI - omega * time))
    profiles = (radial, azimuthal, axial)
    factors = tuple(
        np.cos(k * Z) if name == "cos" else np.sin(k * Z)
        for name in axial_factors
    )
    fields = tuple(
        (profile[:, None, None] * factor * phase).real
        for profile, factor in zip(profiles, factors)
    )
    return S, PHI, Z, *fields


def radial_component_profiles(
    solution: np.ndarray,
    s: np.ndarray,
    indices: tuple[np.ndarray, ...],
    m: int,
    k: float,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complex radial amplitudes of the three cylindrical components."""

    if kind not in {"velocity", "magnetic"}:
        raise ValueError("kind must be 'velocity' or 'magnetic'")
    blocks = tuple(np.asarray(solution)[index] for index in indices)
    first, second = blocks[:2] if kind == "velocity" else blocks[2:]
    _, first_s, first_over_s, _ = potential_radial_data(first, s, m)
    _, second_s, second_over_s, lm_second = potential_radial_data(second, s, m)
    if kind == "velocity":
        return (
            k * second_s + 1j * m * first_over_s,
            1j * k * m * second_over_s - first_s,
            -lm_second,
        )
    return (
        -k * second_s + 1j * m * first_over_s,
        -1j * k * m * second_over_s - first_s,
        -lm_second,
    )


def horizontal_cartesian_components(
    solution: np.ndarray,
    indices: tuple[np.ndarray, ...],
    *,
    m: int,
    k: float,
    omega: complex,
    z: float,
    time: float = 0.0,
    kind: str,
    resolution: int = 401,
) -> tuple[np.ndarray, ...]:
    """Evaluate one horizontal disk directly, without scattered interpolation."""

    axis = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(axis, axis)
    radius = np.sqrt(X**2 + Y**2)
    inside = radius <= 1.0
    phi = np.arctan2(Y, X)
    radial, azimuthal, axial = radial_component_profiles(
        solution, radius[inside], indices, m, k, kind
    )
    phase = np.exp(1j * (m * phi[inside] - omega * time))
    if kind == "velocity":
        horizontal_factor, axial_factor = np.cos(k * z), np.sin(k * z)
    else:
        horizontal_factor, axial_factor = np.sin(k * z), np.cos(k * z)
    radial = np.real(radial * horizontal_factor * phase)
    azimuthal = np.real(azimuthal * horizontal_factor * phase)
    vertical = np.real(axial * axial_factor * phase)
    vx = radial * np.cos(phi[inside]) - azimuthal * np.sin(phi[inside])
    vy = radial * np.sin(phi[inside]) + azimuthal * np.cos(phi[inside])

    fields = []
    for values in (vx, vy, vertical):
        field = np.full(X.shape, np.nan)
        field[inside] = values
        fields.append(field)
    return axis, axis, X, Y, *fields


def meridional_components(
    solution: np.ndarray,
    indices: tuple[np.ndarray, ...],
    *,
    m: int,
    k: float,
    omega: complex,
    height: float,
    phi: float = 0.0,
    time: float = 0.0,
    kind: str,
    resolution: int = 301,
) -> tuple[np.ndarray, ...]:
    """Evaluate a meridional half-plane directly on a uniform display grid."""

    s = np.linspace(0.0, 1.0, resolution)
    z = np.linspace(0.0, height, resolution)
    radial, azimuthal, axial = radial_component_profiles(
        solution, s, indices, m, k, kind
    )
    phase = np.exp(1j * (m * phi - omega * time))
    if kind == "velocity":
        horizontal_factor, axial_factor = np.cos(k * z), np.sin(k * z)
    else:
        horizontal_factor, axial_factor = np.sin(k * z), np.cos(k * z)
    return (
        s,
        z,
        np.real(horizontal_factor[:, None] * radial[None, :] * phase),
        np.real(horizontal_factor[:, None] * azimuthal[None, :] * phase),
        np.real(axial_factor[:, None] * axial[None, :] * phase),
    )


def velocity(*args, **kwargs):
    """Reconstruct the physical velocity field."""

    return compute_field_components(*args, **kwargs, kind="velocity")


def magnetic(*args, **kwargs):
    """Reconstruct the physical magnetic perturbation."""

    return compute_field_components(*args, **kwargs, kind="magnetic")


def cylindrical_to_cartesian(S, PHI, radial, azimuthal):
    """Convert horizontal coordinates and vector components to Cartesian form."""

    X, Y = S * np.cos(PHI), S * np.sin(PHI)
    vx = radial * np.cos(PHI) - azimuthal * np.sin(PHI)
    vy = radial * np.sin(PHI) + azimuthal * np.cos(PHI)
    return X, Y, vx, vy


def interpolate_cartesian(X, Y, *fields, Ncart: int = 500):
    """Interpolate fields on a regular Cartesian disk and mask its exterior."""

    axis = np.linspace(-1.0, 1.0, Ncart)
    XC, YC = np.meshgrid(axis, axis)
    source = np.column_stack((np.ravel(X), np.ravel(Y)))
    disk = XC**2 + YC**2 <= 1.0
    interpolated = []
    for field in fields:
        values = griddata(source, np.ravel(field), (XC, YC), method="linear")
        values[~disk] = np.nan
        interpolated.append(values)
    return axis, axis, XC, YC, interpolated
