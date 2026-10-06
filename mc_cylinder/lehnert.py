"""Analytical torsional-wave benchmarks following Lehnert (1954).

The benchmark is axisymmetric and non-rotating: only ``v_phi`` and ``b_phi``
are non-zero in a cylinder threaded by a uniform axial magnetic field.  Length
and time are nondimensionalized by ``a`` and ``Omega_ref**-1`` so that the
Alfven frequency is ``Le*k`` and magnetic diffusion is measured by ``Em``.
The time convention is ``exp(-1j*omega*t)``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig
from scipy.special import jn_zeros, jv

from .discretization import (
    assemble_mhd_matrices,
    radial_lobatto_nodes,
    radial_operator_matrices,
)


def lehnert_radial_roots(count: int) -> np.ndarray:
    """Return radial roots for a solid cylinder: ``J_1(xi)=0`` at ``s=1``."""

    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    return jn_zeros(1, int(count))


def lehnert_radial_mode(s: np.ndarray, radial_index: int = 1) -> np.ndarray:
    """Return the regular torsional profile ``J_1(xi_j*s)``."""

    if not isinstance(radial_index, (int, np.integer)) or radial_index < 1:
        raise ValueError("radial_index must be a positive integer")
    points = np.asarray(s, dtype=float)
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("s must lie in [0, 1]")
    xi = lehnert_radial_roots(int(radial_index))[-1]
    return jv(1, xi * points)


def lehnert_free_frequencies(
    *, xi: float, k: float, Le: float, Em: float, E: float = 0.0
) -> tuple[complex, complex]:
    r"""Return the two viscous--resistive torsional eigenfrequencies.

    They solve

    ``(omega + i*E*K2)*(omega + i*Em*K2) - (Le*k)**2 = 0``,

    where ``K2=xi**2+k**2``.  Lehnert neglects viscosity, corresponding to
    ``E=0``.  The square-root branch is immaterial because both roots are
    returned.
    """

    if xi <= 0.0 or k <= 0.0:
        raise ValueError("xi and k must be positive")
    if Le < 0.0 or E < 0.0 or Em < 0.0:
        raise ValueError("Le, E and Em must be non-negative")
    K2 = xi**2 + k**2
    discriminant = (Le * k) ** 2 - 0.25 * ((E - Em) * K2) ** 2
    root = np.sqrt(discriminant + 0j)
    centre = -0.5j * (E + Em) * K2
    return complex(centre + root), complex(centre - root)


def lehnert_axial_wavenumber(
    *, omega: float, xi: float, Le: float, Em: float
) -> complex:
    r"""Return Lehnert's complex axial wavenumber at imposed real frequency.

    For negligible viscosity, elimination of ``v_phi`` and ``b_phi`` gives

    ``q**2 = (omega**2 + i*Em*omega*xi**2)/(Le**2-i*Em*omega)``.

    The returned square root has non-negative imaginary part, so a factor
    ``exp(i*q*z)`` attenuates as ``z`` increases.
    """

    if omega <= 0.0 or xi <= 0.0 or Le < 0.0 or Em < 0.0:
        raise ValueError("omega and xi must be positive; Le and Em non-negative")
    denominator = Le**2 - 1j * Em * omega
    if denominator == 0.0:
        raise ValueError("Le and Em cannot both vanish")
    q_squared = (omega**2 + 1j * Em * omega * xi**2) / denominator
    q = complex(np.sqrt(q_squared + 0j))
    return q if q.imag >= 0.0 else -q


def lehnert_dispersion_residual(
    omega: complex, *, xi: float, k: float, Le: float, Em: float, E: float = 0.0
) -> complex:
    """Evaluate the nondimensional torsional dispersion polynomial."""

    K2 = xi**2 + k**2
    return (omega + 1j * E * K2) * (omega + 1j * Em * K2) - (Le * k) ** 2


def solve_lehnert_torsional(
    *, size: int, k: float, Le: float, Em: float, E: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the axisymmetric torsional benchmark by radial collocation.

    Although the physical fields are axisymmetric, their azimuthal components
    are acted on by the order-one vector Laplacian
    ``d2/ds2 + (1/s)d/ds - 1/s2 - k2``.  The regular basis is consequently
    ``s*T_j(2*s**2-1)``.  Lehnert's inviscid-velocity case (``E=0``) needs only
    ``b_phi(1)=0``; when ``E>0``, ``v_phi(1)=0`` is added as well.
    """

    if size < 4:
        raise ValueError("size must be at least 4")
    if k <= 0.0 or Le < 0.0 or E < 0.0 or Em < 0.0:
        raise ValueError("k must be positive; Le, E and Em non-negative")
    nodes = radial_lobatto_nodes(size)
    radial = radial_operator_matrices(nodes, m=1, modes=size, k=k)
    # Divide the physical equations by the common regular factor s.  This
    # retains their finite, non-trivial limits at the cylindrical axis.
    I = radial.regularized_values.astype(complex)
    D = radial.regularized_shifted.astype(complex)
    A = np.zeros((2 * size, 2 * size), dtype=complex)
    B = np.zeros_like(A)
    v, b = slice(0, size), slice(size, 2 * size)
    A[v, v] = 1j * E * D
    A[v, b] = -(Le**2) * k * I
    A[b, v] = -k * I
    A[b, b] = 1j * Em * D
    B[v, v] = I
    B[b, b] = I

    boundary_rows = [size]
    if E > 0.0:
        boundary_rows.append(0)
    for row in boundary_rows:
        A[row] = 0.0
        B[row] = 0.0
    A[size, b] = I[0]  # b_phi(1)=0
    if E > 0.0:
        A[0, v] = I[0]  # v_phi(1)=0

    frequencies, eigenvectors = eig(A, B)
    finite = np.isfinite(frequencies)
    return frequencies[finite], eigenvectors[:, finite]


def solve_lehnert_with_general_mhd(
    *, size: int, k: float, Le: float, Em: float, target: complex
) -> tuple[complex, np.ndarray, object]:
    """Select a torsional ``m=0`` mode from the general four-potential pencil.

    The Coriolis coefficient is set to zero because the Lehnert benchmark is
    non-rotating.  Velocity is inviscid (``E=0``), magnetic diffusion is
    retained, and the same vacuum magnetic rows as in the general solver are
    used.  In the torsional invariant subspace only ``T`` and ``G`` survive;
    the vacuum condition then reduces to ``b_phi(1)=-G'(1)=0``.
    """

    problem = assemble_mhd_matrices(
        size=size, m=0, k=k, Le=Le, E=0.0, Em=Em,
        mechanical_boundary="inviscid", coriolis=0.0,
    )
    frequencies, eigenvectors = eig(problem.A, problem.B)
    finite = np.isfinite(frequencies)
    frequencies = frequencies[finite]
    eigenvectors = eigenvectors[:, finite]
    if frequencies.size == 0:
        raise RuntimeError("the general m=0 pencil has no finite eigenvalue")

    blocks = np.split(eigenvectors, 4, axis=0)
    torsional = np.linalg.norm(blocks[0], axis=0) ** 2
    torsional += np.linalg.norm(blocks[2], axis=0) ** 2
    total = np.linalg.norm(eigenvectors, axis=0) ** 2
    fraction = np.divide(torsional, total, out=np.zeros_like(torsional), where=total > 0)
    scale = max(1.0, abs(target))
    score = abs(frequencies - target) / scale + 10.0 * (1.0 - fraction)
    index = int(np.argmin(score))
    vector = eigenvectors[:, index]
    vector /= np.max(np.abs(vector))
    return complex(frequencies[index]), vector, problem
