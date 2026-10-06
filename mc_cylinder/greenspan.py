"""Analytical inertial modes of an impermeable rotating cylinder.

The convention is ``exp(i*m*phi - i*omega*t)`` and time is scaled by
``Omega**-1``. Consequently inertial frequencies lie in ``[-2, 2]`` and the
lateral impermeability condition is

    omega * xi * J_|m|'(xi) - 2*m*J_|m|(xi) = 0.
"""

from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, jvp


def inertial_frequency(xi: float, k: float, branch: int = 1) -> float:
    """Return ``omega = branch*2*k/sqrt(k**2 + xi**2)``."""

    if k <= 0:
        raise ValueError("k must be strictly positive")
    if xi < 0:
        raise ValueError("xi must be non-negative")
    if branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    return branch * 2.0 * k / np.hypot(k, xi)


def greenspan_residual(xi: float, m: int, k: float, branch: int = 1) -> float:
    """Residual of radial impermeability at ``s=1``."""

    omega = inertial_frequency(xi, k, branch)
    order = abs(m)
    return omega * xi * jvp(order, xi, 1) - 2.0 * m * jv(order, xi)


def legacy_greenspan_residual(xi: float, m: int, k: float) -> float:
    """Residual used with ``exp(i*m*phi + i*lambda*t)``, ``lambda > 0``.

    It is equivalent to the current convention at ``omega = -lambda`` or,
    by spectral symmetry, at azimuthal order ``-m`` and ``omega = lambda``.
    """

    if k <= 0:
        raise ValueError("k must be strictly positive")
    if xi < 0:
        raise ValueError("xi must be non-negative")
    order = abs(m)
    factor = np.sqrt(1.0 + (xi / k) ** 2)
    return xi * jvp(order, xi, 1) + m * factor * jv(order, xi)


def _bracketed_roots(
    function: Callable[[float], float],
    count: int,
    xi_max: float,
    samples: int,
) -> np.ndarray:
    """Find simple positive roots by a dense sign-change scan."""

    if count < 1:
        raise ValueError("count must be a positive integer")
    if xi_max <= 0 or samples < 2:
        raise ValueError("xi_max and samples must define a positive scan")

    # xi=0 implies |omega|=2, where the horizontal momentum matrix used in
    # the derivation is singular. For m>1, J_m and the residual are also so
    # small near zero that floating-point roundoff produces false sign changes.
    # All non-degenerate radial roots of interest are safely above this cutoff.
    xi_min = 1.0e-3
    grid = np.linspace(xi_min, xi_max, samples)
    values = np.asarray([function(value) for value in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left * f_right < 0:
            root = brentq(function, left, right)
            if root > xi_min and (not roots or abs(root - roots[-1]) > 1.0e-8):
                roots.append(root)
                if len(roots) == count:
                    return np.asarray(roots)
    raise RuntimeError(
        f"found only {len(roots)} roots below xi_max={xi_max}; "
        "increase xi_max or samples"
    )


def greenspan_roots(
    m: int,
    n: int,
    Gamma: float,
    *,
    branch: int = 1,
    count: int = 5,
    xi_max: float = 100.0,
    samples: int = 20000,
) -> np.ndarray:
    """Return the first positive radial roots for a closed cylinder."""

    if not isinstance(m, int) or isinstance(m, bool):
        raise ValueError("m must be an integer")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if Gamma <= 0:
        raise ValueError("Gamma must be strictly positive")
    k = n * np.pi / Gamma
    return _bracketed_roots(
        lambda xi: greenspan_residual(xi, m, k, branch),
        count=count,
        xi_max=xi_max,
        samples=samples,
    )


def legacy_greenspan_roots(
    m: int,
    n: int,
    Gamma: float,
    *,
    count: int = 5,
    xi_max: float = 100.0,
    samples: int = 20000,
) -> np.ndarray:
    """Roots in the former ``exp(+i*lambda*t)`` convention."""

    if not isinstance(m, int) or isinstance(m, bool):
        raise ValueError("m must be an integer")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if Gamma <= 0:
        raise ValueError("Gamma must be strictly positive")
    k = n * np.pi / Gamma
    return _bracketed_roots(
        lambda xi: legacy_greenspan_residual(xi, m, k),
        count=count,
        xi_max=xi_max,
        samples=samples,
    )


def legacy_greenspan_eigenvalues(
    m: int, n: int, Gamma: float, *, count: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy roots and positive values ``lambda = |omega|``."""

    roots = legacy_greenspan_roots(m, n, Gamma, count=count)
    k = n * np.pi / Gamma
    eigenvalues = 2.0 / np.sqrt(1.0 + (roots / k) ** 2)
    return roots, eigenvalues


def greenspan_frequencies(
    m: int,
    n: int,
    Gamma: float,
    *,
    branch: int = 1,
    count: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial roots and their inertial frequencies."""

    roots = greenspan_roots(m, n, Gamma, branch=branch, count=count)
    k = n * np.pi / Gamma
    frequencies = np.asarray(
        [inertial_frequency(xi, k, branch) for xi in roots]
    )
    return roots, frequencies


def ideal_magnetocoriolis_frequencies(
    omega_inertial: float, k: float, Le: float
) -> tuple[float, float]:
    """Return the ideal fast and slow MC frequencies for one inertial mode.

    For the nondimensional equations used in this project, elimination of the
    magnetic potentials gives

    ``omega - (Le*k)**2/omega = omega_inertial``.

    The fast root tends to ``omega_inertial`` and the slow root tends to zero
    as ``Le`` tends to zero.  The ordering returned here is ``(fast, slow)``.
    """

    if k <= 0:
        raise ValueError("k must be strictly positive")
    if Le < 0:
        raise ValueError("Le must be non-negative")
    if omega_inertial == 0:
        raise ValueError("omega_inertial must be non-zero")
    discriminant = np.sqrt(omega_inertial**2 + 4.0 * (Le * k) ** 2)
    sign = np.sign(omega_inertial)
    fast = 0.5 * (omega_inertial + sign * discriminant)
    slow = 0.5 * (omega_inertial - sign * discriminant)
    return float(fast), float(slow)
