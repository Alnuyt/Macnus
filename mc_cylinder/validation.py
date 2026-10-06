"""Independent numerical benchmarks for the cylindrical eigensolvers."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .discretization import (
    assemble_ideal_mhd_matrices,
    assemble_inviscid_hydrodynamic_matrices,
    assemble_mhd_matrices,
    assemble_viscous_hydrodynamic_matrices,
)
from .greenspan import (
    greenspan_frequencies,
    ideal_magnetocoriolis_frequencies,
)
from .solver import nearest_eigenvalues, solve_dense_generalized


@lru_cache(maxsize=None)
def _cached_greenspan_frequencies(
    m: int, n: int, Gamma: float, branch: int, count: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    roots, frequencies = greenspan_frequencies(
        m, n, Gamma, branch=branch, count=count
    )
    return tuple(roots), tuple(frequencies)


@dataclass(frozen=True)
class GreenspanBenchmark:
    """One analytical/numerical Kelvin--Greenspan eigenfrequency match."""

    m: int
    n: int
    radial_index: int
    branch: int
    resolution: int
    radial_root: float
    exact_frequency: float
    numerical_frequency: complex
    absolute_error: float
    relative_error: float
    backward_residual: float


@dataclass(frozen=True)
class IdealMCBenchmark:
    """One analytical/numerical ideal magneto--Coriolis match."""

    m: int
    n: int
    radial_index: int
    inertial_branch: int
    mc_branch: str
    resolution: int
    Le: float
    inertial_frequency: float
    exact_frequency: float
    numerical_frequency: complex
    absolute_error: float
    scaled_error: float
    dispersion_residual: float
    backward_residual: float


@dataclass(frozen=True)
class ViscousInertialMode:
    """One no-slip viscous continuation of an inviscid inertial mode."""

    m: int
    n: int
    radial_index: int
    branch: int
    resolution: int
    E: float
    inviscid_frequency: float
    frequency: complex
    backward_residual: float
    boundary_residual: float


@dataclass(frozen=True)
class DissipativeMHDMode:
    """One mode of the fully bordered viscous--resistive MHD pencil."""

    resolution: int
    m: int
    k: float
    Le: float
    E: float
    Em: float
    frequency: complex
    eigenvector: np.ndarray
    fingerprint: np.ndarray
    backward_residual: float
    boundary_residual: float
    overlap_with_previous: float | None


@dataclass(frozen=True)
class ModeEnergies:
    """Radially integrated kinetic and magnetic energy diagnostics."""

    kinetic: float
    magnetic_unweighted: float
    magnetic: float
    magnetic_to_kinetic: float


@dataclass(frozen=True)
class DissipativeMHDSpectrum:
    """Filtered finite spectrum and diagnostics of the bordered MHD pencil."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    backward_residuals: np.ndarray
    boundary_residuals: np.ndarray
    m: int
    k: float
    Le: float
    E: float
    Em: float
    resolution: int


def greenspan_benchmark(
    *,
    m: int,
    n: int,
    Gamma: float,
    resolution: int,
    radial_count: int = 3,
    branches: tuple[int, ...] = (1, -1),
    residual_tolerance: float = 1.0e-9,
) -> tuple[GreenspanBenchmark, ...]:
    """Compare an independent hydrodynamic pencil with analytical modes."""

    k = n * np.pi / Gamma
    problem = assemble_inviscid_hydrodynamic_matrices(
        size=resolution, m=m, k=k
    )
    solution = solve_dense_generalized(
        problem.A, problem.B, residual_tolerance=residual_tolerance
    )

    roots_and_targets = []
    for branch in branches:
        roots, targets = _cached_greenspan_frequencies(
            m, n, Gamma, branch, radial_count
        )
        roots_and_targets.extend(
            (branch, radial_index, root, target)
            for radial_index, (root, target) in enumerate(
                zip(roots, targets), start=1
            )
        )

    targets = np.asarray([item[3] for item in roots_and_targets])
    matches, errors = nearest_eigenvalues(solution.eigenvalues, targets)
    records = []
    for item, match, error in zip(roots_and_targets, matches, errors):
        branch, radial_index, root, target = item
        eigen_index = int(np.argmin(np.abs(solution.eigenvalues - target)))
        records.append(
            GreenspanBenchmark(
                m=m,
                n=n,
                radial_index=radial_index,
                branch=branch,
                resolution=resolution,
                radial_root=float(root),
                exact_frequency=float(target),
                numerical_frequency=complex(match),
                absolute_error=float(error),
                relative_error=float(error / abs(target)),
                backward_residual=float(solution.residuals[eigen_index]),
            )
        )
    return tuple(records)


def greenspan_convergence(
    *,
    modes: tuple[tuple[int, int], ...],
    resolutions: tuple[int, ...],
    Gamma: float,
    radial_count: int = 3,
) -> dict[tuple[int, int], np.ndarray]:
    """Return the worst relative error over radial roots and both branches."""

    curves = {}
    for m, n in modes:
        curves[(m, n)] = np.asarray(
            [
                max(
                    record.relative_error
                    for record in greenspan_benchmark(
                        m=m,
                        n=n,
                        Gamma=Gamma,
                        resolution=resolution,
                        radial_count=radial_count,
                    )
                )
                for resolution in resolutions
            ]
        )
    return curves


def ideal_mc_benchmark(
    *,
    m: int,
    n: int,
    Gamma: float,
    Le: float,
    resolution: int,
    radial_count: int = 3,
    inertial_branches: tuple[int, ...] = (1, -1),
    residual_tolerance: float = 1.0e-9,
) -> tuple[IdealMCBenchmark, ...]:
    """Compare the ideal four-field pencil with all analytical MC branches."""

    k = n * np.pi / Gamma
    problem = assemble_ideal_mhd_matrices(
        size=resolution, m=m, k=k, Le=Le
    )
    solution = solve_dense_generalized(
        problem.A, problem.B, residual_tolerance=residual_tolerance
    )

    targets = []
    for inertial_branch in inertial_branches:
        _, inertial = _cached_greenspan_frequencies(
            m, n, Gamma, inertial_branch, radial_count
        )
        for radial_index, omega_i in enumerate(inertial, start=1):
            fast, slow = ideal_magnetocoriolis_frequencies(omega_i, k, Le)
            targets.extend(
                (inertial_branch, radial_index, name, omega_i, target)
                for name, target in (("fast", fast), ("slow", slow))
            )

    exact = np.asarray([target[-1] for target in targets])
    matches, errors = nearest_eigenvalues(solution.eigenvalues, exact)
    records = []
    for metadata, match, error in zip(targets, matches, errors):
        inertial_branch, radial_index, name, omega_i, target = metadata
        eigen_index = int(np.argmin(np.abs(solution.eigenvalues - target)))
        dispersion = target**2 - omega_i * target - (Le * k) ** 2
        records.append(
            IdealMCBenchmark(
                m=m,
                n=n,
                radial_index=radial_index,
                inertial_branch=inertial_branch,
                mc_branch=name,
                resolution=resolution,
                Le=float(Le),
                inertial_frequency=float(omega_i),
                exact_frequency=float(target),
                numerical_frequency=complex(match),
                absolute_error=float(error),
                scaled_error=float(error / max(1.0, abs(target))),
                dispersion_residual=float(abs(dispersion)),
                backward_residual=float(solution.residuals[eigen_index]),
            )
        )
    return tuple(records)


def viscous_inertial_mode(
    *,
    m: int,
    n: int,
    Gamma: float,
    E: float,
    resolution: int,
    radial_index: int = 1,
    branch: int = 1,
    residual_tolerance: float = 1.0e-8,
) -> ViscousInertialMode:
    """Return the viscous eigenvalue nearest a Kelvin--Greenspan mode."""

    if radial_index < 1:
        raise ValueError("radial_index must be positive")
    k = n * np.pi / Gamma
    _, targets = _cached_greenspan_frequencies(
        m, n, Gamma, branch, radial_index
    )
    target = targets[radial_index - 1]
    problem = assemble_viscous_hydrodynamic_matrices(
        size=resolution, m=m, k=k, E=E
    )
    solution = solve_dense_generalized(
        problem.A, problem.B, residual_tolerance=residual_tolerance
    )
    eigen_index = int(np.argmin(np.abs(solution.eigenvalues - target)))
    frequency = solution.eigenvalues[eigen_index]
    vector = solution.eigenvectors[:, eigen_index]
    boundary_values = problem.A[list(problem.boundary_rows)] @ vector
    boundary_scale = np.linalg.norm(problem.A[list(problem.boundary_rows)], ord=2)
    denominator = boundary_scale * np.linalg.norm(vector)
    boundary_residual = np.linalg.norm(boundary_values) / denominator
    return ViscousInertialMode(
        m=m,
        n=n,
        radial_index=radial_index,
        branch=branch,
        resolution=resolution,
        E=float(E),
        inviscid_frequency=float(target),
        frequency=complex(frequency),
        backward_residual=float(solution.residuals[eigen_index]),
        boundary_residual=float(boundary_residual),
    )


def modal_fingerprint(
    eigenvector: np.ndarray, *, m: int, samples: int = 160
) -> np.ndarray:
    """Evaluate four physical potentials on a common radial comparison grid."""

    vector = np.asarray(eigenvector, dtype=complex)
    if vector.ndim != 1 or vector.size % 4:
        raise ValueError("eigenvector length must be divisible by four")
    if samples < 4:
        raise ValueError("samples must be at least four")
    size = vector.size // 4
    s = np.linspace(0.0, 1.0, samples)
    x = 2.0 * s**2 - 1.0
    physical = [
        s**abs(m) * np.polynomial.chebyshev.chebval(x, coefficients)
        for coefficients in np.split(vector, 4)
    ]
    fingerprint = np.concatenate(physical)
    norm = np.linalg.norm(fingerprint)
    if norm == 0.0:
        raise ValueError("zero eigenvector fingerprint")
    return fingerprint / norm


def fingerprint_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return the phase-invariant normalized overlap of two fingerprints."""

    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("fingerprints must be one-dimensional with equal shape")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        raise ValueError("fingerprints must be non-zero")
    return float(abs(np.vdot(a, b)) / denominator)


def mode_energies(
    eigenvector: np.ndarray, *, m: int, k: float, Le: float,
    quadrature_order: int = 160,
) -> ModeEnergies:
    """Integrate modal energies using the physical cylindrical components."""

    vector = np.asarray(eigenvector, dtype=complex)
    if vector.ndim != 1 or vector.size % 4:
        raise ValueError("eigenvector length must be divisible by four")
    if k <= 0 or Le < 0 or quadrature_order < 8:
        raise ValueError("require k>0, Le>=0 and quadrature_order>=8")
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(quadrature_order)
    s = 0.5 * (gauss_x + 1.0)
    weights = 0.5 * gauss_w
    x = 2.0 * s**2 - 1.0
    ell = abs(m)

    def evaluate(coefficients: np.ndarray):
        derivative = np.polynomial.chebyshev.chebder(coefficients)
        second = np.polynomial.chebyshev.chebder(coefficients, 2)
        f = np.polynomial.chebyshev.chebval(x, coefficients)
        f1 = np.polynomial.chebyshev.chebval(x, derivative)
        f2 = np.polynomial.chebyshev.chebval(x, second)
        q = s**ell * f
        q1 = ell * s ** (ell - 1) * f + 4.0 * s ** (ell + 1) * f1
        lm = s**ell * (8.0 * (ell + 1) * f1 + 8.0 * (x + 1.0) * f2)
        return q, q1, lm

    T, P, G, F = (evaluate(block) for block in np.split(vector, 4))
    tq, tq1, _ = T
    pq, pq1, plm = P
    gq, gq1, _ = G
    fq, fq1, flm = F
    velocity_density = (
        abs(k * pq1 + 1j * m * tq / s) ** 2
        + abs(1j * k * m * pq / s - tq1) ** 2
        + abs(plm) ** 2
    )
    magnetic_density = (
        abs(1j * m * gq / s - k * fq1) ** 2
        + abs(-gq1 - 1j * k * m * fq / s) ** 2
        + abs(flm) ** 2
    )
    # Azimuthal integration and the common axial parity factor cancel in the
    # ratio, so only the cylindrical radial measure s ds is required.
    kinetic = float(np.sum(weights * s * velocity_density).real)
    magnetic_unweighted = float(np.sum(weights * s * magnetic_density).real)
    magnetic = float(Le**2 * magnetic_unweighted)
    return ModeEnergies(
        kinetic=kinetic,
        magnetic_unweighted=magnetic_unweighted,
        magnetic=magnetic,
        magnetic_to_kinetic=magnetic / kinetic,
    )


def solve_dissipative_mhd_mode(
    *,
    resolution: int,
    m: int,
    k: float,
    Le: float,
    E: float,
    Em: float,
    target: complex,
    previous_fingerprint: np.ndarray | None = None,
    residual_tolerance: float = 1.0e-8,
    frequency_window: float = 0.35,
    mechanical_boundary: str = "no_slip",
) -> DissipativeMHDMode:
    """Solve and select a stable MHD mode by frequency and field overlap."""

    problem = assemble_mhd_matrices(
    size=resolution,
    m=m,
    k=k,
    Le=Le,
    E=E,
    Em=Em,
    mechanical_boundary=mechanical_boundary,
    )
    solution = solve_dense_generalized(
        problem.A, problem.B, residual_tolerance=residual_tolerance
    )
    stable = np.flatnonzero(solution.eigenvalues.imag <= 1.0e-9)
    if stable.size == 0:
        raise RuntimeError("no stable finite eigenvalue survived filtering")
    scale = max(1.0, abs(target))
    nearby = stable[np.abs(solution.eigenvalues[stable] - target) <= frequency_window * scale]
    candidates = nearby if nearby.size else stable
    fingerprints = {
        index: modal_fingerprint(solution.eigenvectors[:, index], m=m)
        for index in candidates
    }
    if previous_fingerprint is None:
        selected = int(candidates[np.argmin(abs(solution.eigenvalues[candidates] - target))])
        overlap = None
    else:
        scores = np.asarray([
            fingerprint_overlap(previous_fingerprint, fingerprints[index])
            / (1.0 + abs(solution.eigenvalues[index] - target) / scale)
            for index in candidates
        ])
        selected = int(candidates[np.argmax(scores)])
        overlap = fingerprint_overlap(previous_fingerprint, fingerprints[selected])

    vector = solution.eigenvectors[:, selected]
    boundary_matrix = problem.A[list(problem.boundary_rows)]
    denominator = np.linalg.norm(boundary_matrix, ord=2) * np.linalg.norm(vector)
    boundary_residual = np.linalg.norm(boundary_matrix @ vector) / denominator
    return DissipativeMHDMode(
        resolution=resolution, m=m, k=float(k), Le=float(Le), E=float(E), Em=float(Em),
        frequency=complex(solution.eigenvalues[selected]), eigenvector=vector,
        fingerprint=fingerprints[selected],
        backward_residual=float(solution.residuals[selected]),
        boundary_residual=float(boundary_residual),
        overlap_with_previous=overlap,
    )


def solve_dissipative_mhd_spectrum(
    *, resolution: int, m: int, k: float, Le: float, E: float, Em: float,
    residual_tolerance: float = 1.0e-8,
) -> DissipativeMHDSpectrum:
    """Return the complete resolved spectrum before any physical ranking."""

    problem = assemble_mhd_matrices(
        size=resolution, m=m, k=k, Le=Le, E=E, Em=Em,
        mechanical_boundary="no_slip",
    )
    solution = solve_dense_generalized(
        problem.A, problem.B, residual_tolerance=residual_tolerance
    )
    boundary_matrix = problem.A[list(problem.boundary_rows)]
    boundary_norm = np.linalg.norm(boundary_matrix, ord=2)
    boundary_residuals = np.asarray([
        np.linalg.norm(boundary_matrix @ vector)
        / (boundary_norm * np.linalg.norm(vector))
        for vector in solution.eigenvectors.T
    ])
    return DissipativeMHDSpectrum(
        eigenvalues=solution.eigenvalues,
        eigenvectors=solution.eigenvectors,
        backward_residuals=solution.residuals,
        boundary_residuals=boundary_residuals,
        m=m, k=float(k), Le=float(Le), E=float(E), Em=float(Em),
        resolution=resolution,
    )


def dominant_spectrum_indices(
    spectrum: DissipativeMHDSpectrum,
    *, count: int = 10, min_abs_real_frequency: float = 0.05,
    min_quality_factor: float = 0.0, stability_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Rank stable oscillatory eigenmodes by decreasing temporal dominance."""

    if count < 1 or min_abs_real_frequency < 0 or min_quality_factor < 0:
        raise ValueError("invalid ranking threshold")
    omega = spectrum.eigenvalues
    damping = -omega.imag
    quality = np.abs(omega.real) / np.maximum(2.0 * damping, np.finfo(float).tiny)
    keep = (
        (omega.imag <= stability_tolerance)
        & (np.abs(omega.real) >= min_abs_real_frequency)
        & (quality >= min_quality_factor)
    )
    candidates = np.flatnonzero(keep)
    if candidates.size == 0:
        raise ValueError("no eigenmode satisfies the physical filters")
    # Largest imaginary part means slowest decay with exp(-i omega t).
    order = np.lexsort((np.abs(omega[candidates].real), -omega[candidates].imag))
    return candidates[order[:count]]
