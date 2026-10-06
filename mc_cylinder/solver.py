"""Dense reference solution and, later, PETSc/SLEPc eigensolvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig


@dataclass(frozen=True)
class DenseEigenSolution:
    """Finite generalized eigenpairs and their backward residuals."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residuals: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    infinite_count: int


def generalized_eigenpair_residuals(
    A: np.ndarray,
    B: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Return scale-invariant backward residuals for ``A x = omega B x``."""

    norm_a = np.linalg.norm(A, ord=2)
    norm_b = np.linalg.norm(B, ord=2)
    residuals = np.empty(eigenvalues.size, dtype=float)
    for index, (omega, vector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        vector_norm = np.linalg.norm(vector)
        denominator = (norm_a + abs(omega) * norm_b) * vector_norm
        numerator = np.linalg.norm(A @ vector - omega * (B @ vector))
        residuals[index] = numerator / denominator if denominator else np.inf
    return residuals


def solve_dense_generalized(
    A: np.ndarray,
    B: np.ndarray,
    *,
    beta_tolerance: float = 1.0e-12,
    residual_tolerance: float | None = None,
) -> DenseEigenSolution:
    """Solve a dense generalized eigenproblem and remove infinite eigenvalues.

    LAPACK returns homogeneous pairs ``(alpha, beta)``.  A pair is retained
    only when ``|beta|`` is large relative to its homogeneous scale.  An
    optional backward-residual threshold can then remove poorly resolved
    finite eigenpairs.
    """

    matrix_a = np.asarray(A, dtype=complex)
    matrix_b = np.asarray(B, dtype=complex)
    if matrix_a.ndim != 2 or matrix_a.shape[0] != matrix_a.shape[1]:
        raise ValueError("A must be square")
    if matrix_b.shape != matrix_a.shape:
        raise ValueError("A and B must have the same shape")
    if beta_tolerance <= 0:
        raise ValueError("beta_tolerance must be positive")

    homogeneous, eigenvectors = eig(
        matrix_a,
        matrix_b,
        right=True,
        homogeneous_eigvals=True,
        check_finite=True,
    )
    alpha, beta = homogeneous
    scale = np.maximum(np.abs(alpha), np.abs(beta))
    finite = (scale > 0.0) & (np.abs(beta) > beta_tolerance * scale)
    finite_alpha = alpha[finite]
    finite_beta = beta[finite]
    finite_vectors = eigenvectors[:, finite]
    eigenvalues = finite_alpha / finite_beta
    residuals = generalized_eigenpair_residuals(
        matrix_a, matrix_b, eigenvalues, finite_vectors
    )

    if residual_tolerance is not None:
        if residual_tolerance <= 0:
            raise ValueError("residual_tolerance must be positive")
        accurate = residuals <= residual_tolerance
        finite_alpha = finite_alpha[accurate]
        finite_beta = finite_beta[accurate]
        finite_vectors = finite_vectors[:, accurate]
        eigenvalues = eigenvalues[accurate]
        residuals = residuals[accurate]

    order = np.lexsort((eigenvalues.imag, eigenvalues.real))
    return DenseEigenSolution(
        eigenvalues=eigenvalues[order],
        eigenvectors=finite_vectors[:, order],
        residuals=residuals[order],
        alpha=finite_alpha[order],
        beta=finite_beta[order],
        infinite_count=int(matrix_a.shape[0] - np.count_nonzero(finite)),
    )


def nearest_eigenvalues(
    computed: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest computed value and absolute error for each target."""

    values = np.asarray(computed, dtype=complex)
    references = np.asarray(targets, dtype=complex)
    if values.size == 0:
        raise ValueError("computed eigenvalue array is empty")
    indices = np.asarray([np.argmin(np.abs(values - target)) for target in references])
    matches = values[indices]
    return matches, np.abs(matches - references)


def dominant_mode_indices(
    eigenvalues: np.ndarray,
    *,
    count: int = 10,
    min_abs_frequency: float = 0.0,
    max_abs_frequency: float = np.inf,
    stability_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Rank stable oscillatory modes from least to most strongly damped.

    With the convention ``exp(-i*omega*t)``, temporal dominance is determined
    by decreasing ``Im(omega)``.  Small positive imaginary parts within
    ``stability_tolerance`` are accepted as roundoff.  Frequency bounds allow
    quasi-stationary modes or unrelated high-frequency families to be omitted.
    """

    values = np.asarray(eigenvalues, dtype=complex)
    if count < 1:
        raise ValueError("count must be positive")
    if min_abs_frequency < 0 or max_abs_frequency <= min_abs_frequency:
        raise ValueError("frequency bounds must satisfy 0 <= min < max")
    if stability_tolerance < 0:
        raise ValueError("stability_tolerance must be non-negative")
    magnitude = np.abs(values.real)
    admissible = np.flatnonzero(
        np.isfinite(values)
        & (values.imag <= stability_tolerance)
        & (magnitude >= min_abs_frequency)
        & (magnitude <= max_abs_frequency)
    )
    if admissible.size == 0:
        raise ValueError("no eigenvalue satisfies the dominance filters")
    # Primary key: decreasing imaginary part. Secondary key: decreasing
    # oscillation frequency, which makes ties deterministic in the ideal case.
    order = np.lexsort((-magnitude[admissible], -values.imag[admissible]))
    return admissible[order[:count]]
