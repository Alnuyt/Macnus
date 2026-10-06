"""Dimensionless parameters and conventions for the closed-cylinder problem."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    """Dimensionless control parameters.

    Length is scaled by the cylinder radius ``a`` and time by ``Omega**-1``.
    Consequently the dimensionless cylinder has radius 1 and height ``Gamma``.
    """

    Gamma: float = 1.0
    Le: float = 0.0
    E: float = 0.0
    Em: float = 0.0

    def __post_init__(self) -> None:
        if self.Gamma <= 0:
            raise ValueError("Gamma must be strictly positive")
        if self.Le < 0 or self.E < 0 or self.Em < 0:
            raise ValueError("Le, E, and Em must be non-negative")

    @property
    def Pm(self) -> float:
        """Magnetic Prandtl number E/Em.

        It is undefined in the ideal magnetic limit ``Em = 0``.
        """

        if self.Em == 0:
            raise ZeroDivisionError("Pm is undefined when Em = 0")
        return self.E / self.Em

    def axial_wavenumber(self, n: int) -> float:
        """Return k_n = n*pi/Gamma for a positive axial mode number."""

        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")
        from math import pi

        return n * pi / self.Gamma

