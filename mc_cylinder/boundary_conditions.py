"""Axis and boundary-condition utilities.

The axial parity labels used here refer to physical components at the endcaps:

* ``horizontal_even``: (q_s, q_phi) ~ cos(k*z), q_z ~ sin(k*z);
* ``horizontal_odd``:  (q_s, q_phi) ~ sin(k*z), q_z ~ cos(k*z).

An axial derivative exchanges these two parity classes.
"""

from typing import Literal

AxialParity = Literal["horizontal_even", "horizontal_odd"]


def axial_derivative_parity(parity: AxialParity) -> AxialParity:
    """Return the component parity after one axial derivative."""

    if parity == "horizontal_even":
        return "horizontal_odd"
    if parity == "horizontal_odd":
        return "horizontal_even"
    raise ValueError(f"unknown axial parity: {parity!r}")


def greenspan_velocity_parity() -> AxialParity:
    """Parity imposed by impermeable endcaps on a Greenspan velocity mode."""

    return "horizontal_even"


def perfectly_conducting_magnetic_parity() -> AxialParity:
    """Parity for b_n=0 and tangential current zero at plane endcaps."""

    return "horizontal_even"


def vertical_field_magnetic_parity() -> AxialParity:
    """Parity for b_t=0 and normal derivative of b_n zero."""

    return "horizontal_odd"


def induction_compatible(
    velocity_parity: AxialParity, magnetic_parity: AxialParity
) -> bool:
    """Check parity compatibility of dt(b) = dz(v) + Em*laplacian(b)."""

    return axial_derivative_parity(velocity_parity) == magnetic_parity
