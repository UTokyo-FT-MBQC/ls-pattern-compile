"""Parity rule definitions for RHG (Random Hashing Graph) implementation."""

from __future__ import annotations

# Parity rule aligned with the original RHG implementation
ALLOWED_PARITIES = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
DATA_PARITIES = [(0, 0, 0), (1, 1, 0), (0, 0, 1), (1, 1, 1)]
ANCILLA_Z_PARITY = (0, 1, 0)
ANCILLA_X_PARITY = (1, 0, 1)


def parity3(x: int, y: int, z: int) -> tuple[int, int, int]:
    """Calculate the parity of three coordinates.

    Parameters
    ----------
    x : int
        X coordinate
    y : int
        Y coordinate
    z : int
        Z coordinate

    Returns
    -------
    tuple[int, int, int]
        Tuple of parities (x&1, y&1, z&1)
    """
    return (x & 1, y & 1, z & 1)


def is_allowed(x: int, y: int, z: int) -> bool:
    """Check if coordinates have allowed parity.

    Parameters
    ----------
    x : int
        X coordinate
    y : int
        Y coordinate
    z : int
        Z coordinate

    Returns
    -------
    bool
        True if the parity is in ALLOWED_PARITIES
    """
    return parity3(x, y, z) in ALLOWED_PARITIES


def is_data(x: int, y: int, z: int) -> bool:
    """Check if coordinates correspond to data qubit parity.

    Parameters
    ----------
    x : int
        X coordinate
    y : int
        Y coordinate
    z : int
        Z coordinate

    Returns
    -------
    bool
        True if the parity is in DATA_PARITIES
    """
    return parity3(x, y, z) in DATA_PARITIES


def is_ancilla_x(x: int, y: int, z: int) -> bool:
    """Check if coordinates correspond to X ancilla qubit parity.

    Parameters
    ----------
    x : int
        X coordinate
    y : int
        Y coordinate
    z : int
        Z coordinate

    Returns
    -------
    bool
        True if the parity matches ANCILLA_X_PARITY
    """
    return parity3(x, y, z) == ANCILLA_X_PARITY


def is_ancilla_z(x: int, y: int, z: int) -> bool:
    """Check if coordinates correspond to Z ancilla qubit parity.

    Parameters
    ----------
    x : int
        X coordinate
    y : int
        Y coordinate
    z : int
        Z coordinate

    Returns
    -------
    bool
        True if the parity matches ANCILLA_Z_PARITY
    """
    return parity3(x, y, z) == ANCILLA_Z_PARITY
