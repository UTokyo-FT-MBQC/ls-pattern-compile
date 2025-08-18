from __future__ import annotations

from typing import Tuple

# Parity rule aligned with the original RHG implementation
ALLOWED_PARITIES = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
DATA_PARITIES    = [(0, 0, 0), (1, 1, 0), (0, 0, 1), (1, 1, 1)]
ANCILLA_Z_PARITY = (0, 1, 0)
ANCILLA_X_PARITY = (1, 0, 1)

def parity3(x: int, y: int, z: int) -> Tuple[int, int, int]:
    return (x & 1, y & 1, z & 1)

def is_allowed(x: int, y: int, z: int) -> bool:
    return parity3(x, y, z) in ALLOWED_PARITIES

def is_data(x: int, y: int, z: int) -> bool:
    return parity3(x, y, z) in DATA_PARITIES

def is_ancilla_x(x: int, y: int, z: int) -> bool:
    return parity3(x, y, z) == ANCILLA_X_PARITY

def is_ancilla_z(x: int, y: int, z: int) -> bool:
    return parity3(x, y, z) == ANCILLA_Z_PARITY