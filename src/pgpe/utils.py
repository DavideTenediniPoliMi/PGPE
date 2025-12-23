from collections.abc import Iterable
from typing import Any

import numpy as np


def ensure_positive_int(value: Any, name: str) -> int:
    try:
        val = int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Expected integer for '{name}', got {value}") from e

    if val <= 0:
        raise ValueError(f"Expected positive integer for '{name}', got {val}")
    return val


def ensure_positive_float(value: Any, name: str) -> float:
    try:
        val = float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Expected float for '{name}', got {value}") from e

    if val <= 0.0:
        raise ValueError(f"Expected positive float for '{name}', got {val}")
    return val


def ensure_vector(
    value: float | Iterable[float], length: int, dtype: np.dtype
) -> np.ndarray:
    if np.isscalar(value):
        return np.full(length, value, dtype=dtype)

    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}")
    if arr.size != length:
        raise ValueError(f"Expected vector of length {length}, got {arr.size}")
    return arr
