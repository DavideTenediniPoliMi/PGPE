from collections.abc import Callable, Iterable
from typing import Any

from array_api.latest import Array, ArrayNamespace
from array_api_compat import array_namespace

try:
    import array_api_compat.numpy as np
except ImportError:
    np = None


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
    value: float | Iterable[float] | Array,
    length: int,
    xp: ArrayNamespace,
    dtype: Any,
    device: Any,
) -> Array:
    if isinstance(value, (int, float)):
        return xp.full(length, value, dtype=dtype, device=device)

    arr = xp.asarray(value, dtype=dtype, device=device)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}")
    if arr.shape[0] != length:
        raise ValueError(f"Expected vector of length {length}, got {arr.shape[0]}")
    return arr


def get_xp(*objs: Any) -> tuple[ArrayNamespace, Any]:
    """
    Scans all inputs to find a valid array namespace.
    Prioritizes explicit backends (Torch/JAX) over generic lists.
    If no arrays are found, defaults to NumPy.
    """
    xp: ArrayNamespace | None = None
    device = None

    for obj in objs:
        try:
            # Try to identify this specific object's backend
            # Note: This fails for Python lists/floats
            curr_xp = array_namespace(obj)
        except TypeError:
            # If obj is a plain list/float, we skip it and keep looking
            # for a 'real' array in the other arguments.
            continue

        # If we found a backend, capture it
        if xp is None:
            xp = curr_xp
        # Strict check to ensure user didn't mix backends
        elif xp != curr_xp:
            raise TypeError(
                f"Mixed backends detected: "
                f"{type(xp).__name__} vs {type(curr_xp).__name__}"
            )

        # Check device compatibility if possible (e.g. for PyTorch tensors)
        curr_device = getattr(obj, "device", None)
        if curr_device is not None:
            if device is None:
                device = curr_device
            # Strict check to ensure user didn't mix devices
            elif device != curr_device:
                raise ValueError(f"Mixed devices detected: {device} vs {curr_device}")

    if xp is not None:
        return xp, device

    if np is not None:
        return np, "cpu"  # type: ignore

    raise ImportError(
        "Inputs are Python lists/floats, but NumPy is not installed. "
        "Install NumPy or pass a backend-specific tensor. "
        "You can install NumPy with PGPE with the optional [numpy] group."
    ) from None


def setup_randn(
    xp: ArrayNamespace, dtype: Any, device: Any, seed: int | None
) -> Callable[[tuple[int, ...]], Array]:
    """
    Returns a function 'randn(shape)' that generates random numbers
    on the correct backend and device.
    """
    # PyTorch (GPU/CPU)
    if "torch" in xp.__name__:  # type: ignore
        import torch  # noqa: PLC0415

        # Create a separate generator to avoid affecting global state
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)

        def randn_torch(shape: tuple[int, ...]) -> Array:
            return torch.randn(shape, generator=gen, device=device, dtype=dtype)  # type: ignore

        return randn_torch

    if "jax" in xp.__name__:  # type: ignore
        import jax  # noqa: PLC0415

        class JAXStatefulRNG:
            def __init__(self, seed: int | None) -> None:
                # Handle seed: JAX requires an integer, defaults to 0 if None
                _seed = seed if seed is not None else 0
                self.key = jax.random.PRNGKey(_seed)

            def __call__(self, shape: tuple[int, ...]) -> Array:
                # Split the key: one for generating data, one for the next state
                self.key, subkey = jax.random.split(self.key)
                return jax.random.normal(subkey, shape, dtype=dtype)  # type: ignore

        # Return the bound method or callable instance
        return JAXStatefulRNG(seed)

    # NumPy / CuPy / Compliant Backends
    # Most compliant libraries mirror the NumPy random API
    if hasattr(xp, "random") and hasattr(xp.random, "default_rng"):
        rng = xp.random.default_rng(seed)

        def randn_numpy(shape: tuple[int, ...]) -> Array:
            # We cast to ensure dtype/device are correct (e.g. for CuPy)
            return xp.asarray(rng.standard_normal(shape), dtype=dtype, device=device)

        return randn_numpy

    # Fallback (e.g. strict mode or obscure backends)
    import numpy as np  # noqa: PLC0415

    rng_fallback = np.random.default_rng(seed)

    def randn_fallback(shape: tuple[int, ...]) -> Array:
        return xp.asarray(
            rng_fallback.standard_normal(shape), dtype=dtype, device=device
        )

    return randn_fallback
