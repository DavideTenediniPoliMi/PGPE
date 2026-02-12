from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from array_api_compat import array_namespace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from array_api.latest import Array, ArrayNamespace

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


#### RNG Setup Utility ####


class RNGProtocol(Protocol):
    def __call__(self, shape: tuple[int, ...]) -> Array: ...
    def get_state(self) -> Any: ...
    def set_state(self, state: Any) -> None: ...


class NumpyRNG:
    def __init__(
        self, xp: ArrayNamespace, dtype: Any, device: Any, seed: int | None
    ) -> None:
        self.xp = xp
        self.dtype = dtype
        self.device = device

        # Prefer backend-specific rng if available, else fallback to numpy
        if hasattr(xp, "random") and hasattr(xp.random, "default_rng"):
            self.rng = xp.random.default_rng(seed)
        else:
            import numpy as np  # noqa: PLC0415

            self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> Array:
        # Generate on CPU (NumPy) then cast/move to device
        data = self.rng.standard_normal(shape)
        return self.xp.asarray(data, dtype=self.dtype, device=self.device)

    def get_state(self) -> Any:
        return self.rng.bit_generator.state

    def set_state(self, state: Any) -> None:
        self.rng.bit_generator.state = state


def setup_randn(  # noqa: C901
    xp: ArrayNamespace, dtype: Any, device: Any, seed: int | None
) -> RNGProtocol:
    """
    Returns an RNG object that generates random numbers on the correct backend.
    The object is callable: rng(shape) -> Array.
    It also provides .get_state() and .set_state(state) for checkpointing.
    """

    # --- PyTorch Backend ---
    if "torch" in xp.__name__:  # type: ignore
        import torch  # noqa: PLC0415

        class TorchRNG:
            def __init__(self, device: Any, dtype: Any, seed: int | None) -> None:
                self.gen = torch.Generator(device=device)
                if seed is not None:
                    self.gen.manual_seed(seed)
                self.device = device
                self.dtype = dtype

            def __call__(self, shape: tuple[int, ...]) -> Array:
                return torch.randn(  # type: ignore
                    shape, generator=self.gen, device=self.device, dtype=self.dtype
                )

            def get_state(self) -> Any:
                return self.gen.get_state()

            def set_state(self, state: Any) -> None:
                self.gen.set_state(state)

        return TorchRNG(device, dtype, seed)

    # --- JAX Backend ---
    if "jax" in xp.__name__:  # type: ignore
        import jax  # noqa: PLC0415

        class JAXRNG:
            def __init__(self, seed: int | None) -> None:
                _seed = seed if seed is not None else 0
                self.key = jax.random.PRNGKey(_seed)

            def __call__(self, shape: tuple[int, ...]) -> Array:
                self.key, subkey = jax.random.split(self.key)
                return jax.random.normal(subkey, shape, dtype=dtype)  # type: ignore

            def get_state(self) -> Any:
                return self.key

            def set_state(self, state: Any) -> None:
                self.key = state

        return JAXRNG(seed)

    # --- NumPy / Default Backend ---
    return NumpyRNG(xp, dtype, device, seed)
