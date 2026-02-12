from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .utils import ensure_positive_float, ensure_positive_int

if TYPE_CHECKING:
    from array_api.latest import Array, ArrayNamespace
# ==========================================================================
# The following section of this source file contains optimizer classes
# copied and adapted from OpenAI's evolution-strategies-starter repository.

# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

# evolution-strategies-starter license:
#
# The MIT License
#
# Copyright (c) 2016 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Code copied and adapted from OpenAI's evolution-strategies-starter begins
# here:


class Optimizer(ABC):
    """
    Protocol defining the interface for a gradient-based optimizer.
    """

    def __init__(
        self,
        *,
        dim: int,
        stepsize: float,
        xp: ArrayNamespace,
        dtype: Any,
        device: Any,
    ) -> None:
        self.dim = ensure_positive_int(dim, "dim")
        self.stepsize = ensure_positive_float(stepsize, "stepsize")
        self.xp = xp
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def ascent(self, globalg: Array) -> Array:
        """
        Performs a gradient ascent step.

        Args:
            globalg: The gradient vector (1D array).

        Returns:
            The update step vector (1D array) to be added to the parameters.
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Returns a dictionary containing the optimizer state."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Loads the optimizer state from a checkpoint dictionary."""
        ...


class Adam(Optimizer):
    """
    Adam optimizer implementation.
    """

    def __init__(
        self,
        *,
        dim: int,
        stepsize: float,
        xp: ArrayNamespace,
        dtype: Any = None,
        device: Any = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(
            dim=dim, stepsize=stepsize, xp=xp, dtype=dtype or xp.float32, device=device
        )

        self.beta1 = ensure_positive_float(beta1, "beta1")
        self.beta2 = ensure_positive_float(beta2, "beta2")
        self.epsilon = ensure_positive_float(epsilon, "epsilon")

        self.t = 0
        self.m = self.xp.zeros(self.dim, dtype=self.dtype, device=self.device)
        self.v = self.xp.zeros(self.dim, dtype=self.dtype, device=self.device)

    def ascent(self, globalg: Array) -> Array:
        g = self.xp.asarray(globalg, dtype=self.dtype, device=self.device)
        if g.shape != (self.dim,):
            raise ValueError(
                f"Gradient shape mismatch. Expected ({self.dim},), got {g.shape}"
            )

        self.t += 1

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)

        # Bias correction
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        # Compute the update step
        return self.stepsize * m_hat / (self.xp.sqrt(v_hat) + self.epsilon)

    def state_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "m": self.m,
            "v": self.v,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.t = state["t"]
        self.m = self.xp.asarray(
            state["m"], dtype=self.dtype, device=self.device, copy=True
        )
        self.v = self.xp.asarray(
            state["v"], dtype=self.dtype, device=self.device, copy=True
        )


# Code copied and adapted from OpenAI's evolution-strategies-starter ends here.
# ==========================================================================
