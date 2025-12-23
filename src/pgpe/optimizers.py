from typing import Any, Protocol, runtime_checkable

import numpy as np

from .utils import ensure_positive_float, ensure_positive_int

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


@runtime_checkable
class Optimizer(Protocol):
    """
    Protocol defining the interface for a gradient-based optimizer.
    """

    stepsize: float

    def __init__(
        self,
        *,
        dim: int,
        dtype: np.dtype,
        stepsize: float,
        **kwargs: Any,
    ) -> None: ...

    def ascent(self, globalg: np.ndarray) -> np.ndarray:
        """
        Performs a gradient ascent step.

        Args:
            globalg: The gradient vector (1D array).

        Returns:
            The update step vector (1D array) to be added to the parameters.
        """
        ...


class Adam:
    """
    Adam optimizer implementation.

    Reference:
        Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(
        self,
        *,
        dim: int,
        stepsize: float,
        dtype: np.dtype,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.dim = ensure_positive_int(dim, "dim")
        self.dtype = dtype
        self.stepsize = ensure_positive_float(stepsize, "stepsize")
        self.beta1 = ensure_positive_float(beta1, "beta1")
        self.beta2 = ensure_positive_float(beta2, "beta2")
        self.epsilon = ensure_positive_float(epsilon, "epsilon")

        self.t = 0
        self.m = np.zeros(self.dim, dtype=self.dtype)
        self.v = np.zeros(self.dim, dtype=self.dtype)

    def ascent(self, globalg: np.ndarray) -> np.ndarray:
        g = np.asarray(globalg, dtype=self.dtype)
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
        return self.stepsize * m_hat / (np.sqrt(v_hat) + self.epsilon)


# Code copied and adapted from OpenAI's evolution-strategies-starter ends here.
# ==========================================================================
