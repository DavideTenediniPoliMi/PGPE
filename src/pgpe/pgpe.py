from collections.abc import Iterable
from typing import Any

from array_api.latest import Array

from pgpe.utils import get_xp, setup_randn

from .optimizers import Adam, Optimizer
from .utils import ensure_positive_float, ensure_positive_int, ensure_vector


class PGPE:
    """
    PGPE algorithm (Policy Gradients with Parameter-based Exploration).

    This optimizer maintains a Gaussian distribution over the parameter space
    (defined by a mean `center` and standard deviation `stdev`) and optimizes
    these distribution parameters to maximize the expected fitness of the
    samples.

    Key Features:
    - **Backend Agnostic**: Compatible with NumPy, PyTorch, and JAX via the
      Array API standard. Automatically infers backend/device from input data.
    - **Symmetric Sampling**: Supports antithetic variates (mirrored noise) to
      reduce gradient variance.
    - **Natural Gradients**: Optional support for diagonal natural gradient
      updates to improve convergence on ill-conditioned landscapes.
    - **Adaptive Baselines**: Includes running fitness normalization (z-scoring)
      to stabilize gradient magnitudes.
    - **Optimizer Injection**: Uses an inner optimizer (default: Adam) for the
      center updates, allowing for standard momentum/adaptive learning rates.

    Reference:
        Sehnke, Frank, et al. "Parameter-exploring policy gradients."
        Neural Networks 23.4 (2010): 551-559.
    """

    def __init__(
        self,
        *,
        solution_length: int,
        popsize: int,
        # --- Initialization ---
        center_init: float | Iterable[float] = 0.0,
        stdev_init: float | Iterable[float] = 0.1,
        seed: int | None = None,
        dtype: Any = None,
        device: Any = None,
        # --- Hyperparameters ---
        center_learning_rate: float = 0.15,
        stdev_learning_rate: float = 0.1,
        stdev_clip_percent: float = 0.2,
        symmetric_sampling: bool = True,
        natural_gradient: bool = False,
        normalize_fitness: bool = True,
        # --- Scheduling ---
        max_generations: int = 1000,
        min_lr_ratio: float = 0.2,
        # --- Optimizer Injection ---
        optimizer_class: type[Optimizer] = Adam,
        optimizer_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the PGPE optimizer.

        Args:
            solution_length: The dimension of the parameter vector to optimize.
            popsize: The total number of candidate solutions to generate per
                `ask()` call. If `symmetric_sampling` is True, this must be an
                even number (as samples are generated in +/- pairs).
            center_init: Initial value for the mean (center) of the distribution.
                Can be a scalar (broadcast to all dims) or a vector of length
                `solution_length`. Used to infer the backend if not provided.
            stdev_init: Initial standard deviation (exploration noise).
                Can be a scalar or a vector.
            seed: Seed for the random number generator to ensure reproducibility.
            dtype: Explicit datatype (e.g., `np.float32`, `torch.float32`).
                If None, inferred from `center_init` or defaults to float32.
            device: Computational device (e.g., 'cpu', 'cuda', 'mps').
                If None, inferred from `center_init` (e.g., if passing a Torch
                tensor on GPU).
            center_learning_rate: Initial step size for updating the mean
                (center) parameters.
            stdev_learning_rate: Step size for updating the standard deviation.
            stdev_clip_percent: Maximum allowed relative change for the
                standard deviation per update (e.g., 0.2 means max 20% change).
                Prevents explosive or collapsing variance. Must be between 0 and 1.
                If 0, no clipping is applied.
                Default is 0.2 (20%).
            symmetric_sampling: If True, uses antithetic sampling. For every
                noise vector epsilon, the population will include both
                (center + epsilon) and (center - epsilon). This significantly
                reduces gradient variance. If True, `popsize` must be even.
                Default is True.
            natural_gradient: If True, applies a diagonal approximation of the
                Fisher Information Matrix to the gradient updates.
                Default is False (standard gradient).
            normalize_fitness: If True, fitness values are z-scored using a
                cumulative running mean and variance before gradient computation.
                Recommended to handle shifting fitness landscapes. Default is True.
            max_generations: Total number of generations (steps) used for the
                linear learning rate decay scheduler.
            min_lr_ratio: The minimum fraction of the initial learning rate to
                decay to at `max_generations` (e.g., 0.2 means decay to 20%).
                Must be between 0 and 1. Default is 0.2 (20%).
                If 1, no decay is applied.
            optimizer_class: The class of the inner optimizer used to update the
                center parameters (e.g., `Adam`). Must adhere to the `Optimizer`
                protocol.
            optimizer_config: Dictionary of keyword arguments passed to the
                `optimizer_class` constructor.
        """
        self._length = ensure_positive_int(solution_length, "solution_length")
        self._popsize = ensure_positive_int(popsize, "popsize")
        self._symmetric_sampling = symmetric_sampling
        self._normalize_fitness = normalize_fitness
        self._natural_gradient = natural_gradient

        # Backend Detection
        self._xp, found_device = get_xp(center_init, stdev_init)
        self._device = device or found_device
        self._dtype = dtype or self._xp.float32

        self._randn = setup_randn(
            xp=self._xp, dtype=self._dtype, device=self._device, seed=seed
        )

        if self._symmetric_sampling and self._popsize % 2 != 0:
            raise ValueError("For symmetric sampling, popsize must be even.")

        # Learning Rate Configuration
        self._initial_center_lr = ensure_positive_float(
            center_learning_rate, "center_learning_rate"
        )
        self._initial_stdev_lr = ensure_positive_float(
            stdev_learning_rate, "stdev_learning_rate"
        )
        self._stdev_lr = self._initial_stdev_lr
        if not (0 <= stdev_clip_percent < 1):
            raise ValueError("stdev_clip_percent must be in the range [0, 1).")
        self._stdev_clip_percent = stdev_clip_percent

        # Scheduler Configuration
        self._max_generations = ensure_positive_int(max_generations, "max_generations")
        self._lr_range = 1 - ensure_positive_float(min_lr_ratio, "min_lr_ratio")
        self._generation_count = 0

        # State Initialization
        self._center = ensure_vector(
            center_init,
            self._length,
            xp=self._xp,
            dtype=self._dtype,
            device=self._device,
        )
        self._logstd = self._xp.log(
            ensure_vector(
                stdev_init,
                self._length,
                xp=self._xp,
                dtype=self._dtype,
                device=self._device,
            )
        )

        # Optimizer Instantiation
        self._optimizer = optimizer_class(
            dim=self._length,
            stepsize=self._initial_center_lr,
            xp=self._xp,
            dtype=self._dtype,
            device=self._device,
            **(optimizer_config or {}),
        )

        # Fitness Normalization Stats
        self._running_mean = self._xp.asarray(
            0.0, dtype=self._dtype, device=self._device
        )
        self._running_var = self._xp.asarray(
            1.0, dtype=self._dtype, device=self._device
        )

        self._noises: Array | None = None

    @property
    def center(self) -> Array:
        return self._xp.asarray(self._center, copy=True)

    @property
    def stdev(self) -> Array:
        return self._xp.exp(self._logstd)

    def ask(self) -> Array:
        """Generates a new population of candidate solutions of size `popsize`."""
        if self._symmetric_sampling:
            num_base = self._popsize // 2
            base_noises = self._randn((num_base, self._length))
            self._noises = self._xp.concat([base_noises, -base_noises], axis=0)
        else:
            self._noises = self._randn((self._popsize, self._length))

        return self._center + self._xp.exp(self._logstd) * self._noises

    def tell(self, fitnesses: Array) -> None:
        """Updates the internal distribution based on evaluated fitnesses.
        Must be called after `ask()` and with fitnesses corresponding to the
        solutions returned by `ask()`.
        The array must match the device/dtype of the internal state.
        """
        if self._noises is None:
            raise RuntimeError("Called tell() before ask().")

        if hasattr(fitnesses, "device") and fitnesses.device != self._device:
            raise ValueError(
                f"Fitness is on {fitnesses.device}, but PGPE is on {self._device}."
            )

        fitness_arr = self._xp.asarray(fitnesses, device=self._device)

        if fitness_arr.ndim > 2 or (
            fitness_arr.ndim == 2 and fitness_arr.shape[1] != 1
        ):
            raise ValueError(
                f"Fitness must be 1D or column vector, got {fitness_arr.shape}"
            )

        fitness_arr = self._xp.reshape(fitness_arr, (-1, 1))
        if fitness_arr.shape[0] != self._popsize:
            raise ValueError(
                f"Expected {self._popsize} fitness values, got {fitness_arr.shape[0]}"
            )

        # 1. Fold for Symmetric Sampling & Normalize Fitness (if needed)
        base_noises = self._noises
        if self._symmetric_sampling:
            num_pairs = self._popsize // 2
            fitness_pos = fitness_arr[:num_pairs, :]
            fitness_neg = fitness_arr[num_pairs:, :]
            fitness_arr = fitness_pos - fitness_neg
            base_noises = self._noises[:num_pairs, :]

        if self._normalize_fitness:
            self._update_running_stats(fitness_arr)
            fitness_arr = (fitness_arr - self._running_mean) / (
                self._xp.sqrt(self._running_var) + 1e-8
            )

        # 2. Compute Gradients
        stdev = self._xp.exp(self._logstd)

        grad_center = self._xp.mean(fitness_arr * base_noises, axis=0) / stdev
        grad_log_stdev = self._xp.mean(fitness_arr * (base_noises**2 - 1), axis=0)

        # 3. Apply Natural Gradient Adjustment
        if self._natural_gradient:
            grad_center = grad_center * stdev**2
            grad_log_stdev = grad_log_stdev / 2

        # 4. Parameter Updates
        self._center = self._center + self._optimizer.ascent(grad_center)

        # Stdev update (simple gradient ascent with clipping)
        delta_logstd = self._stdev_lr * grad_log_stdev
        if self._stdev_clip_percent > 0:
            limit_arr = self._xp.asarray(
                1.0 + self._stdev_clip_percent, dtype=self._dtype, device=self._device
            )
            limit_arr = self._xp.log(limit_arr)
            delta_logstd = self._xp.clip(delta_logstd, -limit_arr, limit_arr)
        self._logstd += delta_logstd

        # 5. Scheduling
        self._update_learning_rates()

    def _update_running_stats(self, fitness: Array) -> None:
        self._generation_count += 1
        alpha = 1.0 / self._generation_count

        batch_mean = self._xp.mean(fitness)
        batch_var = self._xp.var(fitness)

        self._running_mean = self._running_mean + alpha * (
            batch_mean - self._running_mean
        )
        self._running_var = self._running_var + alpha * (batch_var - self._running_var)

    def _update_learning_rates(self) -> None:
        if self._generation_count >= self._max_generations:
            return

        # Linear decay from 1.0 down to min_lr_ratio
        progress = self._generation_count / self._max_generations
        multiplier = 1.0 - progress * self._lr_range

        # Update Center LR
        self._optimizer.stepsize = self._initial_center_lr * multiplier

        # Update Stdev LR
        self._stdev_lr = self._initial_stdev_lr * multiplier

    def state_dict(self) -> dict[str, Any]:
        """Returns a dictionary containing the optimizer state."""
        return {
            "center": self._center,
            "logstd": self._logstd,
            "running_mean": self._running_mean,
            "running_var": self._running_var,
            "generation_count": self._generation_count,
            "optimizer_state": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores the optimizer state."""

        # We must cast the loaded state back to the correct backend/device
        def _load(value: Any) -> Array:
            return self._xp.asarray(
                value, dtype=self._dtype, device=self._device, copy=True
            )

        self._center = _load(state["center"])
        self._logstd = _load(state["logstd"])
        self._running_mean = _load(state["running_mean"])
        self._running_var = _load(state["running_var"])
        self._generation_count = state["generation_count"]

        # Restore Optimizer
        opt_state = state["optimizer_state"]
        self._optimizer.load_state_dict(opt_state)

        # Re-sync learning rates in case generation count changed
        self._update_learning_rates()
