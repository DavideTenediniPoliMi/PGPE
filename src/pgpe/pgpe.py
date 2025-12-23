from collections.abc import Iterable
from typing import Any, TypedDict, Unpack

import numpy as np

from .optimizers import Adam, Optimizer
from .utils import ensure_positive_float, ensure_positive_int, ensure_vector


class D(TypedDict):
    test: int
    prova: str


class PGPE:
    """
    PGPE algorithm (Policy Gradients with Parameter-based Exploration).

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
        dtype: np.dtype | str = "float32",
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
        **kwargs: Unpack[D],
    ) -> None:
        """
        Args:
            solution_length: Dimension of the parameter vector.
            popsize: Number of samples per generation.
            center_init: Initial mean (scalar or vector).
            stdev_init: Initial standard deviation (scalar or vector).
            seed: Random seed.
            dtype: Numpy dtype.
            center_learning_rate: Initial step size for the center.
            stdev_learning_rate: Learning rate for standard deviation.
            stdev_clip_percent: Max relative change for stdev per step (clip).
            symmetric_sampling: Whether to use symmetric sampling (+/- noise).
            natural_gradient: Whether to use natural gradient updates.
            normalize_fitness: Whether to z-score fitness values using a running average.
            max_generations: Total generations for linear LR decay.
            min_lr_ratio: Final LR as a fraction of initial LR.
            optimizer_class: The class of the optimizer to use.
            optimizer_config: Keyword arguments passed to the optimizer constructor.
        """
        self._dtype = np.dtype(dtype)
        self._length = ensure_positive_int(solution_length, "solution_length")
        self._popsize = ensure_positive_int(popsize, "popsize")
        self._symmetric_sampling = symmetric_sampling
        self._normalize_fitness = normalize_fitness
        self._natural_gradient = natural_gradient

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
        self._stdev_clip_percent = stdev_clip_percent

        # Scheduler Configuration
        self._max_generations = ensure_positive_int(
            max_generations, "max_generations"
        )
        self._lr_range = 1 - ensure_positive_float(
            min_lr_ratio, "min_lr_ratio"
        )
        self._generation_count = 0

        # State Initialization
        self._center = ensure_vector(center_init, self._length, self._dtype)
        self._logstd = np.log(
            ensure_vector(stdev_init, self._length, self._dtype)
        )

        # Optimizer Instantiation
        self._optimizer = optimizer_class(
            solution_length=self._length,
            stepsize=self._initial_center_lr,
            dtype=self._dtype,
            **(optimizer_config or {}),
        )

        # Fitness Normalization Stats
        self._running_mean = 0.0
        self._running_var = 1.0

        # Random State
        self._rndgen = np.random.RandomState(seed)

        # Buffers
        self._noises: np.ndarray | None = None
        self._solutions: np.ndarray | None = None

    @property
    def center(self) -> np.ndarray:
        view = self._center[:]
        view.flags.writeable = False
        return view

    @property
    def stdev(self) -> np.ndarray:
        return np.exp(self._logstd)

    def ask(self) -> np.ndarray:
        """Generates a new population of candidate solutions of size `popsize`."""
        if self._symmetric_sampling:
            num_base = self._popsize // 2
            base_noises = self._rndgen.randn(num_base, self._length).astype(
                self._dtype
            )
            self._noises = np.concatenate([base_noises, -base_noises], axis=0)
        else:
            self._noises = self._rndgen.randn(
                self._popsize, self._length
            ).astype(self._dtype)

        self._solutions = self._center + np.exp(self._logstd) * self._noises
        return self._solutions.copy()

    def tell(self, fitnesses: Iterable[float]) -> None:
        """Updates the internal distribution based on evaluated fitnesses.
        Must be called after `ask()` and with fitnesses corresponding to the
        solutions returned by `ask()`.
        """
        if self._noises is None:
            raise RuntimeError("Called tell() before ask().")

        fitness_arr = np.asarray(fitnesses, dtype=self._dtype)
        if len(fitness_arr) != self._popsize:
            raise ValueError(
                f"Expected {self._popsize} fitness values, got {len(fitness_arr)}"
            )

        # 1. Fold for Symmetric Sampling & Normalize Fitness (if needed)
        base_noises = self._noises
        if self._symmetric_sampling:
            num_pairs = self._popsize // 2
            fitness_pos = fitness_arr[:num_pairs]
            fitness_neg = fitness_arr[num_pairs:]
            fitness_arr = fitness_pos - fitness_neg
            base_noises = self._noises[:num_pairs]

        if self._normalize_fitness:
            self._update_running_stats(fitness_arr)
            fitness_arr = (fitness_arr - self._running_mean) / (
                np.sqrt(self._running_var) + 1e-8
            )

        stdev = np.exp(self._logstd)

        # 2. Compute Gradients
        grad_center: np.ndarray = np.mean(
            fitness_arr[:, None] * base_noises / stdev, axis=0
        )
        grad_log_stdev: np.ndarray = np.mean(
            fitness_arr[:, None] * (base_noises**2 - 1), axis=0
        )

        # 3. Apply Natural Gradient Adjustment
        if self._natural_gradient:
            grad_center *= stdev**2
            grad_log_stdev /= 2

        # 4. Parameter Updates
        self._center += self._optimizer.ascent(grad_center)

        # Stdev update (simple gradient ascent with clipping)
        delta_logstd = self._stdev_lr * grad_log_stdev
        if self._stdev_clip_percent is not None:
            limit = np.log(1.0 + self._stdev_clip_percent)
            delta_logstd = np.clip(delta_logstd, -limit, limit)
        self._logstd += delta_logstd

        # 5. Scheduling
        self._update_learning_rates()

    def _update_running_stats(self, fitness: np.ndarray) -> None:
        self._generation_count += 1
        alpha = 1.0 / self._generation_count
        batch_mean = np.mean(fitness)
        batch_var = float(np.var(fitness))

        self._running_mean += alpha * (batch_mean - self._running_mean)
        self._running_var += alpha * (batch_var - self._running_var)

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
