import numpy as np
import pytest
from pgpe import PGPE

# Constants for testing
SEED = 42
DIM = 5
POP = 10


def simple_fitness(solutions):
    """Deterministic fitness function (Sphere)."""
    # Works for numpy, torch, jax if they support basic operators
    return -(solutions**2).sum(axis=1)


def run_steps(optimizer, steps, fitness_fn):
    """Runs the optimizer for a number of steps."""
    for _ in range(steps):
        solutions = optimizer.ask()
        fitness = fitness_fn(solutions)
        optimizer.tell(fitness)


def verify_resume_determinism(xp_backend, device, dtype_name):
    """
    Generic verification logic.
    1. Train 'A' for 10 steps. Save at step 5.
    2. Resume 'B' from step 5.
    3. Assert A and B match at step 10.
    """
    # --- Setup Backend specific args ---
    if xp_backend == "numpy":
        dtype = np.float32
        center_init = np.zeros(DIM, dtype=dtype)
    elif xp_backend == "torch":
        import torch

        dtype = getattr(torch, dtype_name)
        center_init = torch.zeros(DIM, dtype=dtype, device=device)
    elif xp_backend == "jax":
        import jax.numpy as jnp

        dtype = getattr(jnp, dtype_name)
        # JAX device handling is implicit or via put, typically cpu for tests
        center_init = jnp.zeros(DIM, dtype=dtype)
    else:
        raise ValueError(f"Unknown backend {xp_backend}")

    # --- Run A (The "Ground Truth") ---
    pgpe_a = PGPE(
        solution_length=DIM,
        popsize=POP,
        center_init=center_init,
        seed=SEED,
        center_learning_rate=0.1,
    )

    # Run first 5 steps
    run_steps(pgpe_a, 5, simple_fitness)

    # SAVE STATE
    checkpoint = pgpe_a.state_dict()

    # Run 5 more steps (Total 10)
    run_steps(pgpe_a, 5, simple_fitness)
    final_center_a = pgpe_a.center

    # --- Run B (The "Resumed" Run) ---
    # Initialize with DIFFERENT seed to ensure we aren't just getting lucky
    pgpe_b = PGPE(
        solution_length=DIM,
        popsize=POP,
        center_init=center_init,  # Same shape/device
        seed=SEED + 999,
        center_learning_rate=0.1,
    )

    # LOAD STATE (Should overwrite the internal RNG state)
    pgpe_b.load_state_dict(checkpoint)

    # Run 5 steps (Should replicate steps 6-10 of A)
    run_steps(pgpe_b, 5, simple_fitness)
    final_center_b = pgpe_b.center

    # --- Robust Assertions ---
    # We cast everything to NumPy for the final check.
    # This avoids "DeviceArray" boolean issues in JAX and keeps the error messages readable.

    if hasattr(final_center_a, "cpu"):
        # Torch / generic
        val_a = final_center_a.cpu().numpy()
        val_b = final_center_b.cpu().numpy()
    elif hasattr(final_center_a, "__array__"):
        # JAX / NumPy
        val_a = np.array(final_center_a)
        val_b = np.array(final_center_b)
    else:
        # Fallback
        val_a = final_center_a
        val_b = final_center_b

    # This function gives a nice diff report on failure
    np.testing.assert_allclose(
        val_a, val_b, rtol=1e-5, err_msg="Resumed run did not match continuous run!"
    )


# =============================================================================
# 1. NumPy Test
# =============================================================================
def test_checkpointing_numpy_deterministic():
    verify_resume_determinism("numpy", "cpu", "float32")


# =============================================================================
# 2. PyTorch Test
# =============================================================================
def test_checkpointing_torch_deterministic():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    # CPU
    verify_resume_determinism("torch", "cpu", "float32")

    # CUDA (Optional)
    if torch.cuda.is_available():
        verify_resume_determinism("torch", "cuda", "float32")


# =============================================================================
# 3. JAX Test
# =============================================================================
def test_checkpointing_jax_deterministic():
    try:
        import jax
    except ImportError:
        pytest.skip("JAX not installed")

    verify_resume_determinism("jax", "cpu", "float32")
