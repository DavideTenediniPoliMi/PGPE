import numpy as np
import pytest
import torch
import array_api_strict as xp

from pgpe import PGPE

SEED = 42


def test_numpy_explicit_backend():
    """
    Verify that passing NumPy arrays forces the NumPy backend
    and returns NumPy arrays.
    """
    # 1. Initialize with explicit NumPy arrays
    center_init = np.zeros(5, dtype=np.float32)
    pgpe = PGPE(
        solution_length=5,
        popsize=10,
        center_init=center_init,
        seed=SEED
    )

    # 2. Check internal backend detection
    #    Note: array_api_compat often returns the module itself
    assert "numpy" in pgpe._xp.__name__ or pgpe._xp == np
    
    # 3. Check output types
    solutions = pgpe.ask()
    assert isinstance(solutions, np.ndarray)
    assert solutions.dtype == np.float32
    
    # 4. Check tell() with numpy input
    fitness = np.random.rand(10).astype(np.float32)
    pgpe.tell(fitness)
    
    # Center should still be numpy
    assert isinstance(pgpe.center, np.ndarray)


def test_torch_cpu_backend():
    """
    Verify that passing PyTorch tensors forces the Torch backend
    and returns Torch tensors.
    """
    # 1. Initialize with PyTorch tensors (CPU)
    center_init = torch.zeros(5, dtype=torch.float32)
    pgpe = PGPE(
        solution_length=5,
        popsize=10,
        center_init=center_init,
        seed=SEED
    )

    # 2. Check internal backend detection
    #    The backend name should contain 'torch'
    assert 'torch' in pgpe._xp.__name__
    
    # 3. Check output types
    solutions = pgpe.ask()
    assert isinstance(solutions, torch.Tensor)
    assert solutions.dtype == torch.float32
    assert solutions.device.type == 'cpu'
    
    # 4. Check tell() with torch input
    fitness = torch.rand(10, dtype=torch.float32)
    pgpe.tell(fitness)
    
    # Center should still be torch
    assert isinstance(pgpe.center, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_cuda_backend():
    """
    Verify that passing CUDA tensors keeps everything on the GPU.
    """
    device = torch.device('cuda')
    
    # 1. Initialize with CUDA tensor
    center_init = torch.zeros(5, device=device)
    
    pgpe = PGPE(
        solution_length=5,
        popsize=10,
        center_init=center_init,
        seed=SEED
    )

    # 2. Verify internal device matches
    assert pgpe._device == device

    # 3. Check output location
    solutions = pgpe.ask()
    assert isinstance(solutions, torch.Tensor)
    assert solutions.device.type == 'cuda'
    
    # 4. Check optimization step stays on GPU
    fitness = torch.rand(10, device=device)
    pgpe.tell(fitness)
    
    assert pgpe.center.device.type == 'cuda'


def test_mixed_backends_error():
    """
    Test that mixing NumPy and Torch inputs raises a TypeError
    (caught by your robust get_xp implementation).
    """
    with pytest.raises(TypeError, match="Mixed backends"):
        PGPE(
            solution_length=5,
            popsize=10,
            center_init=np.zeros(5),       # NumPy
            stdev_init=torch.zeros(5),      # Torch
            seed=SEED
        )

def test_strict_compliance_loop():
    """
    Runs a full ask/tell loop using the 'array-api-strict' backend.
    If this runs without error, the code is compliant with the standard.
    """
    # 1. Create inputs using the strict backend
    #    This forces PGPE to detect 'xp' as array_api_strict
    center = xp.asarray([0.0, 0.0], dtype=xp.float32)
    
    # 2. Initialize PGPE
    optimizer = PGPE(
        solution_length=2,
        popsize=4,
        center_init=center,
        stdev_init=0.1,  # Scalar is fine, strict backend handles 0-d creation
        center_learning_rate=0.1,
        stdev_learning_rate=0.1,
        max_generations=5,
        dtype=xp.float32, # Pass the actual strict dtype object
        seed=SEED
    )

    # 3. Verify backend detection
    #    Internal check to ensure we aren't secretly using numpy
    assert optimizer._xp.__name__ == "array_api_strict", "Failed to detect strict backend"

    # 4. Run the loop (Coverage)
    #    We must call every method we want to verify.
    for _ in range(3):
        # ASK: Checks xp.concat, xp.randn, xp.exp
        solutions = optimizer.ask()
        
        # Verify output is a strict array
        # Note: In strict mode, isinstance checks might be tricky depending on version,
        # but the object should definitely NOT be numpy.ndarray
        assert type(solutions).__module__ == 'array_api_strict._array_object'
        
        # Fake fitness (must be a strict array too!)
        fitness = xp.asarray([1.0, 0.5, -0.5, -1.0], dtype=xp.float32)
        
        # TELL: Checks xp.mean, xp.sqrt, optimizer steps
        optimizer.tell(fitness)

def test_everything_enabled():
    """
    Runs a PGPE loop with ALL features enabled to maximize code coverage.
    
    Features covered:
    - Symmetric Sampling (if self._symmetric_sampling...)
    - Fitness Normalization (if self._normalize_fitness...)
    - Natural Gradients (if self._natural_gradient...)
    - Stdev Clipping (if self._stdev_clip_percent > 0...)
    - Optimizer steps (Adam)
    """
    # 1. Setup with EVERYTHING enabled
    popsize = 10
    length = 5
    
    pgpe = PGPE(
        solution_length=length,
        popsize=popsize,
        
        # Enable Symmetric Sampling (Ask/Tell logic)
        symmetric_sampling=True,
        
        # Enable Natural Gradients (Fisher info logic)
        natural_gradient=True,
        
        # Enable Fitness Normalization (Running stats logic)
        normalize_fitness=True,
        
        # Enable Clipping (Force a high LR and strict clip to trigger logic)
        stdev_learning_rate=0.5, 
        stdev_clip_percent=0.1,
        
        seed=SEED
    )

    # 2. ASK Step
    # Triggers symmetric noise generation (creating + and - noise)
    solutions = pgpe.ask()
    
    assert solutions.shape == (popsize, length)
    # Check that noises were mirrored internally
    half = popsize // 2
    assert np.allclose(pgpe._noises[:half], -pgpe._noises[half:])

    # 3. TELL Step
    # Create random fitnesses
    fitness = np.random.randn(popsize)
    
    # We want to force clipping logic, so let's make gradients likely large
    # by having diverse fitness values
    fitness[0] = 100.0
    fitness[1] = -100.0
    
    # This single call hits:
    # - Rank folding (symmetric)
    # - Running mean/var updates (normalization)
    # - Fisher matrix adjustment (natural gradient)
    # - Gradient clipping (stdev update)
    pgpe.tell(fitness)

    # 4. Verify State Updated
    # Center should have moved
    assert not np.allclose(pgpe.center, 0.0)
    
    # Stdev should have changed (and ideally been clipped)
    # We can't easily assert "it was clipped" without mocking, 
    # but we covered the lines of code.
    assert not np.allclose(pgpe.stdev, 0.1)
    
    # 5. Verify Running Stats updated
    assert pgpe._generation_count == 1
    # Variance should no longer be exactly 1.0
    assert pgpe._running_var != 1.0

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

def test_jax_strict_device_enforcement():
    """
    Verifies that PGPE strictly enforces device locality:
    1. Crashing during __init__ if inputs are on mixed devices.
    2. Crashing during tell() if fitness is on the wrong device.
    """
    # 2. Safe Import (skips test if JAX is missing)
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed", allow_module_level=True)

    # 3. Verify we actually have 2 devices
    devices = jax.local_devices()
    if len(devices) < 2:
        pytest.skip("Could not emulate multiple devices (check XLA_FLAGS)")

    dev0 = devices[0]
    dev1 = devices[1]

    print(f"Testing with JAX devices: {dev0} vs {dev1}")

    # ---------------------------------------------------------------------
    # SCENARIO A: Mixed Devices in Initialization (Caught by get_xp)
    # ---------------------------------------------------------------------
    center = jax.device_put(jnp.zeros(5), dev0)
    stdev  = jax.device_put(jnp.ones(5) * 0.1, dev1)

    # This MUST fail because get_xp scans both inputs and sees mismatch
    with pytest.raises(ValueError, match="Mixed devices detected"):
        PGPE(
            solution_length=5,
            popsize=10,
            center_init=center,
            stdev_init=stdev
        )

    # ---------------------------------------------------------------------
    # SCENARIO B: Wrong Device in tell() (Caught by tell strict check)
    # ---------------------------------------------------------------------
    # Initialize correctly on dev0
    pgpe = PGPE(
        solution_length=5,
        popsize=10,
        center_init=jax.device_put(jnp.zeros(5), dev0),
        device=dev0
    )

    # Verify initialization landed on dev0
    assert pgpe.center.device == dev0

    # Create fitness on dev1
    fitness_bad = jax.device_put(jnp.zeros(10), dev1)
    pgpe.ask()
    # This MUST fail. Silent copy is strictly forbidden.
    with pytest.raises(ValueError, match="Fitness is on"):
        pgpe.tell(fitness_bad)

    # ---------------------------------------------------------------------
    # SCENARIO C: Correct Usage (Should pass)
    # ---------------------------------------------------------------------
    # ask() should generate noise on dev0
    solutions = pgpe.ask()
    assert solutions.device == dev0

    # Create fitness on dev0
    fitness_good = jax.device_put(jnp.zeros(10), dev0)

    # This should succeed
    pgpe.tell(fitness_good)
    
    # Confirm parameter update happened on dev0
    assert pgpe.center.device == dev0

def test_jax():
    """
    Basic sanity check to ensure JAX backend works without device issues.
    This is a more relaxed test than the strict device enforcement one.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed", allow_module_level=True)

    # Initialize PGPE with JAX arrays
    center = jnp.zeros(5)
    stdev = jnp.ones(5) * 0.1

    pgpe = PGPE(
        solution_length=5,
        popsize=10,
        center_init=center,
        stdev_init=stdev,
        seed=SEED
    )

    # Run a simple ask/tell loop
    for _ in range(3):
        solutions = pgpe.ask()
        fitness = jax.random.normal(jax.random.PRNGKey(SEED), (10,))
        pgpe.tell(fitness)