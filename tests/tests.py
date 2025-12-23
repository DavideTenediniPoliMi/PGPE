import numpy as np
import pytest

from pgpe import PGPE, Adam

# Fix randomness for reproducibility in tests
SEED = 42


def test_initialization():
    """Test that PGPE initializes with correct shapes and defaults."""
    pgpe = PGPE(solution_length=10, popsize=20, seed=SEED)

    assert pgpe.center.shape == (10,)
    assert pgpe.stdev.shape == (10,)
    assert pgpe.center.dtype == np.float32
    # Check default center is 0.0
    assert np.allclose(pgpe.center, 0.0)
    # Check default stdev is 0.1
    assert np.allclose(pgpe.stdev, 0.1)


def test_initialization_errors():
    """Test that invalid arguments raise appropriate errors."""
    with pytest.raises(ValueError, match="popsize must be even"):
        PGPE(solution_length=5, popsize=11, symmetric_sampling=True)

    with pytest.raises(ValueError, match="Expected positive integer"):
        PGPE(solution_length=-5, popsize=10)


def test_ask_shapes_and_symmetry():
    """Test that ask() returns correct shapes and respects symmetric sampling."""
    popsize = 20
    length = 5
    pgpe = PGPE(
        solution_length=length,
        popsize=popsize,
        symmetric_sampling=True,
        seed=SEED,
    )

    solutions = pgpe.ask()

    assert solutions.shape == (popsize, length)

    # Check symmetry: for every x, there should be a corresponding mirrored sample
    # relative to the center. Since center starts at 0, x_i = -x_{i+half}
    half = popsize // 2
    # We need to access the internal noises to verify symmetry perfectly,
    # but we can also check the solutions relative to center.
    # Note: center is 0, so solutions = noise * sigma.
    # pos_half = solutions[:half]
    # neg_half = solutions[half:]
    # assert np.allclose(pos_half, -neg_half)
    # NOTE: The implementation generates center + sigma * noise.
    # noise is concatenated [base, -base].

    # Let's verify via internal logic
    assert pgpe._noises is not None
    base = pgpe._noises[:half]
    mirror = pgpe._noises[half:]
    assert np.allclose(base, -mirror)


def test_ask_tell_loop_reinforce():
    """Test a full step of ask and tell using REINFORCE."""
    pgpe = PGPE(
        solution_length=2,
        popsize=10,
        center_learning_rate=0.1,
        stdev_learning_rate=0.1,
        natural_gradient=False,
        seed=SEED,
    )

    initial_center = pgpe.center.copy()
    initial_stdev = pgpe.stdev.copy()

    solutions = pgpe.ask()
    # Fake fitness: maximize x[0]
    fitness = solutions[:, 0]

    pgpe.tell(fitness)

    # Center should move in positive direction of x[0]
    assert pgpe.center[0] > initial_center[0]
    # Stdev should change
    assert not np.allclose(pgpe.stdev, initial_stdev)


def test_natural_gradients():
    """Test that natural gradients run without error."""
    pgpe = PGPE(solution_length=5, popsize=10, natural_gradient=True, seed=SEED)
    pgpe.ask()
    pgpe.tell(np.random.rand(10))
    # Just checking for math errors/shapes
    assert pgpe.center.shape == (5,)


def test_optimizer_injection():
    """Test that we can inject a custom optimizer configuration."""
    # Pass a custom epsilon to Adam
    pgpe = PGPE(
        solution_length=5,
        popsize=10,
        optimizer_class=Adam,
        optimizer_config={"epsilon": 1.0, "beta1": 0.5},
    )

    assert pgpe._optimizer.epsilon == 1.0
    assert pgpe._optimizer.beta1 == 0.5


def test_tell_before_ask_error():
    """Test that calling tell before ask raises RuntimeError."""
    pgpe = PGPE(solution_length=5, popsize=10)
    with pytest.raises(RuntimeError, match="Called tell.*before ask"):
        pgpe.tell([1.0] * 10)


def test_mismatched_fitness_length():
    """Test that passing wrong number of fitnesses raises ValueError."""
    pgpe = PGPE(solution_length=5, popsize=10)
    pgpe.ask()
    with pytest.raises(ValueError, match="Expected 10 fitness values"):
        pgpe.tell([1.0] * 5)


def test_learning_rate_decay():
    """Test that learning rate decays correctly over generations."""
    max_gens = 100
    min_lr_ratio = 0.1
    initial_lr = 0.5

    pgpe = PGPE(
        solution_length=5,
        popsize=10,
        center_learning_rate=initial_lr,
        max_generations=max_gens,
        min_lr_ratio=min_lr_ratio,
        seed=SEED,
    )

    for gen in range(max_gens):
        pgpe.ask()

        expected_multiplier = 1.0 - (gen / max_gens) * (1.0 - min_lr_ratio)
        expected_lr = initial_lr * expected_multiplier
        assert np.isclose(pgpe._optimizer.stepsize, expected_lr), (
            f"Generation {gen}: expected {expected_lr}, got {pgpe._optimizer.stepsize}"
        )

        pgpe.tell(np.random.rand(10))


def test_fit_on_simple_function():
    """Test PGPE optimization on a simple quadratic function."""

    def fitness_function(x):
        return -np.sum((x - 3.0) ** 2)

    pgpe = PGPE(
        solution_length=2,
        popsize=20,
        center_learning_rate=0.2,
        stdev_learning_rate=0.1,
        seed=SEED,
    )

    for _ in range(50):
        solutions = pgpe.ask()
        fitnesses = [fitness_function(sol) for sol in solutions]
        pgpe.tell(fitnesses)

    # After optimization, center should be close to [3.0, 3.0]
    assert np.allclose(pgpe.center, 3.0, atol=0.5)
