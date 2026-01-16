"""Test suite for custom_seed context manager."""

import numpy as np
import pytest
from gridfm_datakit.utils.random_seed import custom_seed


def test_seed_set_inside_with_block():
    """Verify that the seed is properly set inside the with block."""
    # Get expected value with seed=100
    np.random.seed(100)
    expected_value = np.random.rand()

    # Now test with context manager from a different seed
    np.random.seed(42)
    with custom_seed(100):
        actual_value = np.random.rand()

    assert np.isclose(actual_value, expected_value), (
        "Seed not set correctly inside with block"
    )


def test_state_restored_after_with_block():
    """Verify that the random state is restored after exiting the with block."""
    # Set up: seed=42, consume one value
    np.random.seed(42)
    _ = np.random.rand()  # Skip first value

    # Use different seed temporarily
    with custom_seed(999):
        _ = np.random.rand()  # Use seed=999 inside

    # After exiting, should continue from seed=42 sequence (2nd value)
    second_value = np.random.rand()

    # Verify it matches expected 2nd value from seed=42 sequence
    np.random.seed(42)
    _ = np.random.rand()  # skip first
    expected_second = np.random.rand()

    assert np.isclose(second_value, expected_second), (
        "State not restored correctly after exiting with block"
    )


def test_reproducibility_with_same_seed():
    """Verify that using the same seed produces identical results."""

    def generate_data_with_seed(seed):
        with custom_seed(seed):
            return np.random.randn(10)

    # Generate data twice with same seed but different outer states
    np.random.seed(999)
    data1 = generate_data_with_seed(100)

    np.random.seed(777)
    data2 = generate_data_with_seed(100)

    assert np.allclose(data1, data2), "Same seed didn't produce identical results"


def test_none_seed_preserves_state():
    """Verify that passing None as seed just saves/restores state without setting a new seed."""
    np.random.seed(42)
    values_before = [np.random.rand() for _ in range(3)]

    np.random.seed(42)
    values_with_none = []
    with custom_seed(None):
        values_with_none = [np.random.rand() for _ in range(3)]

    assert np.allclose(values_before, values_with_none), (
        "None seed should not change the random sequence"
    )


def test_exception_still_restores_state():
    """Verify that state is restored even if an exception occurs inside the with block."""
    np.random.seed(42)
    _ = np.random.rand()  # Skip first value

    try:
        with custom_seed(999):
            _ = np.random.rand()
            raise ValueError("Test exception")
    except ValueError:
        pass

    # State should still be restored
    value_after = np.random.rand()

    # Verify it matches expected sequence
    np.random.seed(42)
    _ = np.random.rand()  # skip first
    expected_value = np.random.rand()

    assert np.isclose(value_after, expected_value), (
        "State not restored after exception in with block"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
