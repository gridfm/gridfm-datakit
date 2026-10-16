"""Context manager for temporary random seed management."""

import secrets
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class _SeedPolicy:
    """Define and validate every random seed derived by data generation."""

    min_seed: int = 0
    max_seed: int = 2**32 - 1
    auto_seed_upper_bound: int = 50_000
    distributed_stride: int = 20_000

    def random_base_seed(self) -> int:
        """Return a random base seed from the configured half-open range."""
        return secrets.randbelow(self.auto_seed_upper_bound)

    def sequential_seed(self, base_seed: int) -> int:
        """Return the independent perturbation seed for sequential generation."""
        self._require_supported(base_seed)
        return self._require_supported(base_seed + 1)

    def distributed_seed(self, base_seed: int, scenario_offset: int) -> int:
        """Return the deterministic seed for a distributed scenario chunk."""
        return self._require_supported(
            self._distributed_seed_value(base_seed, scenario_offset),
        )

    def maximum_distributed_seed(
        self,
        base_seed: Optional[int],
        scenario_count: int,
    ) -> int:
        """Return the largest possible chunk seed for a generation run."""
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        effective_seed = (
            self.auto_seed_upper_bound - 1 if base_seed is None else base_seed
        )
        return self._distributed_seed_value(effective_seed, scenario_count - 1)

    def _distributed_seed_value(self, base_seed: int, scenario_offset: int) -> int:
        """Calculate a chunk seed without applying the derived upper bound."""
        self._require_supported(base_seed)
        if scenario_offset < 0:
            raise ValueError("scenario_offset must be non-negative")
        return base_seed * self.distributed_stride + scenario_offset + 1

    def _require_supported(self, seed: int) -> int:
        """Return a seed if NumPy supports it, otherwise raise a clear error."""
        if not self.min_seed <= seed <= self.max_seed:
            raise ValueError(
                f"Derived random seed {seed} must be between "
                f"{self.min_seed} and {self.max_seed}",
            )
        return seed


_DEFAULT_SEED_POLICY = _SeedPolicy()


class custom_seed:
    """Context manager to temporarily set a custom random seed.

    This context manager saves the current numpy random state, sets a new seed,
    and restores the previous state upon exit. This is useful for ensuring
    reproducibility in specific code blocks while maintaining the overall
    random state flow.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> print(np.random.rand())  # Will use seed 42
        >>> with custom_seed(100):
        ...     print(np.random.rand())  # Will use seed 100
        >>> print(np.random.rand())  # Will continue from seed 42's sequence

    Args:
        seed: The seed value to use within the context. If None, no seed is set.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the context manager with a custom seed.

        Args:
            seed: The seed value to use. If None, state is saved but no new seed is set.
        """
        self.seed = seed
        self.saved_state = None

    def __enter__(self):
        """Save current random state and set the custom seed."""
        self.saved_state = np.random.get_state()
        if self.seed is not None:
            np.random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the previously saved random state."""
        if self.saved_state is not None:
            np.random.set_state(self.saved_state)
        return False  # Don't suppress exceptions
