"""Context manager for temporary random seed management."""

import numpy as np
from typing import Optional


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
