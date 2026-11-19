
import numpy as np


def generate_bits(num_bits: int, seed: int = None) -> np.ndarray:
    """Generate a random binary array of 0s and 1s."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=num_bits, dtype=np.int8)

