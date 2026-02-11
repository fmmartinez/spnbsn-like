from __future__ import annotations

import numpy as np


def pure_state_mapping(n_states: int, occupied_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simple deterministic mapping initialization scaffold."""
    x = np.zeros(n_states)
    p = np.zeros(n_states)
    x[occupied_index] = 1.0
    return x, p
