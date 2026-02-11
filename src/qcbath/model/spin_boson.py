from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinBosonParams:
    """Two-level spin-boson subsystem parameters."""

    epsilon: float
    delta: float

    def subsystem_hamiltonian(self) -> np.ndarray:
        return np.array(
            [[0.5 * self.epsilon, self.delta], [self.delta, -0.5 * self.epsilon]],
            dtype=float,
        )
