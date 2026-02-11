from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MappingState:
    """Mapping variables for N electronic states."""

    x: np.ndarray
    p: np.ndarray

    def copy(self) -> "MappingState":
        return MappingState(self.x.copy(), self.p.copy())
