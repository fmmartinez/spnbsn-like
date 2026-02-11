from __future__ import annotations

from typing import Protocol

from qcbath.model.hamiltonian import FullParams, FullState


class Integrator(Protocol):
    def step(self, state: FullState, dt: float, params: FullParams) -> FullState:
        ...
