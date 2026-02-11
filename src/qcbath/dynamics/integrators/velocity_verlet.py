from __future__ import annotations

from dataclasses import dataclass

from qcbath.dynamics.eom import PBMEEOM
from qcbath.model.hamiltonian import FullParams, FullState


@dataclass
class EulerIntegrator:
    """Simple explicit Euler skeleton (placeholder for symplectic splitting)."""

    eom: PBMEEOM

    def step(self, state: FullState, dt: float, params: FullParams) -> FullState:
        deriv = self.eom.derivatives(state, params)
        out = state.copy()
        out.x += dt * deriv.x
        out.p += dt * deriv.p
        out.r += dt * deriv.r
        out.p_r += dt * deriv.p_r
        out.t += dt
        return out
