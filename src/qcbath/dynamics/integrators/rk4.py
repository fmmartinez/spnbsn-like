from __future__ import annotations

from dataclasses import dataclass

from qcbath.dynamics.eom import PBMEEOM
from qcbath.model.hamiltonian import FullParams, FullState


@dataclass
class RK4Integrator:
    eom: PBMEEOM

    def step(self, state: FullState, dt: float, params: FullParams) -> FullState:
        k1 = self.eom.derivatives(state, params)
        k2 = self.eom.derivatives(_add_scaled(state, k1, 0.5 * dt), params)
        k3 = self.eom.derivatives(_add_scaled(state, k2, 0.5 * dt), params)
        k4 = self.eom.derivatives(_add_scaled(state, k3, dt), params)

        out = state.copy()
        out.x += (dt / 6.0) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x)
        out.p += (dt / 6.0) * (k1.p + 2 * k2.p + 2 * k3.p + k4.p)
        out.r += (dt / 6.0) * (k1.r + 2 * k2.r + 2 * k3.r + k4.r)
        out.p_r += (dt / 6.0) * (k1.p_r + 2 * k2.p_r + 2 * k3.p_r + k4.p_r)
        out.t += dt
        return out


def _add_scaled(a: FullState, b: FullState, scale: float) -> FullState:
    out = a.copy()
    out.x += scale * b.x
    out.p += scale * b.p
    out.r += scale * b.r
    out.p_r += scale * b.p_r
    out.t += scale * b.t
    return out
