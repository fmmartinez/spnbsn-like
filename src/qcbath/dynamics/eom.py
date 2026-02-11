from __future__ import annotations

import numpy as np

from qcbath.model.hamiltonian import FullParams, FullState


class PBMEEOM:
    """Equations of motion scaffold for PBME-like mixed dynamics.

    Replace/extend these derivatives with your preferred PBME convention.
    """

    def derivatives(self, state: FullState, params: FullParams) -> FullState:
        h = params.subsystem.subsystem_hamiltonian()

        # Mapping variable equations (harmonic-like scaffold)
        dx = h @ state.p
        dp = -(h @ state.x)

        # Classical bath equations
        dr = np.zeros_like(state.r)
        dp_r = np.zeros_like(state.p_r)

        sigma_z_like = 0.0
        if len(state.x) >= 2:
            sigma_z_like = 0.5 * ((state.x[0] ** 2 + state.p[0] ** 2) - (state.x[1] ** 2 + state.p[1] ** 2))

        for j, mode in enumerate(params.bath.modes):
            dr[j] = state.p_r[j] / mode.mass
            dp_r[j] = -mode.mass * mode.omega**2 * state.r[j] - mode.coupling * sigma_z_like

        return FullState(x=dx, p=dp, r=dr, p_r=dp_r, t=1.0)
