from __future__ import annotations

import numpy as np

from qcbath.dynamics.eom import PBMEEOM
from qcbath.dynamics.integrators.rk4 import RK4Integrator
from qcbath.model.bath import BathMode, BathParams
from qcbath.model.hamiltonian import FullParams, FullState
from qcbath.model.spin_boson import SpinBosonParams


def test_rk4_step_advances_time() -> None:
    state = FullState(
        x=np.array([1.0, 0.0]),
        p=np.array([0.0, 0.0]),
        r=np.array([0.0]),
        p_r=np.array([0.0]),
    )
    params = FullParams(
        subsystem=SpinBosonParams(epsilon=0.2, delta=0.4),
        bath=BathParams(modes=(BathMode(omega=1.0, coupling=0.1),), temperature=0.1),
    )
    out = RK4Integrator(eom=PBMEEOM()).step(state, dt=0.01, params=params)
    assert out.t > state.t
