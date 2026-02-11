from __future__ import annotations

import numpy as np

from qcbath.dynamics.eom import PBMEEOM
from qcbath.model.bath import BathMode, BathParams
from qcbath.model.hamiltonian import FullParams, FullState
from qcbath.model.spin_boson import SpinBosonParams


def test_derivative_shapes_match_state() -> None:
    state = FullState(
        x=np.array([1.0, 0.0]),
        p=np.array([0.0, 1.0]),
        r=np.array([0.0, 0.1]),
        p_r=np.array([0.1, -0.2]),
    )
    params = FullParams(
        subsystem=SpinBosonParams(epsilon=0.0, delta=1.0),
        bath=BathParams(
            modes=(BathMode(omega=0.5, coupling=0.1), BathMode(omega=1.0, coupling=0.2)),
            temperature=0.3,
        ),
    )

    deriv = PBMEEOM().derivatives(state, params)
    assert deriv.x.shape == state.x.shape
    assert deriv.p.shape == state.p.shape
    assert deriv.r.shape == state.r.shape
    assert deriv.p_r.shape == state.p_r.shape
