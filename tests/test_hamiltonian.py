from __future__ import annotations

import numpy as np

from qcbath.model.bath import BathMode, BathParams
from qcbath.model.hamiltonian import FullParams, FullState, total_hamiltonian
from qcbath.model.spin_boson import SpinBosonParams


def test_total_hamiltonian_returns_finite_float() -> None:
    params = FullParams(
        subsystem=SpinBosonParams(epsilon=0.1, delta=0.2),
        bath=BathParams(modes=(BathMode(omega=1.0, coupling=0.1),), temperature=0.2),
    )
    state = FullState(
        x=np.array([1.0, 0.0]),
        p=np.array([0.0, 0.0]),
        r=np.array([0.1]),
        p_r=np.array([0.2]),
    )
    energy = total_hamiltonian(state, params)
    assert isinstance(energy, float)
    assert np.isfinite(energy)
