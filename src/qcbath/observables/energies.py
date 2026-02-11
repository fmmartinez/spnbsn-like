from __future__ import annotations

from qcbath.model.hamiltonian import FullParams, FullState, total_hamiltonian


def total_energy(state: FullState, params: FullParams) -> dict[str, float]:
    return {"energy": total_hamiltonian(state, params)}
