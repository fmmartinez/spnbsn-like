from __future__ import annotations

from qcbath.model.hamiltonian import FullParams, FullState


def two_level_populations(state: FullState, params: FullParams) -> dict[str, float]:
    if len(state.x) < 2:
        return {"pop_0": 1.0}
    pop0 = 0.5 * (state.x[0] ** 2 + state.p[0] ** 2)
    pop1 = 0.5 * (state.x[1] ** 2 + state.p[1] ** 2)
    return {"pop_0": float(pop0), "pop_1": float(pop1), "pop_diff": float(pop0 - pop1)}
