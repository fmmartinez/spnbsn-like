from __future__ import annotations

from qcbath.model.hamiltonian import FullParams, FullState


def two_level_coherence(state: FullState, params: FullParams) -> dict[str, float]:
    if len(state.x) < 2:
        return {"coh_re_01": 0.0}
    coh_re = 0.5 * (state.x[0] * state.x[1] + state.p[0] * state.p[1])
    return {"coh_re_01": float(coh_re)}
