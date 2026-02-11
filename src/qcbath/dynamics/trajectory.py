from __future__ import annotations

from typing import Callable

from qcbath.dynamics.integrators.base import Integrator
from qcbath.model.hamiltonian import FullParams, FullState


def propagate_trajectory(
    initial_state: FullState,
    params: FullParams,
    integrator: Integrator,
    dt: float,
    n_steps: int,
    observe: Callable[[FullState, FullParams], dict[str, float]],
    save_every: int = 1,
) -> list[dict[str, float]]:
    state = initial_state.copy()
    series: list[dict[str, float]] = []
    for step in range(n_steps + 1):
        if step % save_every == 0:
            row = {"t": state.t}
            row.update(observe(state, params))
            series.append(row)
        if step < n_steps:
            state = integrator.step(state, dt, params)
    return series
