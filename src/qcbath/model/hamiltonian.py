from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bath import BathParams
from .spin_boson import SpinBosonParams


@dataclass
class FullState:
    x: np.ndarray
    p: np.ndarray
    r: np.ndarray
    p_r: np.ndarray
    t: float = 0.0

    def copy(self) -> "FullState":
        return FullState(self.x.copy(), self.p.copy(), self.r.copy(), self.p_r.copy(), self.t)


@dataclass(frozen=True)
class FullParams:
    subsystem: SpinBosonParams
    bath: BathParams


def bath_energy(state: FullState, params: FullParams) -> float:
    e = 0.0
    for j, mode in enumerate(params.bath.modes):
        e += 0.5 * state.p_r[j] ** 2 / mode.mass
        e += 0.5 * mode.mass * mode.omega**2 * state.r[j] ** 2
    return float(e)


def subsystem_mapping_energy(state: FullState, params: FullParams) -> float:
    h = params.subsystem.subsystem_hamiltonian()
    # Scaffold bilinear form; adapt to your exact PBME convention.
    return float(0.5 * (state.x @ h @ state.x + state.p @ h @ state.p))


def coupling_energy(state: FullState, params: FullParams) -> float:
    if len(state.x) < 2:
        return 0.0
    sigma_z_like = 0.5 * ((state.x[0] ** 2 + state.p[0] ** 2) - (state.x[1] ** 2 + state.p[1] ** 2))
    return float(sum(mode.coupling * state.r[j] * sigma_z_like for j, mode in enumerate(params.bath.modes)))


def total_hamiltonian(state: FullState, params: FullParams) -> float:
    return bath_energy(state, params) + subsystem_mapping_energy(state, params) + coupling_energy(state, params)
