from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BathMode:
    omega: float
    coupling: float
    mass: float = 1.0


@dataclass(frozen=True)
class BathParams:
    modes: tuple[BathMode, ...]
    temperature: float

    @property
    def n_modes(self) -> int:
        return len(self.modes)


def discretize_ohmic_exp_cutoff(
    n_modes: int,
    eta: float,
    omega_c: float,
    temperature: float,
    omega_max_factor: float = 5.0,
) -> BathParams:
    """Simple linear-frequency discretization scaffold for J(w)=eta*w*exp(-w/wc)."""
    omegas = np.linspace(omega_c / n_modes, omega_max_factor * omega_c, n_modes)
    delta_w = omegas[1] - omegas[0] if n_modes > 1 else omega_c
    couplings = np.sqrt(2.0 * eta * omegas * np.exp(-omegas / omega_c) * delta_w)
    modes = tuple(BathMode(omega=float(w), coupling=float(c)) for w, c in zip(omegas, couplings))
    return BathParams(modes=modes, temperature=temperature)
