from __future__ import annotations

import numpy as np

from qcbath.model.bath import BathParams


def sample_classical_bath_state(bath: BathParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample (R, P) from a classical Boltzmann distribution for harmonic modes."""
    r = np.zeros(bath.n_modes)
    p_r = np.zeros(bath.n_modes)

    kb_t = bath.temperature
    for j, mode in enumerate(bath.modes):
        sigma_r = np.sqrt(kb_t / (mode.mass * mode.omega**2))
        sigma_p = np.sqrt(mode.mass * kb_t)
        r[j] = rng.normal(0.0, sigma_r)
        p_r[j] = rng.normal(0.0, sigma_p)

    return r, p_r
