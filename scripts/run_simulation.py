#!/usr/bin/env python3
from __future__ import annotations

import argparse

from qcbath.config import load_config
from qcbath.dynamics.eom import PBMEEOM
from qcbath.dynamics.ensemble import average_timeseries
from qcbath.dynamics.integrators.rk4 import RK4Integrator
from qcbath.dynamics.integrators.velocity_verlet import EulerIntegrator
from qcbath.dynamics.trajectory import propagate_trajectory
from qcbath.init.bath_init import sample_classical_bath_state
from qcbath.init.quantum_init import pure_state_mapping
from qcbath.io.results import write_timeseries_csv
from qcbath.model.bath import discretize_ohmic_exp_cutoff
from qcbath.model.hamiltonian import FullParams, FullState
from qcbath.model.spin_boson import SpinBosonParams
from qcbath.observables.energies import total_energy
from qcbath.observables.populations import two_level_populations
from qcbath.utils.rng import make_rng


def build_observer():
    def observe(state: FullState, params: FullParams) -> dict[str, float]:
        row = {}
        row.update(two_level_populations(state, params))
        row.update(total_energy(state, params))
        return row

    return observe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="outputs/timeseries.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)

    subsystem = SpinBosonParams(epsilon=cfg.model.epsilon, delta=cfg.model.delta)
    bath = discretize_ohmic_exp_cutoff(
        n_modes=cfg.bath.n_modes,
        eta=cfg.bath.eta,
        omega_c=cfg.bath.omega_c,
        temperature=cfg.bath.temperature,
    )
    params = FullParams(subsystem=subsystem, bath=bath)

    eom = PBMEEOM()
    integrator = RK4Integrator(eom=eom) if cfg.dynamics.integrator.lower() == "rk4" else EulerIntegrator(eom=eom)

    rng = make_rng(cfg.ensemble.seed)
    all_series = []
    for _ in range(cfg.ensemble.n_traj):
        x, p = pure_state_mapping(n_states=cfg.model.n_states, occupied_index=0)
        r, p_r = sample_classical_bath_state(bath, rng)
        initial = FullState(x=x, p=p, r=r, p_r=p_r)
        series = propagate_trajectory(
            initial_state=initial,
            params=params,
            integrator=integrator,
            dt=cfg.dynamics.dt,
            n_steps=cfg.dynamics.n_steps,
            observe=build_observer(),
            save_every=cfg.output.save_every,
        )
        all_series.append(series)

    averaged = average_timeseries(all_series)
    write_timeseries_csv(args.out, averaged)
    print(f"Wrote {len(averaged)} rows to {args.out}")


if __name__ == "__main__":
    main()
