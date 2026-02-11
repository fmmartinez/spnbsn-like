# qcbath

Starter code skeleton for simulating quantum subsystem dynamics (PBME-style mapping variables)
coupled to a classical harmonic bath with spin-boson-like Hamiltonians.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_simulation.py --config examples/sbm_population_decay.yaml
```

## Current status

This is a scaffold:
- data models for subsystem + bath
- Hamiltonian decomposition
- EOM interface
- simple Euler / RK4 integrators
- trajectory + ensemble driver
- placeholder observables

You can now fill in PBME-specific equations in `src/qcbath/dynamics/eom.py`.
