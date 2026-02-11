from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    epsilon: float
    delta: float
    n_states: int = 2


@dataclass(frozen=True)
class BathConfig:
    n_modes: int
    temperature: float
    eta: float = 0.1
    omega_c: float = 1.0


@dataclass(frozen=True)
class DynamicsConfig:
    dt: float
    n_steps: int
    integrator: str = "rk4"


@dataclass(frozen=True)
class EnsembleConfig:
    n_traj: int = 1
    seed: int = 1234


@dataclass(frozen=True)
class OutputConfig:
    save_every: int = 1


@dataclass(frozen=True)
class SimulationConfig:
    model: ModelConfig
    bath: BathConfig
    dynamics: DynamicsConfig
    ensemble: EnsembleConfig
    output: OutputConfig


def load_config(path: str | Path) -> SimulationConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return SimulationConfig(
        model=ModelConfig(**raw["model"]),
        bath=BathConfig(**raw["bath"]),
        dynamics=DynamicsConfig(**raw["dynamics"]),
        ensemble=EnsembleConfig(**raw.get("ensemble", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
