from __future__ import annotations

import csv
from pathlib import Path


def write_timeseries_csv(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
