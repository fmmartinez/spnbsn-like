from __future__ import annotations

from collections import defaultdict


def average_timeseries(all_series: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    if not all_series:
        return []
    n_traj = len(all_series)
    n_rows = len(all_series[0])
    out: list[dict[str, float]] = []

    for i in range(n_rows):
        acc: dict[str, float] = defaultdict(float)
        for series in all_series:
            for k, v in series[i].items():
                acc[k] += float(v)
        out.append({k: v / n_traj for k, v in acc.items()})
    return out
