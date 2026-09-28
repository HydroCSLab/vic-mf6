"""Read sparse native connectivity for saved FLOW-JA-FACE diagnostics.

Topology is needed to identify reverse flow pairs; array positions alone
cannot establish which two groundwater nodes exchanged water."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .mf6_metadata import _time_step_length


def _load_flowja(
    cbc_path: Path,
    cbc: Any,
    flowja_name: str | None,
    times: np.ndarray,
    node_count: int,
) -> dict[str, Any] | None:
    if flowja_name is None:
        return None
    grid_path = _find_grb(cbc_path.parent)
    if grid_path is None:
        return {"available": False, "reason": "binary grid file not found"}
    try:
        try:
            from flopy.mf6.utils.binarygrid_util import MfGrdFile
        except ImportError:
            from flopy.utils import MfGrdFile
        grb = MfGrdFile(str(grid_path))
        ia = np.asarray(grb.ia, dtype=np.int64).reshape(-1)
        ja = np.asarray(grb.ja, dtype=np.int64).reshape(-1)
    except Exception as exc:
        return {"available": False, "reason": f"could not read {grid_path}: {exc}"}

    if ia.size != node_count + 1:
        return {
            "available": False,
            "reason": f"IA length {ia.size} != nodes+1 {node_count + 1}",
        }
    if ia[0] == 1:
        ia = ia - 1
    if ja.min(initial=0) >= 1 and ja.max(initial=0) <= node_count:
        ja = ja - 1

    pairs: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for time_index, time_value in enumerate(times):
        try:
            raw_records = cbc.get_data(text=flowja_name, totim=float(time_value))
        except Exception:
            continue
        if not raw_records:
            continue
        flowja = np.asarray(raw_records[0], dtype=np.float64).reshape(-1)
        if flowja.size != ja.size:
            continue
        dt = _time_step_length(times, time_index)

        reverse: dict[tuple[int, int], float] = {}
        row_sum = np.zeros(node_count, dtype=np.float64)
        for i in range(node_count):
            for position in range(int(ia[i]), int(ia[i + 1])):
                j = int(ja[position])
                if i == j:
                    continue
                q = float(flowja[position])
                row_sum[i] += q
                reverse[(i, j)] = q

        for i in range(node_count):
            cells.append(
                {
                    "time_days": float(time_value),
                    "dt_days": dt,
                    "node": i + 1,
                    "flowja_row_net_rate_m3_day": float(row_sum[i]),
                    "flowja_row_net_volume_m3": float(row_sum[i]) * dt,
                }
            )
        for (i, j), q_ij in reverse.items():
            if i >= j:
                continue
            q_ji = reverse.get((j, i))
            if q_ji is None:
                continue
            pairs.append(
                {
                    "time_days": float(time_value),
                    "dt_days": dt,
                    "node_i": i + 1,
                    "node_j": j + 1,
                    "q_i_to_j_record_m3_day": q_ij,
                    "q_j_to_i_record_m3_day": q_ji,
                    "antisymmetry_error_m3_day": q_ij + q_ji,
                    "pair_magnitude_m3_day": 0.5 * (abs(q_ij) + abs(q_ji)),
                    "pair_volume_m3": 0.5 * (abs(q_ij) + abs(q_ji)) * dt,
                }
            )
    return {
        "available": bool(cells),
        "grid_path": grid_path,
        "cell_rows": cells,
        "pair_rows": pairs,
    }


def _find_grb(workspace: Path) -> Path | None:
    candidates = sorted(workspace.glob("*.grb"))
    return candidates[0] if len(candidates) == 1 else None
