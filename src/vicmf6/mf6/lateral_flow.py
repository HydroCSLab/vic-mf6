"""Reconstruct internal lateral-flow accounting from sparse MF6 connections.

Every directed off-diagonal flow has a reverse connection. Their cancellation
is a separate diagnostic from the external VIC interface water transfer."""

from __future__ import annotations

import numpy as np

from ..errors import Mf6RuntimeError
from .types import LateralFlowDiagnostics


def lateral_flow_diagnostics(
    flowja_volume_m3: np.ndarray,
    ia: np.ndarray,
    ja: np.ndarray,
) -> LateralFlowDiagnostics:
    """reconstruct off-diagonal FLOWJA pairs and nodewise net internal volume."""

    flow = np.asarray(flowja_volume_m3, dtype=np.float64).reshape(-1)
    ia_values = np.asarray(ia, dtype=np.int64).reshape(-1)
    ja_values = np.asarray(ja, dtype=np.int64).reshape(-1)
    if flow.size != ja_values.size:
        raise Mf6RuntimeError(
            f"FLOWJA/JA size mismatch: flowja={flow.size} ja={ja_values.size}"
        )
    if ia_values.size < 2:
        raise Mf6RuntimeError("IA must contain at least two offsets")

    # mf6 internally stores ia/ja using one-based Fortran indexing in many XMI
    # views. normalize both arrays once before interpreting the sparse rows.
    ia_zero = ia_values - 1 if ia_values[0] == 1 else ia_values.copy()
    if ia_zero[0] != 0 or ia_zero[-1] != flow.size:
        raise Mf6RuntimeError(
            f"IA does not span FLOWJA: first={ia_zero[0]} last={ia_zero[-1]} nja={flow.size}"
        )
    node_count = ia_zero.size - 1
    if ja_values.size and ja_values.min() >= 1 and ja_values.max() <= node_count:
        ja_zero = ja_values - 1
    else:
        ja_zero = ja_values.copy()
    if np.any(ja_zero < 0) or np.any(ja_zero >= node_count):
        raise Mf6RuntimeError("JA contains node numbers outside the IA topology")

    net = np.zeros(node_count, dtype=np.float64)
    directed: dict[tuple[int, int], float] = {}
    for node in range(node_count):
        for position in range(int(ia_zero[node]), int(ia_zero[node + 1])):
            neighbor = int(ja_zero[position])
            if neighbor == node:
                continue
            value = float(flow[position])
            net[node] += value
            directed[(node, neighbor)] = value

    maximum_antisymmetry = 0.0
    gross_pair = 0.0
    seen: set[tuple[int, int]] = set()
    for (node, neighbor), value in directed.items():
        pair = (min(node, neighbor), max(node, neighbor))
        if pair in seen:
            continue
        seen.add(pair)
        reverse = directed.get((neighbor, node))
        if reverse is None:
            raise Mf6RuntimeError(
                f"FLOWJA topology is missing reverse connection for nodes {node + 1} and {neighbor + 1}"
            )
        maximum_antisymmetry = max(maximum_antisymmetry, abs(value + reverse))
        gross_pair += abs(value)

    return LateralFlowDiagnostics(
        net_by_node_m3=net,
        domain_net_m3=float(net.sum(dtype=np.float64)),
        gross_pair_volume_m3=float(gross_pair),
        maximum_pair_antisymmetry_m3=float(maximum_antisymmetry),
    )
