"""Translate full input-grid identities to MF6's active solver numbering.

Exchange CSVs retain one-based user nodes. MF6 removes inactive and pass-through
cells before solving, so X and API NODELIST use a different, compact numbering.
Only the native adapter performs this translation; geometry stays unchanged.
"""

import numpy as np

from ..errors import Mf6RuntimeError
from .variables import resolve_variable_address


def read_native_node_indices(xmi, model_name: str, coupled_user_nodes: np.ndarray):
    """Return user-grid size, active user indices, and one-based API solver nodes."""

    def read(name):
        # DIS is the runtime namespace for DIS, DISV, and DISU. Input-package
        # copies of NODES can coexist with this active solver count.
        address = resolve_variable_address(xmi, name, model_name, "DIS")
        return np.asarray(xmi.get_value_ptr(address), dtype=np.int64).reshape(-1)

    user_count = int(read("NODESUSER")[0])
    solver_count = int(read("NODES")[0])
    if not 0 < solver_count <= user_count:
        raise Mf6RuntimeError(f"invalid MF6 grid sizes for {model_name}")
    active_user_indices = (
        read("NODEUSER") - 1 if solver_count < user_count else np.arange(user_count)
    )
    if (
        active_user_indices.size != solver_count
        or np.any(active_user_indices < 0)
        or np.any(active_user_indices >= user_count)
        or np.any(np.diff(active_user_indices) <= 0)
    ):
        raise Mf6RuntimeError(f"invalid MF6 NODEUSER mapping for {model_name}")
    if np.any(coupled_user_nodes < 1) or np.any(coupled_user_nodes > user_count):
        raise Mf6RuntimeError(f"exchange table references nodes outside {model_name}")
    user_to_solver = np.zeros(user_count, dtype=np.int64)
    user_to_solver[active_user_indices] = np.arange(1, solver_count + 1)
    coupled_solver_nodes = user_to_solver[coupled_user_nodes - 1]
    if np.any(coupled_solver_nodes == 0):
        raise Mf6RuntimeError(
            f"exchange table references inactive or pass-through nodes in {model_name}"
        )
    return user_count, active_user_indices, coupled_solver_nodes
