"""Persistent rank-local MODFLOW lifecycle through XMI.

Initialize once, preserve groundwater state across VIC restarts, and finalize
only after successful collective completion. Every accepted advance ends on
an original TDIS boundary and samples API flows after finalizing the solve."""

from __future__ import annotations

from collections.abc import Sequence
from logging import Logger

import numpy as np

from ..config import Mf6Config
from ..errors import Mf6RuntimeError
from .advance import advance_model_to_window_boundary
from .boundary import ApiFluxBoundary, SfrRunoffBoundary
from .native_flow import NativeLateralFlow
from .nodes import read_native_node_indices
from .records import Mf6AdvanceResult
from .time_steps import TdisSchedule
from .variables import resolve_variable_address


class Mf6Runtime:
    """Own one persistent rank-local GWF model and its native adapters."""

    def __init__(
        self,
        config: Mf6Config,
        *,
        model_name: str,
        coupled_nodes: Sequence[int],
        logger: Logger,
        sfr_runoff_package: str | None = None,
    ) -> None:
        self.config = config
        self.model_name = str(model_name).upper()
        self.api_package = config.api_package_for(self.model_name)
        self.solution_id = config.solution_id_for(self.model_name)
        nodes = np.unique(np.asarray(coupled_nodes, dtype=np.int64).reshape(-1))
        if nodes.size == 0 or np.any(nodes < 1):
            raise Mf6RuntimeError(f"{self.model_name} has no valid coupled MF6 nodes")
        self.coupled_nodes = nodes
        self.logger = logger
        self.sfr_runoff_package = sfr_runoff_package
        self.xmi: object | None = None
        self._head_address: str | None = None
        self.boundary: ApiFluxBoundary | None = None
        self.lateral_flow: NativeLateralFlow | None = None
        self.sfr_runoff: SfrRunoffBoundary | None = None
        self.time_steps = TdisSchedule(config.time_step_boundaries_days)
        self._node_count: int | None = None
        self._active_user_indices: np.ndarray | None = None

    def initialize(
        self, mpi_comm_handle: int, *, mpi_comm_size: int | None = None
    ) -> None:
        """Initialize XMI once so groundwater state survives every VIC window.

        A single-model worker communicator uses ordinary XMI initialization and
        MODFLOW's sequential IMS solver.  Multi-model simulations use the
        parallel initialization path on their shared worker communicator.
        """

        if mpi_comm_size is None:
            from mpi4py import MPI

            mpi_comm_size = MPI.Comm.f2py(mpi_comm_handle).Get_size()

        try:
            import xmipy
        except ImportError as exc:
            raise Mf6RuntimeError(
                "xmipy is required for coupled execution; install vicmf6[runtime]"
            ) from exc

        self.xmi = xmipy.XmiWrapper(
            str(self.config.library), working_directory=str(self.config.workspace)
        )
        if mpi_comm_size < 1:
            raise Mf6RuntimeError("MF6 worker communicator size must be positive")
        if mpi_comm_size == 1:
            self.xmi.initialize()
        else:
            if not hasattr(self.xmi, "initialize_mpi"):
                raise Mf6RuntimeError(
                    "the configured xmipy/libmf6 stack does not expose initialize_mpi"
                )
            # initialize_mpi is the complete parallel XMI initialization path.
            # Calling initialize() afterward would initialize the library twice.
            self.xmi.initialize_mpi(int(mpi_comm_handle))

        self._head_address = resolve_variable_address(self.xmi, "X", self.model_name)
        self._node_count, self._active_user_indices, solver_nodes = (
            read_native_node_indices(self.xmi, self.model_name, self.coupled_nodes)
        )
        self.boundary = ApiFluxBoundary(
            self.xmi, self.model_name, self.api_package, solver_nodes
        )
        self.boundary.validate_capacity(self._active_user_indices.size)
        self.lateral_flow = NativeLateralFlow(self.xmi, self.model_name, self.logger)
        if self.sfr_runoff_package is not None:
            self.sfr_runoff = SfrRunoffBoundary(
                self.xmi, self.model_name, self.sfr_runoff_package
            )

        self.logger.info(
            f"mf6 worker initialized model={self.model_name} nodes={self.node_count} api={self.api_package}",
        )

    @property
    def node_count(self) -> int:
        if self._node_count is None:
            self._node_count = int(self.current_head().size)
        return self._node_count

    def current_head(self) -> np.ndarray:
        xmi = self._require_xmi()
        if self._head_address is None:
            raise Mf6RuntimeError("MF6 head address has not been initialized")
        head = np.asarray(
            xmi.get_value_ptr(self._head_address), dtype=np.float64
        ).reshape(-1)
        if self._active_user_indices is not None:
            # Additional package unknowns (e.g. MAW heads) follow grid heads in X.
            head = head[: self._active_user_indices.size]
        if not np.all(np.isfinite(head)):
            bad = np.flatnonzero(~np.isfinite(head))
            raise Mf6RuntimeError(
                f"non-finite MF6 heads for {self.model_name} at nodes {(bad[:20] + 1).tolist()}"
            )
        if self._active_user_indices is None:
            return head.copy()
        user_heads = np.full(self.node_count, np.nan)
        user_heads[self._active_user_indices] = head
        return user_heads

    def current_time_days(self) -> float:
        return float(self._require_xmi().get_current_time())

    def advance_to(
        self,
        target_time_days: float,
        volume_by_node_m3: np.ndarray,
        *,
        api_tolerance_m3_per_day: float,
        sfr_runoff_rates_m3_per_day: np.ndarray | None = None,
        time_tolerance_days: float = 1.0e-10,
    ) -> Mf6AdvanceResult:
        """Apply a fixed exchange rate until the requested native time boundary."""
        return advance_model_to_window_boundary(
            self,
            target_time_days,
            volume_by_node_m3,
            api_tolerance_m3_per_day=api_tolerance_m3_per_day,
            sfr_runoff_rates_m3_per_day=sfr_runoff_rates_m3_per_day,
            time_tolerance_days=time_tolerance_days,
        )

    def finalize(self) -> None:
        if self.xmi is None:
            return
        self.xmi.finalize()
        self.xmi = None

    def _require_xmi(self) -> object:
        if self.xmi is None:
            raise Mf6RuntimeError("MF6 runtime has not been initialized")
        return self.xmi
