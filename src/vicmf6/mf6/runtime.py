"""Persistent rank-local MODFLOW lifecycle through XMI.

Initialize once, preserve groundwater state across VIC restarts, and finalize
only after successful collective completion. Every accepted advance ends on
an original TDIS boundary and samples API flows after finalizing the solve."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import Mf6Config
from ..errors import Mf6RuntimeError
from .advance import advance_model_to_window_boundary
from .boundary import ApiFluxBoundary
from .native_flow import NativeLateralFlow
from .time_steps import TdisSchedule
from .types import Mf6AdvanceResult
from .variables import resolve_variable_address


class Mf6Runtime:
    """Own one persistent rank-local GWF model and its native adapters."""

    def __init__(
        self,
        config: Mf6Config,
        *,
        model_name: str,
        coupled_nodes: Sequence[int],
        logger: object,
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
        self.xmi: object | None = None
        self._head_address: str | None = None
        self.boundary: ApiFluxBoundary | None = None
        self.lateral_flow: NativeLateralFlow | None = None
        self.time_steps = TdisSchedule(config.time_step_boundaries_days)
        self._node_count: int | None = None

    def initialize(self, mpi_comm_handle: int) -> None:
        """initialize XMI once so groundwater state survives every VIC window."""

        try:
            import xmipy
        except ImportError as exc:
            raise Mf6RuntimeError(
                "xmipy is required for coupled execution; install vicmf6[runtime]"
            ) from exc

        self.xmi = xmipy.XmiWrapper(
            str(self.config.library), working_directory=str(self.config.workspace)
        )
        if not hasattr(self.xmi, "initialize_mpi"):
            raise Mf6RuntimeError(
                "the configured xmipy/libmf6 stack does not expose initialize_mpi"
            )

        # initialize_mpi is the complete parallel XMI initialization path.
        # MODFLOW sets the supplied worker communicator and then performs the
        # normal BMI initialization internally. calling initialize() afterward
        # would attempt to initialize the same library a second time.
        self.xmi.initialize_mpi(int(mpi_comm_handle))

        self._head_address = resolve_variable_address(self.xmi, "X", self.model_name)
        self._node_count = int(self.current_head().size)
        self.boundary = ApiFluxBoundary(
            self.xmi, self.model_name, self.api_package, self.coupled_nodes
        )
        self.boundary.validate_capacity(self.node_count)
        self.lateral_flow = NativeLateralFlow(self.xmi, self.model_name, self.logger)

        _log(
            self.logger,
            "info",
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
        if not np.all(np.isfinite(head)):
            bad = np.flatnonzero(~np.isfinite(head))
            raise Mf6RuntimeError(
                f"non-finite MF6 heads for {self.model_name} at nodes {(bad[:20] + 1).tolist()}"
            )
        return head.copy()

    def current_time_days(self) -> float:
        return float(self._require_xmi().get_current_time())

    def advance_to(
        self,
        target_time_days: float,
        volume_by_node_m3: np.ndarray,
        *,
        api_tolerance_m3_per_day: float,
        time_tolerance_days: float = 1.0e-10,
    ) -> Mf6AdvanceResult:
        """Apply a fixed exchange rate until the requested native time boundary."""
        return advance_model_to_window_boundary(
            self,
            target_time_days,
            volume_by_node_m3,
            api_tolerance_m3_per_day=api_tolerance_m3_per_day,
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


def _log(logger: object, level: str, message: str) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message)
