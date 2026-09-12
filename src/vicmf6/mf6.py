"""persistent modflow 6 XMI runtime and signed API6 boundary handling."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import Mf6Config
from .errors import Mf6RuntimeError
from .exchange import SignedVolume


@dataclass(frozen=True, slots=True)
class LateralFlowDiagnostics:
    """integrated internal FLOWJA accounting for one model advance."""

    net_by_node_m3: np.ndarray
    domain_net_m3: float
    gross_pair_volume_m3: float
    maximum_pair_antisymmetry_m3: float


@dataclass(frozen=True, slots=True)
class Mf6AdvanceResult:
    """groundwater state and interface diagnostics at one coupling boundary."""

    head_m: np.ndarray
    requested: SignedVolume
    applied: SignedVolume
    maximum_api_error_m3_per_day: float
    nonlinear_iterations: int
    lateral: LateralFlowDiagnostics | None


def api_rhs_from_rate(rate_m3_per_day: np.ndarray | float) -> np.ndarray:
    """convert the project sign convention to the MF6 API pure-flux RHS."""

    rate = np.asarray(rate_m3_per_day, dtype=np.float64)
    if not np.all(np.isfinite(rate)):
        raise Mf6RuntimeError("API boundary rate contains non-finite values")
    return -rate


class Mf6Runtime:
    """one persistent rank-local GWF model controlled through xmipy."""

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
        self._api_addresses: dict[str, str] = {}
        self._flowja_address: str | None = None
        self._ia_address: str | None = None
        self._ja_address: str | None = None
        self._ia: np.ndarray | None = None
        self._ja: np.ndarray | None = None

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

        self._head_address = self._address("X", self.model_name)
        self._api_addresses = {
            name: self._address(name, self.model_name, self.api_package)
            for name in ("NBOUND", "NODELIST", "HCOF", "RHS", "SIMVALS")
        }
        self._initialize_api_layout()
        self._discover_flowja_topology()

        _log(
            self.logger,
            "info",
            f"mf6 worker initialized model={self.model_name} nodes={self.node_count} api={self.api_package}",
        )

    @property
    def node_count(self) -> int:
        return int(self.current_head().size)

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
        """hold one interface rate field fixed while MF6 advances to the window boundary."""

        xmi = self._require_xmi()
        current = self.current_time_days()
        target = float(target_time_days)
        if target <= current + time_tolerance_days:
            raise Mf6RuntimeError(
                f"MF6 target time must be later than current time: current={current} target={target}"
            )

        volume = np.asarray(volume_by_node_m3, dtype=np.float64).reshape(-1)
        if volume.size != self.node_count:
            raise Mf6RuntimeError(
                f"mapped volume for {self.model_name} has {volume.size} nodes; MF6 head has {self.node_count}"
            )
        if not np.all(np.isfinite(volume)):
            raise Mf6RuntimeError(
                f"mapped volume for {self.model_name} contains non-finite values"
            )

        window_days = target - current
        rates = volume / window_days
        boundary_rates = rates[self.coupled_nodes - 1]
        requested = SignedVolume.from_values(volume[self.coupled_nodes - 1])

        applied_volumes: list[np.ndarray] = []
        flowja_volume: np.ndarray | None = None
        maximum_api_error = 0.0
        total_iterations = 0

        while current < target - time_tolerance_days:
            # immediately after XMI initialization MODFLOW reports dt=0.0.
            # TDIS is the authoritative source of the numerical time-step
            # schedule, so derive the next step from the already parsed model
            # input rather than treating the pre-formulation XMI value as a
            # valid time step. this also prevents the coupler from silently
            # changing the user's MF6 discretization.
            dt = _next_tdis_time_step_days(
                self.config.time_step_boundaries_days,
                current,
                tolerance_days=time_tolerance_days,
            )
            if current + dt > target + time_tolerance_days:
                raise Mf6RuntimeError(
                    "an MF6 time step would cross a coupling boundary; refusing to silently change the numerical scheme: "
                    f"model={self.model_name} current={current:.17g} dt={dt:.17g} target={target:.17g}"
                )

            xmi.prepare_time_step(dt)
            reported_dt = float(xmi.get_time_step())
            if not np.isfinite(reported_dt) or abs(
                reported_dt - dt
            ) > time_tolerance_days * max(abs(dt), 1.0):
                raise Mf6RuntimeError(
                    "MF6 time step after prepare_time_step does not match TDIS: "
                    f"model={self.model_name} expected={dt:.17g} reported={reported_dt:.17g}"
                )

            self._write_api_rates(boundary_rates)
            xmi.prepare_solve(self.solution_id)

            converged = False
            iterations = 0
            for _ in range(self.config.max_solve_iterations):
                iterations += 1
                if bool(xmi.solve(self.solution_id)):
                    converged = True
                    break
            total_iterations += iterations
            if not converged:
                raise Mf6RuntimeError(
                    f"MF6 did not converge for {self.model_name} at time {current} after {iterations} solve calls"
                )

            # finalizing the nonlinear solve calculates the converged package
            # flows and model FLOWJA terms. sample those arrays after this call
            # and before finalize_time_step advances the model lifecycle.
            xmi.finalize_solve(self.solution_id)

            applied_rate = self._read_api_simvals()
            api_error = float(
                np.max(np.abs(applied_rate - boundary_rates), initial=0.0)
            )
            maximum_api_error = max(maximum_api_error, api_error)
            if api_error > api_tolerance_m3_per_day:
                raise Mf6RuntimeError(
                    f"API6 rate mismatch for {self.model_name}: max_error={api_error:.6e} m3/day tolerance={api_tolerance_m3_per_day:.6e}"
                )
            applied_volumes.append(applied_rate * dt)

            if self._flowja_address is not None:
                flowja_rate = np.asarray(
                    xmi.get_value_ptr(self._flowja_address), dtype=np.float64
                ).reshape(-1)
                if flowja_volume is None:
                    flowja_volume = np.zeros_like(flowja_rate, dtype=np.float64)
                if flowja_rate.size != flowja_volume.size:
                    raise Mf6RuntimeError("FLOWJA size changed during the MF6 run")
                flowja_volume += flowja_rate * dt

            xmi.finalize_time_step()
            current = self.current_time_days()
            if current > target + time_tolerance_days:
                raise Mf6RuntimeError(
                    f"MF6 advanced past coupling boundary for {self.model_name}: current={current} target={target}"
                )

        if abs(current - target) > time_tolerance_days:
            raise Mf6RuntimeError(
                f"MF6 failed to land on coupling boundary for {self.model_name}: current={current} target={target}"
            )

        applied_boundary_volume = (
            np.sum(np.stack(applied_volumes, axis=0), axis=0)
            if applied_volumes
            else np.zeros(self.coupled_nodes.size, dtype=np.float64)
        )
        applied = SignedVolume.from_values(applied_boundary_volume)
        lateral = None
        if flowja_volume is not None and self._ia is not None and self._ja is not None:
            lateral = lateral_flow_diagnostics(flowja_volume, self._ia, self._ja)

        return Mf6AdvanceResult(
            head_m=self.current_head(),
            requested=requested,
            applied=applied,
            maximum_api_error_m3_per_day=maximum_api_error,
            nonlinear_iterations=total_iterations,
            lateral=lateral,
        )

    def finalize(self) -> None:
        if self.xmi is None:
            return
        self.xmi.finalize()
        self.xmi = None

    def _initialize_api_layout(self) -> None:
        xmi = self._require_xmi()
        nodelist = np.asarray(
            xmi.get_value_ptr(self._api_addresses["NODELIST"])
        ).reshape(-1)
        hcof = np.asarray(xmi.get_value_ptr(self._api_addresses["HCOF"])).reshape(-1)
        rhs = np.asarray(xmi.get_value_ptr(self._api_addresses["RHS"])).reshape(-1)
        simvals = np.asarray(xmi.get_value_ptr(self._api_addresses["SIMVALS"])).reshape(
            -1
        )
        maxbound = min(nodelist.size, hcof.size, rhs.size, simvals.size)
        if maxbound < self.coupled_nodes.size:
            raise Mf6RuntimeError(
                f"API package {self.api_package} in {self.model_name} has MAXBOUND={maxbound}, but {self.coupled_nodes.size} coupled nodes are required"
            )
        if int(self.coupled_nodes.max()) > self.node_count:
            raise Mf6RuntimeError(
                f"exchange table references MF6 node {int(self.coupled_nodes.max())}, but {self.model_name} has {self.node_count} nodes"
            )

    def _write_api_rates(self, boundary_rates_m3_per_day: np.ndarray) -> None:
        xmi = self._require_xmi()
        rates = np.asarray(boundary_rates_m3_per_day, dtype=np.float64).reshape(-1)
        if rates.size != self.coupled_nodes.size:
            raise Mf6RuntimeError("internal API rate vector size mismatch")

        nbound = np.asarray(xmi.get_value_ptr(self._api_addresses["NBOUND"])).reshape(
            -1
        )
        nodelist = np.asarray(
            xmi.get_value_ptr(self._api_addresses["NODELIST"])
        ).reshape(-1)
        hcof = np.asarray(xmi.get_value_ptr(self._api_addresses["HCOF"])).reshape(-1)
        rhs = np.asarray(xmi.get_value_ptr(self._api_addresses["RHS"])).reshape(-1)
        if nbound.size < 1:
            raise Mf6RuntimeError("API NBOUND pointer is empty")

        count = rates.size
        nbound[0] = count
        nodelist[:count] = self.coupled_nodes
        hcof[:count] = 0.0
        rhs[:count] = api_rhs_from_rate(rates)
        if count < hcof.size:
            hcof[count:] = 0.0
        if count < rhs.size:
            rhs[count:] = 0.0

    def _read_api_simvals(self) -> np.ndarray:
        xmi = self._require_xmi()
        values = np.asarray(
            xmi.get_value_ptr(self._api_addresses["SIMVALS"]), dtype=np.float64
        ).reshape(-1)
        selected = values[: self.coupled_nodes.size].copy()
        if not np.all(np.isfinite(selected)):
            raise Mf6RuntimeError(
                f"API SIMVALS contains non-finite values for {self.model_name}"
            )
        return selected

    def _discover_flowja_topology(self) -> None:
        xmi = self._require_xmi()
        self._flowja_address = self._optional_address("FLOWJA", self.model_name)
        if self._flowja_address is None:
            return

        self._ia_address = self._find_address_by_suffix("IA")
        self._ja_address = self._find_address_by_suffix("JA")
        if self._ia_address is None or self._ja_address is None:
            _log(
                self.logger,
                "warning",
                f"FLOWJA is available for {self.model_name}, but IA/JA topology was not exposed; lateral diagnostics are disabled",
            )
            self._flowja_address = None
            return
        self._ia = (
            np.asarray(xmi.get_value_ptr(self._ia_address), dtype=np.int64)
            .reshape(-1)
            .copy()
        )
        self._ja = (
            np.asarray(xmi.get_value_ptr(self._ja_address), dtype=np.int64)
            .reshape(-1)
            .copy()
        )

    def _find_address_by_suffix(self, variable_name: str) -> str | None:
        xmi = self._require_xmi()
        target = variable_name.upper()
        names: list[str] = []
        for method_name in ("get_input_var_names", "get_output_var_names"):
            method = getattr(xmi, method_name, None)
            if not callable(method):
                continue
            try:
                names.extend(str(name) for name in method())
            except Exception:
                continue
        candidates = [
            name
            for name in names
            if name.upper().endswith("/" + target)
            and self.model_name in name.upper().split("/")
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # discretization arrays are preferred when multiple packages expose
            # similarly named values.
            dis = [
                name for name in candidates if re.search(r"/DIS[UV]?/", name.upper())
            ]
            if len(dis) == 1:
                return dis[0]
        return None

    def _optional_address(
        self, var_name: str, component_name: str, subcomponent_name: str = ""
    ) -> str | None:
        try:
            address = self._address(var_name, component_name, subcomponent_name)
            _ = self._require_xmi().get_value_ptr(address)
            return address
        except Exception:
            return None

    def _address(
        self, var_name: str, component_name: str, subcomponent_name: str = ""
    ) -> str:
        try:
            return str(
                self._require_xmi().get_var_address(
                    var_name, component_name, subcomponent_name
                )
            )
        except Exception as exc:
            suffix = (
                f"/{subcomponent_name}/{var_name}"
                if subcomponent_name
                else f"/{var_name}"
            )
            raise Mf6RuntimeError(
                f"failed to resolve XMI variable {component_name}{suffix}: {exc}"
            ) from exc

    def _require_xmi(self) -> object:
        if self.xmi is None:
            raise Mf6RuntimeError("MF6 runtime has not been initialized")
        return self.xmi


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


def _strip_comment(raw_line: str) -> str:
    return raw_line.split("#", 1)[0].strip()


def _next_tdis_time_step_days(
    boundaries_days: Sequence[float],
    current_time_days: float,
    *,
    tolerance_days: float,
) -> float:
    """return the next model-defined TDIS step without modifying the schedule."""

    current = float(current_time_days)
    tolerance = float(tolerance_days)
    boundaries = np.asarray(boundaries_days, dtype=np.float64).reshape(-1)
    if boundaries.size == 0:
        raise Mf6RuntimeError("MF6 TDIS contains no time-step boundaries")
    if not np.all(np.isfinite(boundaries)):
        raise Mf6RuntimeError("MF6 TDIS time-step boundaries contain non-finite values")

    previous = 0.0
    for boundary in boundaries:
        boundary_value = float(boundary)
        if abs(current - previous) <= tolerance:
            dt = boundary_value - previous
            if dt <= 0.0:
                raise Mf6RuntimeError(
                    f"MF6 TDIS contains a non-positive time step: {dt}"
                )
            return dt
        previous = boundary_value

    if abs(current - previous) <= tolerance:
        raise Mf6RuntimeError(
            f"MF6 is already at the final TDIS boundary: current={current:.17g}"
        )

    raise Mf6RuntimeError(
        "MF6 current time does not coincide with a parsed TDIS boundary: "
        f"current={current:.17g}"
    )


def _log(logger: object, level: str, message: str) -> None:
    method = getattr(logger, level, None)
    if callable(method):
        method(message)
