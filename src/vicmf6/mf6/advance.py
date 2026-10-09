"""Advance persistent groundwater state using the original native TDIS schedule.

The exchange is a window-integrated volume. Dividing by the window duration
produces a constant rate; each native substep contributes rate * dt to the
applied volume. Accumulation uses one vector, independent of substep count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..errors import Mf6RuntimeError
from ..exchange import SignedVolume
from .lateral_flow import lateral_flow_diagnostics
from .records import Mf6AdvanceResult

if TYPE_CHECKING:
    from .runtime import Mf6Runtime


def advance_model_to_window_boundary(
    runtime: Mf6Runtime,
    target_time_days: float,
    volume_by_node_m3: np.ndarray,
    *,
    api_tolerance_m3_per_day: float,
    sfr_runoff_rates_m3_per_day: np.ndarray | None = None,
    time_tolerance_days: float = 1.0e-10,
) -> Mf6AdvanceResult:
    """hold one interface rate field fixed while MF6 advances to the window boundary."""

    xmi = runtime._require_xmi()
    current = runtime.current_time_days()
    target = float(target_time_days)
    if target <= current + time_tolerance_days:
        raise Mf6RuntimeError(
            f"MF6 target time must be later than current time: current={current} target={target}"
        )

    volume = np.asarray(volume_by_node_m3, dtype=np.float64).reshape(-1)
    if volume.size != runtime.node_count:
        raise Mf6RuntimeError(
            f"mapped volume for {runtime.model_name} has {volume.size} nodes; MF6 head has {runtime.node_count}"
        )
    if not np.all(np.isfinite(volume)):
        raise Mf6RuntimeError(
            f"mapped volume for {runtime.model_name} contains non-finite values"
        )

    window_days = target - current
    rates = volume / window_days
    boundary_rates = rates[runtime.coupled_nodes - 1]
    requested = SignedVolume.from_values(volume[runtime.coupled_nodes - 1])

    applied_boundary_volume = np.zeros(runtime.coupled_nodes.size, dtype=np.float64)
    applied_runoff_volume = None
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
        dt = runtime.time_steps.next_step_days(
            current,
            tolerance_days=time_tolerance_days,
        )
        if current + dt > target + time_tolerance_days:
            raise Mf6RuntimeError(
                "an MF6 time step would cross a coupling boundary; refusing to silently change the numerical scheme: "
                f"model={runtime.model_name} current={current:.17g} dt={dt:.17g} target={target:.17g}"
            )

        applied_rate, api_error, iterations, runoff_rate = solve_groundwater_time_step(
            runtime,
            dt,
            boundary_rates,
            api_tolerance_m3_per_day=api_tolerance_m3_per_day,
            time_tolerance_days=time_tolerance_days,
            sfr_runoff_rates_m3_per_day=sfr_runoff_rates_m3_per_day,
        )
        maximum_api_error = max(maximum_api_error, api_error)
        total_iterations += iterations
        applied_boundary_volume += applied_rate * dt
        if runoff_rate is not None:
            if applied_runoff_volume is None:
                applied_runoff_volume = np.zeros_like(runoff_rate)
            applied_runoff_volume += runoff_rate * dt

        flowja_rate = (
            runtime.lateral_flow.current_rates() if runtime.lateral_flow else None
        )
        if flowja_rate is not None:
            if flowja_volume is None:
                flowja_volume = np.zeros_like(flowja_rate, dtype=np.float64)
            if flowja_rate.size != flowja_volume.size:
                raise Mf6RuntimeError("FLOWJA size changed during the MF6 run")
            flowja_volume += flowja_rate * dt

        xmi.finalize_time_step()
        next_time = runtime.current_time_days()
        if not np.isfinite(next_time) or next_time <= current:
            raise Mf6RuntimeError(
                f"MF6 did not advance time for {runtime.model_name}: current={current} next={next_time}"
            )
        current = next_time
        if current > target + time_tolerance_days:
            raise Mf6RuntimeError(
                f"MF6 advanced past coupling boundary for {runtime.model_name}: current={current} target={target}"
            )

    if abs(current - target) > time_tolerance_days:
        raise Mf6RuntimeError(
            f"MF6 failed to land on coupling boundary for {runtime.model_name}: current={current} target={target}"
        )

    applied = SignedVolume.from_values(applied_boundary_volume)
    lateral = None
    if flowja_volume is not None:
        assert runtime.lateral_flow is not None
        lateral = lateral_flow_diagnostics(
            flowja_volume, runtime.lateral_flow.ia, runtime.lateral_flow.ja
        )

    return Mf6AdvanceResult(
        head_m=runtime.current_head(),
        requested=requested,
        applied=applied,
        maximum_api_error_m3_per_day=maximum_api_error,
        nonlinear_iterations=total_iterations,
        lateral=lateral,
        applied_runoff=(
            None
            if applied_runoff_volume is None
            else SignedVolume.from_values(applied_runoff_volume)
        ),
    )


def solve_groundwater_time_step(
    runtime: Mf6Runtime,
    dt: float,
    boundary_rates: np.ndarray,
    *,
    api_tolerance_m3_per_day: float,
    time_tolerance_days: float,
    sfr_runoff_rates_m3_per_day: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int, np.ndarray | None]:
    """Prepare, solve, and sample one substep, leaving time finalization to the caller.

    API SIMVALS and FLOWJA are valid only after finalize_solve. Both must be
    sampled before finalize_time_step changes the native model state.
    """
    xmi = runtime._require_xmi()
    boundary = runtime.boundary
    if boundary is None:
        raise Mf6RuntimeError("MF6 API boundary has not been initialized")
    xmi.prepare_time_step(dt)
    reported_dt = float(xmi.get_time_step())
    if not np.isfinite(reported_dt) or abs(
        reported_dt - dt
    ) > time_tolerance_days * max(abs(dt), 1.0):
        raise Mf6RuntimeError(
            "MF6 time step after prepare_time_step does not match TDIS: "
            f"model={runtime.model_name} expected={dt:.17g} reported={reported_dt:.17g}"
        )

    boundary.write_rates(boundary_rates)
    if sfr_runoff_rates_m3_per_day is not None:
        if runtime.sfr_runoff is None:
            raise Mf6RuntimeError(
                "SFR runoff rates were provided without an initialized SFR boundary"
            )
        runtime.sfr_runoff.write_rates(sfr_runoff_rates_m3_per_day)
    xmi.prepare_solve(runtime.solution_id)

    converged = False
    iterations = 0
    for _ in range(runtime.config.max_solve_iterations):
        iterations += 1
        if bool(xmi.solve(runtime.solution_id)):
            converged = True
            break
    if not converged:
        raise Mf6RuntimeError(
            f"MF6 did not converge for {runtime.model_name} at time {runtime.current_time_days()} after {iterations} solve calls"
        )

    # finalizing the nonlinear solve calculates the converged package
    # flows and model FLOWJA terms. sample those arrays after this call
    # and before finalize_time_step advances the model lifecycle.
    xmi.finalize_solve(runtime.solution_id)

    applied_rate = boundary.read_applied_rates()
    api_error = float(np.max(np.abs(applied_rate - boundary_rates), initial=0.0))
    if api_error > api_tolerance_m3_per_day:
        raise Mf6RuntimeError(
            f"API6 rate mismatch for {runtime.model_name}: max_error={api_error:.6e} m3/day tolerance={api_tolerance_m3_per_day:.6e}"
        )
    runoff_rate = None
    if sfr_runoff_rates_m3_per_day is not None:
        runoff_rate = runtime.sfr_runoff.read_applied_rates()
        requested_runoff = np.zeros_like(runoff_rate)
        requested_runoff[: sfr_runoff_rates_m3_per_day.size] = (
            sfr_runoff_rates_m3_per_day
        )
        runoff_error = float(
            np.max(np.abs(runoff_rate - requested_runoff), initial=0.0)
        )
        if runoff_error > api_tolerance_m3_per_day:
            raise Mf6RuntimeError(
                f"SFR runoff rate mismatch for {runtime.model_name}: max_error={runoff_error:.6e} m3/day"
            )
    return applied_rate, api_error, iterations, runoff_rate
