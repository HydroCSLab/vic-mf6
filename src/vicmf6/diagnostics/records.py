"""Canonical scientific and timing record for one accepted coupling window.

All terms use named units. The record keeps overlap-scale and node-scale
volumes distinct so legitimate cancellation is not mistaken for lost water."""

from __future__ import annotations

from dataclasses import dataclass

from ..exchange import SignedVolume
from ..schedule import CouplingWindow


@dataclass(frozen=True, slots=True)
class WindowDiagnostics:
    """one row in the coupled numerical and performance record."""

    step: int
    start: str
    end: str
    vic_positive_m3: float
    vic_negative_m3: float
    vic_net_m3: float
    mapped_positive_m3: float
    mapped_negative_m3: float
    mapped_net_m3: float
    boundary_positive_m3: float
    boundary_negative_m3: float
    boundary_net_m3: float
    applied_positive_m3: float
    applied_negative_m3: float
    applied_net_m3: float
    positive_mapping_error_m3: float
    negative_mapping_error_m3: float
    net_mapping_error_m3: float
    aggregation_net_error_m3: float
    aggregation_positive_cancellation_m3: float
    aggregation_negative_cancellation_m3: float
    positive_api_error_m3: float
    negative_api_error_m3: float
    net_api_error_m3: float
    head_min_m: float
    head_max_m: float
    head_mean_m: float
    maximum_api_rate_error_m3_per_day: float
    maximum_vic_water_error_mm: float | None
    lateral_domain_net_m3: float | None
    lateral_gross_pair_m3: float | None
    lateral_maximum_pair_antisymmetry_m3: float | None
    nonlinear_iterations_max: int
    vic_prepare_seconds: float
    vic_seconds: float
    mf6_seconds: float
    mapping_seconds: float
    mpi_seconds: float
    io_seconds: float
    window_seconds: float


def make_window_diagnostics(
    *,
    window: CouplingWindow,
    vic: SignedVolume,
    mapped: SignedVolume,
    boundary_target: SignedVolume,
    applied: SignedVolume,
    head_min_m: float,
    head_max_m: float,
    head_mean_m: float,
    maximum_api_rate_error_m3_per_day: float,
    maximum_vic_water_error_mm: float | None,
    lateral_domain_net_m3: float | None,
    lateral_gross_pair_m3: float | None,
    lateral_maximum_pair_antisymmetry_m3: float | None,
    nonlinear_iterations_max: int,
    vic_prepare_seconds: float,
    vic_seconds: float,
    mf6_seconds: float,
    mapping_seconds: float,
    mpi_seconds: float,
    io_seconds: float,
    window_seconds: float,
) -> WindowDiagnostics:
    return WindowDiagnostics(
        step=window.index + 1,
        start=window.start.isoformat(),
        end=window.end.isoformat(),
        vic_positive_m3=vic.positive_m3,
        vic_negative_m3=vic.negative_m3,
        vic_net_m3=vic.net_m3,
        mapped_positive_m3=mapped.positive_m3,
        mapped_negative_m3=mapped.negative_m3,
        mapped_net_m3=mapped.net_m3,
        boundary_positive_m3=boundary_target.positive_m3,
        boundary_negative_m3=boundary_target.negative_m3,
        boundary_net_m3=boundary_target.net_m3,
        applied_positive_m3=applied.positive_m3,
        applied_negative_m3=applied.negative_m3,
        applied_net_m3=applied.net_m3,
        positive_mapping_error_m3=mapped.positive_m3 - vic.positive_m3,
        negative_mapping_error_m3=mapped.negative_m3 - vic.negative_m3,
        net_mapping_error_m3=mapped.net_m3 - vic.net_m3,
        aggregation_net_error_m3=boundary_target.net_m3 - mapped.net_m3,
        aggregation_positive_cancellation_m3=(
            mapped.positive_m3 - boundary_target.positive_m3
        ),
        aggregation_negative_cancellation_m3=(
            boundary_target.negative_m3 - mapped.negative_m3
        ),
        positive_api_error_m3=applied.positive_m3 - boundary_target.positive_m3,
        negative_api_error_m3=applied.negative_m3 - boundary_target.negative_m3,
        net_api_error_m3=applied.net_m3 - boundary_target.net_m3,
        head_min_m=head_min_m,
        head_max_m=head_max_m,
        head_mean_m=head_mean_m,
        maximum_api_rate_error_m3_per_day=maximum_api_rate_error_m3_per_day,
        maximum_vic_water_error_mm=maximum_vic_water_error_mm,
        lateral_domain_net_m3=lateral_domain_net_m3,
        lateral_gross_pair_m3=lateral_gross_pair_m3,
        lateral_maximum_pair_antisymmetry_m3=lateral_maximum_pair_antisymmetry_m3,
        nonlinear_iterations_max=nonlinear_iterations_max,
        vic_prepare_seconds=vic_prepare_seconds,
        vic_seconds=vic_seconds,
        mf6_seconds=mf6_seconds,
        mapping_seconds=mapping_seconds,
        mpi_seconds=mpi_seconds,
        io_seconds=io_seconds,
        window_seconds=window_seconds,
    )
