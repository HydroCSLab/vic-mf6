"""Values passed between coupling stages, with physical units in field names."""

from dataclasses import dataclass

import numpy as np

from ..exchange import SignedVolume


@dataclass
class WindowTimings:
    """Rank-local wall times; worker maxima are combined at the end of a window."""

    mapping_seconds: float = 0.0
    mf6_seconds: float = 0.0
    vic_prepare_seconds: float = 0.0
    vic_seconds: float = 0.0
    io_seconds: float = 0.0


@dataclass(frozen=True)
class SurfaceExchange:
    """The broadcast VIC depth and controller-only diagnostics for one window."""

    depth_mm: np.ndarray
    source_volume: SignedVolume
    maximum_water_error_mm: float | None = None
    runoff_depth_mm: np.ndarray | None = None
    runoff_source_volume: SignedVolume | None = None


@dataclass(frozen=True)
class BoundaryExchange:
    """Worker node volumes and controller-only domain conservation totals."""

    volume_by_node_m3: np.ndarray | None
    mapped: SignedVolume | None
    target: SignedVolume | None


@dataclass(frozen=True)
class GroundwaterStatistics:
    """Reduced diagnostics for the whole groundwater domain, on the controller."""

    maximum_api_error: float
    maximum_iterations: int
    lateral_count: int
    lateral_domain_net: float
    lateral_gross: float
    lateral_pair_error: float
    head_min: float
    head_max: float
    head_mean: float
    mapping_seconds: float
    mf6_seconds: float
    mpi_seconds: float
