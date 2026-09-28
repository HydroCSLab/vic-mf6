"""Groundwater results returned at an accepted coupling boundary.

Requested and applied volumes remain separate: writing an API value alone
does not demonstrate that the converged native solver applied it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exchange import SignedVolume


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
