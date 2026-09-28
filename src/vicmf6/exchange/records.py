"""Named quantities passed across the VIC--MODFLOW interface.

Positive and negative transfer are tracked separately because a net-only check
can hide equal and opposite errors. Opposite signs may legitimately cancel
when several overlap rows are combined into one groundwater boundary node."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignedVolume:
    """gross positive, gross negative, and signed net transfer in cubic metres."""

    positive_m3: float
    negative_m3: float
    net_m3: float

    @classmethod
    def from_values(cls, values_m3: Sequence[float] | np.ndarray) -> SignedVolume:
        values = np.asarray(values_m3, dtype=np.float64)
        positive = float(values[values > 0.0].sum(dtype=np.float64))
        negative = float(values[values < 0.0].sum(dtype=np.float64))
        return cls(
            positive_m3=positive,
            negative_m3=negative,
            net_m3=float(values.sum(dtype=np.float64)),
        )


@dataclass(frozen=True, slots=True)
class MappingResult:
    """one model's overlap-scale and node-aggregated interface volumes."""

    volume_by_node_m3: np.ndarray
    signed_volume: SignedVolume
    node_signed_volume: SignedVolume


@dataclass(frozen=True, slots=True)
class HeadContribution:
    """partial reverse-map sums to be reduced across groundwater workers."""

    head_area_sum_m3: np.ndarray
    area_sum_m2: np.ndarray


@dataclass(frozen=True, slots=True)
class VicCell:
    """explicit vic array identity used independently from vic_id."""

    position: int
    vic_id: str
    row: int
    col: int
    area_m2: float
    latitude: float | None
    longitude: float | None
    interface_elevation_m: float | None
