"""Find native TDIS substeps without changing the model's time discretization.

Validate the fixed schedule once. Binary search then locates a substep in
O(log steps), avoiding a full scan from day zero at every model advance.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

import numpy as np

from ..errors import Mf6RuntimeError


class TdisSchedule:
    """An increasing sequence of elapsed-day boundaries, preceded by time zero."""

    def __init__(self, boundaries_days: Sequence[float]) -> None:
        boundaries = np.asarray(boundaries_days, dtype=np.float64).reshape(-1)
        if not boundaries.size:
            raise Mf6RuntimeError("MF6 TDIS contains no time-step boundaries")
        if not np.all(np.isfinite(boundaries)):
            raise Mf6RuntimeError(
                "MF6 TDIS time-step boundaries contain non-finite values"
            )
        self.boundaries = (0.0, *map(float, boundaries))
        steps = np.diff(self.boundaries)
        if np.any(steps <= 0.0):
            raise Mf6RuntimeError(
                f"MF6 TDIS contains a non-positive time step: {steps.min()}"
            )

    def next_step_days(
        self, current_time_days: float, *, tolerance_days: float
    ) -> float:
        """Require the current time to coincide with an original model boundary.

        Choose the earliest matching boundary, as the original sequential scan
        did. Looking back one position protects the tolerance comparison from
        rounding in the binary-search lower bound.
        """
        current = float(current_time_days)
        tolerance = float(tolerance_days)
        if not np.isfinite(current) or not np.isfinite(tolerance) or tolerance < 0.0:
            raise Mf6RuntimeError(
                "MF6 current time and nonnegative time tolerance must be finite"
            )
        index = max(0, bisect_left(self.boundaries, current - tolerance) - 1)
        while index < len(self.boundaries):
            previous = self.boundaries[index]
            if abs(current - previous) <= tolerance:
                if index + 1 == len(self.boundaries):
                    raise Mf6RuntimeError(
                        f"MF6 is already at the final TDIS boundary: current={current:.17g}"
                    )
                return self.boundaries[index + 1] - previous
            if previous > current + tolerance:
                break
            index += 1
        raise Mf6RuntimeError(
            "MF6 current time does not coincide with a parsed TDIS boundary: "
            f"current={current:.17g}"
        )
