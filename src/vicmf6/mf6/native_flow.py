"""Optional native FLOWJA rates and their fixed sparse connectivity.

The API exchange remains required; lateral diagnostics are available only when
MF6 exposes both flow values and enough topology to identify reverse pairs.
"""

from __future__ import annotations

from logging import Logger

import numpy as np

from .variables import find_model_variable_by_suffix, optional_variable_address


class NativeLateralFlow:
    """Retain topology once and read current rates after a converged solve."""

    def __init__(self, xmi: object, model_name: str, logger: Logger) -> None:
        self.xmi = xmi
        self.ia = self.ja = None
        self.address = optional_variable_address(xmi, "FLOWJA", model_name)
        if self.address is None:
            return

        ia_address = find_model_variable_by_suffix(xmi, model_name, "IA")
        ja_address = find_model_variable_by_suffix(xmi, model_name, "JA")
        if ia_address is None or ja_address is None:
            logger.warning(
                f"FLOWJA is available for {model_name}, but IA/JA topology was not exposed; lateral diagnostics are disabled",
            )
            self.address = None
            return
        self.ia = (
            np.asarray(xmi.get_value_ptr(ia_address), dtype=np.int64).reshape(-1).copy()
        )
        self.ja = (
            np.asarray(xmi.get_value_ptr(ja_address), dtype=np.int64).reshape(-1).copy()
        )

    def current_rates(self) -> np.ndarray | None:
        if self.address is None:
            return None
        return np.asarray(
            self.xmi.get_value_ptr(self.address), dtype=np.float64
        ).reshape(-1)
