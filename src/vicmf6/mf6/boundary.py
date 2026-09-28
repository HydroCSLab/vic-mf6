"""Translate the project flux sign into the native MODFLOW API convention.

Positive exchange enters groundwater. API6 represents Q = HCOF*h - RHS, so
a prescribed flux uses HCOF = 0 and RHS = -Q. Keep this sign change in one place."""

from __future__ import annotations

import numpy as np

from ..errors import Mf6RuntimeError
from .variables import resolve_variable_address


def api_rhs_from_rate(rate_m3_per_day: np.ndarray | float) -> np.ndarray:
    """convert the project sign convention to the MF6 API pure-flux RHS."""

    rate = np.asarray(rate_m3_per_day, dtype=np.float64)
    if not np.all(np.isfinite(rate)):
        raise Mf6RuntimeError("API boundary rate contains non-finite values")
    return -rate


class ApiFluxBoundary:
    """Write prescribed fluxes and read the actual converged API package flows.

    Project sign: positive volume enters groundwater. API6 contributes HCOF*h
    minus RHS, so prescribed flux Q uses HCOF=0 and RHS=-Q. NODELIST remains
    one-based because it is passed directly to MODFLOW's native arrays.
    """

    def __init__(
        self, xmi: object, model_name: str, package_name: str, coupled_nodes: np.ndarray
    ) -> None:
        self.xmi = xmi
        self.model_name = model_name
        self.package_name = package_name
        self.coupled_nodes = coupled_nodes
        self.addresses = {
            name: resolve_variable_address(xmi, name, model_name, package_name)
            for name in ("NBOUND", "NODELIST", "HCOF", "RHS", "SIMVALS")
        }

    def validate_capacity(self, node_count: int) -> None:
        xmi = self.xmi
        nodelist = np.asarray(xmi.get_value_ptr(self.addresses["NODELIST"])).reshape(-1)
        hcof = np.asarray(xmi.get_value_ptr(self.addresses["HCOF"])).reshape(-1)
        rhs = np.asarray(xmi.get_value_ptr(self.addresses["RHS"])).reshape(-1)
        simvals = np.asarray(xmi.get_value_ptr(self.addresses["SIMVALS"])).reshape(-1)
        maxbound = min(nodelist.size, hcof.size, rhs.size, simvals.size)
        if maxbound < self.coupled_nodes.size:
            raise Mf6RuntimeError(
                f"API package {self.package_name} in {self.model_name} has MAXBOUND={maxbound}, but {self.coupled_nodes.size} coupled nodes are required"
            )
        if int(self.coupled_nodes.max()) > node_count:
            raise Mf6RuntimeError(
                f"exchange table references MF6 node {int(self.coupled_nodes.max())}, but {self.model_name} has {node_count} nodes"
            )

    def write_rates(self, boundary_rates_m3_per_day: np.ndarray) -> None:
        xmi = self.xmi
        rates = np.asarray(boundary_rates_m3_per_day, dtype=np.float64).reshape(-1)
        if rates.size != self.coupled_nodes.size:
            raise Mf6RuntimeError("internal API rate vector size mismatch")

        nbound = np.asarray(xmi.get_value_ptr(self.addresses["NBOUND"])).reshape(-1)
        nodelist = np.asarray(xmi.get_value_ptr(self.addresses["NODELIST"])).reshape(-1)
        hcof = np.asarray(xmi.get_value_ptr(self.addresses["HCOF"])).reshape(-1)
        rhs = np.asarray(xmi.get_value_ptr(self.addresses["RHS"])).reshape(-1)
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

    def read_applied_rates(self) -> np.ndarray:
        xmi = self.xmi
        values = np.asarray(
            xmi.get_value_ptr(self.addresses["SIMVALS"]), dtype=np.float64
        ).reshape(-1)
        selected = values[: self.coupled_nodes.size].copy()
        if not np.all(np.isfinite(selected)):
            raise Mf6RuntimeError(
                f"API SIMVALS contains non-finite values for {self.model_name}"
            )
        return selected
