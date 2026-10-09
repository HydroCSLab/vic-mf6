"""Account for the groundwater storage and external-flow budget.

Interface conservation is necessary but does not by itself establish closure
of the complete groundwater model with all its other stresses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_mf6_mass_balance_table(
    budget_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-model and domain-integrated MF6 CBC mass-balance totals.

    MODFLOW CBC storage terms (for example STO-SS and STO-SY) are flows from
    storage into the groundwater equation.  The corresponding physical storage
    state change therefore has the opposite sign.
    """

    rows: list[dict[str, Any]] = []
    domain_api = 0.0
    domain_storage_budget = 0.0
    domain_other = 0.0
    complete = bool(budget_data)
    reasons: list[str] = [] if budget_data else ["no native budget evidence"]

    for model_name, data in budget_data.items():
        if not data.get("available"):
            complete = False
            reasons.append(f"{model_name}: {data.get('reason') or 'CBC unavailable'}")
            continue

        api_name = data.get("api_record_name")
        if not api_name or not data.get("complete") or not data.get("aggregate_terms"):
            complete = False
            reasons.append(f"{model_name}: complete native budget evidence is required")
            continue

        by_term: dict[str, float] = defaultdict(float)
        for item in data.get("aggregate_terms", []):
            by_term[str(item["term"])] += float(item["net_volume_m3"])

        api_key = str(api_name).upper()
        api_volume = sum(
            value for term, value in by_term.items() if term.upper() == api_key
        )
        storage_terms = {
            term: value
            for term, value in by_term.items()
            if term.upper().startswith("STO-")
        }
        storage_budget = float(sum(storage_terms.values()))
        other_terms = {
            term: value
            for term, value in by_term.items()
            if term.upper() != api_key and not term.upper().startswith("STO-")
        }
        other_volume = float(sum(other_terms.values()))
        residual = api_volume + storage_budget + other_volume
        storage_state_change = -storage_budget

        rows.append(
            {
                "model": model_name,
                "api_volume_m3": api_volume,
                "storage_budget_volume_m3": storage_budget,
                "storage_state_change_m3": storage_state_change,
                "other_nonstorage_volume_m3": other_volume,
                "cbc_budget_residual_m3": residual,
                "storage_terms": ";".join(sorted(storage_terms)),
                "other_nonstorage_terms": ";".join(sorted(other_terms)),
                "complete": True,
                "reason": "",
            }
        )
        domain_api += api_volume
        domain_storage_budget += storage_budget
        domain_other += other_volume

    domain_residual = domain_api + domain_storage_budget + domain_other
    rows.append(
        {
            "model": "__DOMAIN__",
            "api_volume_m3": domain_api if complete else None,
            "storage_budget_volume_m3": domain_storage_budget if complete else None,
            "storage_state_change_m3": -domain_storage_budget if complete else None,
            "other_nonstorage_volume_m3": domain_other if complete else None,
            "cbc_budget_residual_m3": domain_residual if complete else None,
            "storage_terms": "",
            "other_nonstorage_terms": "",
            "complete": complete,
            "reason": "; ".join(reasons),
        }
    )
    return rows


def enrich_mass_balance_summary(
    summary: dict[str, Any],
    mf6_mass_balance_rows: list[dict[str, Any]],
) -> None:
    """Attach one headline end-to-end coupled mass balance to run_summary."""

    domain = next(
        (row for row in mf6_mass_balance_rows if row.get("model") == "__DOMAIN__"),
        None,
    )
    if domain is None or not domain.get("complete"):
        summary["mass_balance_m3"] = {
            "available": False,
            "reason": None if domain is None else domain.get("reason"),
        }
        return

    vic_net = float(summary["exchange_m3"]["vic"]["net_m3"])
    api_net = float(domain["api_volume_m3"])
    storage_budget = float(domain["storage_budget_volume_m3"])
    storage_state_change = float(domain["storage_state_change_m3"])
    other = float(domain["other_nonstorage_volume_m3"])
    cbc_residual = float(domain["cbc_budget_residual_m3"])

    cell_budget = summary.get("cell_budget_m3") or {}
    head_storage = cell_budget.get("storage_change")

    summary["mass_balance_m3"] = {
        "available": True,
        "vic_interface_net_m3": vic_net,
        "mf6_api_net_m3": api_net,
        "mf6_other_nonstorage_net_m3": other,
        "mf6_storage_budget_net_m3": storage_budget,
        "mf6_storage_state_change_m3": storage_state_change,
        "vic_to_api_residual_m3": api_net - vic_net,
        "mf6_cbc_residual_m3": cbc_residual,
        "end_to_end_residual_m3": storage_state_change - (vic_net + other),
        "head_derived_storage_change_m3": head_storage,
        "head_vs_cbc_storage_residual_m3": (
            None if head_storage is None else float(head_storage) - storage_state_change
        ),
    }
