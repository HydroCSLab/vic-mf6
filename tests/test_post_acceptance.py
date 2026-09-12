from datetime import datetime
from types import SimpleNamespace

from vicmf6.post.acceptance import evaluate_acceptance


def test_acceptance_checks_runtime_conservation_without_reference():
    config = SimpleNamespace(
        coupling=SimpleNamespace(
            start_time=datetime(2000, 1, 1),
            end_time=datetime(2000, 1, 3),
            interval_days=1.0,
            conservation_absolute_tolerance_m3=1.0e-6,
            conservation_relative_tolerance=1.0e-12,
            api_absolute_tolerance_m3_per_day=1.0e-6,
        )
    )
    rows = []
    for step in (1, 2):
        rows.append(
            {
                "step": step,
                "positive_mapping_error_m3": 0.0,
                "negative_mapping_error_m3": 0.0,
                "net_mapping_error_m3": 0.0,
                "aggregation_net_error_m3": 0.0,
                "positive_api_error_m3": 0.0,
                "negative_api_error_m3": 0.0,
                "net_api_error_m3": 0.0,
                "maximum_api_rate_error_m3_per_day": 0.0,
                "maximum_vic_water_error_mm": 1.0e-14,
            }
        )
    summary = {
        "exchange_m3": {
            "vic": {"positive_m3": 1.0, "negative_m3": -3.0, "net_m3": -2.0},
            "node_target": {"positive_m3": 0.5, "negative_m3": -2.5, "net_m3": -2.0},
            "api_applied": {"positive_m3": 0.5, "negative_m3": -2.5, "net_m3": -2.0},
        },
        "errors": {
            "maximum_lateral_pair_antisymmetry_m3_day": None,
            "maximum_cell_budget_residual_m3": None,
        },
        "mass_balance_m3": {
            "available": True,
            "vic_interface_net_m3": -2.0,
            "mf6_api_net_m3": -2.0,
            "mf6_other_nonstorage_net_m3": 0.0,
            "mf6_storage_budget_net_m3": 2.0,
            "mf6_storage_state_change_m3": -2.0,
            "mf6_cbc_residual_m3": 0.0,
            "end_to_end_residual_m3": 0.0,
        },
    }
    result = evaluate_acceptance(
        config,
        rows,
        summary,
        exchange_table_vic_cells=2,
        exchange_table_overlap_rows=2,
        mf6_cell_count=2,
        reference_path=None,
    )
    assert result.passed
