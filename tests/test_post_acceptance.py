from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from vicmf6.postprocess.acceptance import (
    check_domain_mass_balance,
    check_window_conservation,
    evaluate_acceptance,
)
from vicmf6.schedule import build_windows


def test_acceptance_rejects_nonconserving_model_outputs():
    config, rows = _conservation_fixture(2)
    summary = {
        "exchange_m3": {
            "vic": {"positive_m3": 0.0, "negative_m3": 0.0, "net_m3": 0.0},
            "node_target": {"positive_m3": 0.0, "negative_m3": 0.0, "net_m3": 0.0},
            "api_applied": {"positive_m3": 0.0, "negative_m3": 0.0, "net_m3": 0.0},
        },
        "errors": {
            "maximum_lateral_pair_antisymmetry_m3_day": None,
            "maximum_cell_budget_residual_m3": None,
        },
        "mass_balance_m3": {
            "available": True,
            "vic_interface_net_m3": 0.0,
            "mf6_api_net_m3": 0.0,
            "mf6_other_nonstorage_net_m3": 0.0,
            "mf6_storage_budget_net_m3": 0.0,
            "mf6_storage_state_change_m3": 0.0,
            "mf6_cbc_residual_m3": 0.0,
            "end_to_end_residual_m3": 0.0,
        },
    }
    result = evaluate_acceptance(config, rows, summary)
    assert result.passed

    rows[0]["applied_net_m3"] = 1.0
    assert not evaluate_acceptance(config, rows, summary).passed


def test_domain_budget_uses_solver_scale_tolerance() -> None:
    config, _ = _conservation_fixture()
    summary = {
        "mass_balance_m3": {
            "available": True,
            "vic_interface_net_m3": 401931.0,
            "mf6_api_net_m3": 401931.0,
            "mf6_other_nonstorage_net_m3": -168661.0,
            "mf6_storage_budget_net_m3": -233270.0,
            "mf6_storage_state_change_m3": 233270.0,
            "mf6_cbc_residual_m3": 0.0027,
            "end_to_end_residual_m3": -0.0027,
        },
    }

    assert all(check.passed for check in check_domain_mass_balance(config, summary))
    summary["mass_balance_m3"]["end_to_end_residual_m3"] = 10.0
    assert not all(check.passed for check in check_domain_mass_balance(config, summary))


def _conservation_fixture(days=1, interval=1):
    coupling = SimpleNamespace(
        start_time=datetime(2000, 1, 1),
        end_time=datetime(2000, 1, 1) + timedelta(days=days),
        interval_days=interval,
        conservation_absolute_tolerance_m3=1e-6,
        conservation_relative_tolerance=1e-12,
        api_absolute_tolerance_m3_per_day=1e-6,
    )
    rows = []
    for window in build_windows(coupling.start_time, coupling.end_time, interval):
        row = {
            "step": window.index + 1,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            **{
                f"{prefix}_{sign}_m3": 0.0
                for prefix in ("vic", "mapped", "boundary", "applied")
                for sign in ("positive", "negative", "net")
            },
        }
        row.update(
            {
                f"{sign}_{stage}_error_m3": 0.0
                for sign in ("positive", "negative", "net")
                for stage in ("mapping", "api")
            }
        )
        row.update(
            aggregation_net_error_m3=0.0,
            maximum_api_rate_error_m3_per_day=0.0,
            maximum_vic_water_error_mm=0.0,
        )
        rows.append(row)
    return SimpleNamespace(coupling=coupling), rows


def test_partial_final_window_is_accepted():
    config, rows = _conservation_fixture(5, 2)
    assert len(rows) == 3
    assert all(check.passed for check in check_window_conservation(config, rows))


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "nan",
        "missing_water",
        "wrong_time",
        "nan_source",
        "inconsistent_error",
    ],
)
def test_incomplete_or_invalid_window_evidence_fails(fault):
    config, rows = _conservation_fixture(2)
    if fault == "missing":
        del rows[1]["positive_mapping_error_m3"]
    elif fault == "nan":
        rows[1]["negative_mapping_error_m3"] = float("nan")
    elif fault == "missing_water":
        rows[1]["maximum_vic_water_error_mm"] = None
    elif fault == "nan_source":
        rows[1]["vic_positive_m3"] = float("nan")
    elif fault == "inconsistent_error":
        rows[1]["mapped_positive_m3"] = 100.0
    else:
        rows[1]["start"] = rows[0]["start"]
    assert not all(check.passed for check in check_window_conservation(config, rows))


def test_post_uses_relative_tolerance_per_transfer():
    config, rows = _conservation_fixture()
    row = rows[0]
    for prefix in ("vic", "mapped", "boundary", "applied"):
        row[f"{prefix}_positive_m3"] = row[f"{prefix}_net_m3"] = 1e9
    row["mapped_positive_m3"] += 0.0005
    row["positive_mapping_error_m3"] = 0.0005
    assert all(check.passed for check in check_window_conservation(config, rows))
    # The large positive transfer must not relax the independent negative check.
    row["negative_mapping_error_m3"] = 0.0005
    assert not all(check.passed for check in check_window_conservation(config, rows))
