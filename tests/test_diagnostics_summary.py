import json
from datetime import datetime

import pytest

from vicmf6.diagnostics import DiagnosticsWriter, make_window_diagnostics
from vicmf6.exchange import SignedVolume
from vicmf6.schedule import CouplingWindow


def test_summary_reports_cumulative_cross_model_error(tmp_path) -> None:
    writer = DiagnosticsWriter(tmp_path)
    row = make_window_diagnostics(
        window=CouplingWindow(0, datetime(1949, 1, 1), datetime(1949, 1, 2)),
        vic=SignedVolume(2.0, -5.0, -3.0),
        mapped=SignedVolume(2.0, -5.0, -3.0),
        boundary_target=SignedVolume(1.5, -4.5, -3.0),
        applied=SignedVolume(1.5, -4.499999, -2.999999),
        head_min_m=-101.0,
        head_max_m=-99.0,
        head_mean_m=-100.0,
        maximum_api_rate_error_m3_per_day=1.0e-6,
        maximum_vic_water_error_mm=1.0e-14,
        lateral_domain_net_m3=0.0,
        lateral_gross_pair_m3=4.0,
        lateral_maximum_pair_antisymmetry_m3=0.0,
        nonlinear_iterations_max=2,
        vic_prepare_seconds=0.1,
        vic_seconds=1.0,
        mf6_seconds=0.5,
        mapping_seconds=0.01,
        mpi_seconds=0.02,
        io_seconds=0.03,
        window_seconds=1.7,
    )
    writer.append(row)
    writer.write_summary(total_wall_seconds=1.7)

    summary = json.loads((tmp_path / "run_summary.json").read_text())

    assert summary["cumulative_errors_m3"]["aggregation"]["net_m3"] == pytest.approx(
        0.0
    )
    assert summary["cumulative_errors_m3"]["api"]["net_m3"] == pytest.approx(1.0e-6)
    assert summary["cumulative_errors_m3"]["cross_model"]["net_m3"] == pytest.approx(
        1.0e-6
    )
    assert summary["relative_cross_model_net_error"] > 0.0
