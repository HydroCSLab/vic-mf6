from types import SimpleNamespace

import numpy as np

from vicmf6.exchange import ExchangeTable
from vicmf6.post.metrics import (
    build_mapping_tables,
    build_vic_exchange_tables,
)


def _exchange_records():
    return [
        {
            "vic_id": "A",
            "vic_row": 0,
            "vic_col": 0,
            "vic_area_m2": 100.0,
            "vic_lat": 1.0,
            "vic_lon": 2.0,
            "mf6_model": "GW",
            "mf6_node": 1,
            "mf6_area_m2": 150.0,
            "overlap_area_m2": 100.0,
        },
        {
            "vic_id": "B",
            "vic_row": 0,
            "vic_col": 1,
            "vic_area_m2": 100.0,
            "vic_lat": 1.0,
            "vic_lon": 3.0,
            "mf6_model": "GW",
            "mf6_node": 2,
            "mf6_area_m2": 150.0,
            "overlap_area_m2": 100.0,
        },
    ]


def test_mapping_tables_preserve_explicit_identities():
    vic, mf6 = build_mapping_tables(_exchange_records())
    assert len(vic) == 2
    assert len(mf6) == 2
    assert vic[0]["coverage_fraction"] == 1.0
    assert mf6[0]["mf6_node"] == 1


def test_vic_exchange_table_reconstructs_runtime_volume():
    table = ExchangeTable.from_records(_exchange_records())
    diagnostics = [
        {
            "step": 1,
            "start": "2000-01-01T00:00:00",
            "end": "2000-01-02T00:00:00",
            "vic_net_m3": -0.1,
        }
    ]
    fields = [
        {
            "field_mm": np.array([[1.0, -2.0]], dtype=float),
            "water_error_max_mm": 0.0,
            "files": (),
        }
    ]
    config = SimpleNamespace(
        coupling=SimpleNamespace(
            conservation_absolute_tolerance_m3=1.0e-12,
            conservation_relative_tolerance=1.0e-12,
        )
    )

    cells, summary = build_vic_exchange_tables(config, table, diagnostics, fields)
    assert len(cells) == 2
    assert summary[0]["positive_m3"] == 0.1
    assert summary[0]["negative_m3"] == -0.2
    assert summary[0]["net_m3"] == -0.1


def test_connected_confined_cell_budget_uses_full_storage_geometry():
    from datetime import datetime

    from vicmf6.post.metrics import build_cell_budget_table
    from vicmf6.post.types import Mf6Geometry, Mf6HeadSeries

    config = SimpleNamespace(coupling=SimpleNamespace(start_time=datetime(2000, 1, 1)))
    series = Mf6HeadSeries(
        model_name="GW",
        times_days=np.array([0.0, 1.0]),
        heads_m=np.array([[0.0], [1.0]]),
        initial_heads_m=np.array([0.0]),
    )
    geometry = Mf6Geometry(
        model_name="GW",
        node=np.array([1]),
        area_m2=np.array([100.0]),
        top_m=np.array([10.0]),
        bottom_m=np.array([0.0]),
        specific_storage_per_m=np.array([0.001]),
        specific_yield=np.array([0.1]),
        iconvert=np.array([0]),
        x=None,
        y=None,
        row=None,
        col=None,
        vertices=None,
        source="test",
    )
    windows = [
        {
            "step": 1,
            "start": "2000-01-01T00:00:00",
            "end": "2000-01-02T00:00:00",
        }
    ]
    boundary = [
        {
            "step": 1,
            "model": "GW",
            "node": 1,
            "target_volume_m3": 1.0,
        }
    ]
    rows = build_cell_budget_table(
        config,
        windows,
        {"GW": series},
        {"GW": geometry},
        boundary,
        [],
    )
    assert len(rows) == 1
    assert rows[0]["storage_change_m3"] == 1.0
    assert rows[0]["cell_budget_residual_m3"] == 0.0
