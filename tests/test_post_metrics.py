from types import SimpleNamespace

import numpy as np
import pytest

from vicmf6.exchange import ExchangeTable
from vicmf6.postprocess.loaders.mf6_metadata import read_budget_step_durations
from vicmf6.postprocess.metrics import (
    build_vic_exchange_tables,
)
from vicmf6.postprocess.metrics.exchange import _aggregate_api_volume
from vicmf6.postprocess.metrics.mass_balance import build_mf6_mass_balance_table


def _exchange_records():
    return [
        dict(
            vic_id=cell,
            vic_row=0,
            vic_col=column,
            vic_area_m2=100.0,
            mf6_model="GW",
            mf6_node=column + 1,
            overlap_area_m2=100.0,
        )
        for column, cell in enumerate(("A", "B"))
    ]


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


@pytest.mark.parametrize(
    "lateral_available,other_stresses", [(True, ()), (False, ()), (True, ("WEL",))]
)
def test_connected_confined_cell_budget_uses_full_storage_geometry(
    lateral_available, other_stresses, tmp_path
):
    from contextlib import ExitStack
    from datetime import datetime

    from vicmf6.postprocess.metrics import iter_cell_budget_rows
    from vicmf6.postprocess.stored_rows import StoredRows
    from vicmf6.postprocess.types import Mf6Geometry, Mf6HeadSeries

    config = SimpleNamespace(coupling=SimpleNamespace(start_time=datetime(2000, 1, 1)))
    series = Mf6HeadSeries(
        model_name="GW",
        times_days=np.array([0.0, 1.0 + 5e-10, 2.0]),
        read_heads=np.array([[0.0], [1.0], [3.0]]).__getitem__,
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
        dict(
            step=step,
            start=f"2000-01-0{step}T00:00:00",
            end=f"2000-01-0{step + 1}T00:00:00",
        )
        for step in (1, 2)
    ]
    boundary = [
        dict(step=step, model="GW", node=1, target_volume_m3=volume)
        for step, volume in ((1, 0.5), (2, 2.5))
    ]
    lateral = [
        dict(model="GW", node=1, time_days=time, flowja_row_net_volume_m3=volume)
        for time, volume in ((0.25, 0.2), (1.0, 0.3), (1.5, -0.2), (2.0, -0.3))
    ]
    with ExitStack() as resources:
        stored = StoredRows(tmp_path / "boundaries", resources)
        stored.extend(boundary)
        independent_reader = iter(stored)
        assert next(independent_reader) == boundary[0]
        rows = list(
            iter_cell_budget_rows(
                config,
                windows,
                {"GW": series},
                {"GW": geometry},
                stored,
                lateral,
                budget_data={
                    "GW": {
                        "complete": True,
                        "api_record_name": "API",
                        "record_names": ("API", "STO-SS", "FLOW-JA-FACE")
                        + other_stresses,
                        "lateral": {"available": lateral_available},
                    }
                },
            )
        )
        assert list(independent_reader) == boundary[1:]
    if not lateral_available or other_stresses:
        assert rows == []
        return
    assert [row["storage_change_m3"] for row in rows] == [1.0, 2.0]
    assert [row["cell_budget_residual_m3"] for row in rows] == [0.0, 0.0]


def test_postprocessing_keeps_each_models_own_geometry_and_grid(tmp_path):
    flopy = pytest.importorskip("flopy")
    from vicmf6.postprocess.loaders.mf6_geometry import load_mf6_geometries
    from vicmf6.postprocess.loaders.mf6_metadata import find_model_binary_grid

    simulation = flopy.mf6.MFSimulation(sim_ws=tmp_path)
    records = []
    for index, name in enumerate(("a", "b")):
        model = flopy.mf6.ModflowGwf(simulation, modelname=name, model_rel_path=name)
        flopy.mf6.ModflowGwfdisu(
            model,
            nodes=1,
            nja=1,
            top=10,
            bot=0,
            area=100,
            iac=[1],
            ja=[0],
            ihc=[0],
            cl12=[1],
            hwva=[1],
            grb_filerecord="custom-b.grb" if index else None,
        )
        grid_path = tmp_path / ("custom-b.grb" if index else "a/a.disu.grb")
        grid_path.parent.mkdir(exist_ok=True)
        grid_path.touch()
        assert find_model_binary_grid(model, tmp_path) == grid_path
        records.append(
            dict(
                mf6_model=name.upper(),
                mf6_node=1,
                vic_lon=40 + index,
                vic_lat=10,
                overlap_area_m2=100,
            )
        )

    archive = tmp_path / "reference"
    archive.mkdir()
    (archive / "mf6_cells.csv").write_text(
        "mf6_id,center_lon,center_lat,row,col,rectangle_area_m2\n0,999,999,0,0,999\n"
    )
    config = SimpleNamespace(
        coupling=SimpleNamespace(exchange_table=tmp_path / "exchange.csv"),
        mf6_source=SimpleNamespace(
            models=[SimpleNamespace(name=name) for name in ("a", "b")]
        ),
    )
    geometries = load_mf6_geometries(config, simulation, records)
    for index, name in enumerate(("A", "B")):
        np.testing.assert_array_equal(geometries[name].x, [40 + index])
        np.testing.assert_array_equal(geometries[name].area_m2, [100])
        assert geometries[name].source == "overlap_weighted_vic_centroids"
    (tmp_path / "a/a.disu.grb").unlink()
    assert find_model_binary_grid(simulation.get_model("a"), tmp_path) is None
    simulation.get_model("b").disu.nogrb.set_data(True)
    assert find_model_binary_grid(simulation.get_model("b"), tmp_path) is None


def test_sparse_saved_budget_never_invents_unsaved_step_volume():
    cbc = SimpleNamespace(
        recordarray=np.array([(2.0, 1.0)], dtype=[("totim", "f8"), ("delt", "f8")])
    )
    times = np.array([2.0])
    budget = dict(
        available=True,
        times_days=times,
        dt_days=read_budget_step_durations(cbc, times),
        read_api_rates={2.0: np.array([10.0])}.get,
    )
    np.testing.assert_array_equal(_aggregate_api_volume(budget, 1.0, 2.0, 1), [10.0])
    assert _aggregate_api_volume(budget, 0.0, 2.0, 1) is None


def test_empty_budget_cannot_establish_mass_closure():
    rows = build_mf6_mass_balance_table(
        {
            "GW": dict(
                available=True, complete=True, api_record_name="API", aggregate_terms=[]
            )
        }
    )
    assert not rows[-1]["complete"]
    assert rows[-1]["cbc_budget_residual_m3"] is None
    assert not build_mf6_mass_balance_table({})[-1]["complete"]
