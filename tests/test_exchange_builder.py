"""Pure-unit tests for exchange-table preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from vicmf6.preprocess.exchange_builder.mf6_geometry import (
    _selected_surface_nodes,
)
from vicmf6.preprocess.exchange_builder.vic_geometry import (
    _load_vic_soil_base,
)


def test_disu_surface_selection_collapses_vertical_footprints_before_polygons():
    class Array:
        def __init__(self, value):
            self.array = value

    dtype = [
        ("icell2d", "i8"),
        ("xc", "f8"),
        ("yc", "f8"),
        ("ncvert", "i8"),
        ("icvert_0", "i8"),
        ("icvert_1", "i8"),
        ("icvert_2", "i8"),
        ("icvert_3", "i8"),
    ]
    cell2d = np.array(
        [
            (0, 0.5, 0.5, 4, 0, 1, 2, 3),
            (1, 1.5, 0.5, 4, 1, 4, 5, 2),
            (2, 0.5, 0.5, 4, 2, 3, 0, 1),
            (3, 1.5, 0.5, 4, 2, 5, 4, 1),
        ],
        dtype=dtype,
    )

    disu = SimpleNamespace(
        nodes=Array(np.array(4)),
        idomain=Array(np.ones(4, dtype=int)),
        top=Array(np.array([10.0, 11.0, 5.0, 6.0])),
        cell2d=Array(cell2d),
    )
    assert _selected_surface_nodes(SimpleNamespace(disu=disu), "DISU") == [0, 1]


def test_spatial_vic_soil_base_uses_all_soil_layers(tmp_path: Path):
    parameter_file = tmp_path / "parameters.nc"
    with Dataset(parameter_file, "w") as dataset:
        dataset.createDimension("soil_layer", 3)
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 2)
        elevation = dataset.createVariable("elev", "f8", ("y", "x"))
        depth = dataset.createVariable("depth", "f8", ("soil_layer", "y", "x"))
        elevation[:] = [[100.0, 110.0], [120.0, 130.0]]
        depth[:] = np.asarray([0.1, 0.2, 0.7])[:, None, None]

    interface, info = _load_vic_soil_base(parameter_file, (2, 2), Dataset)

    np.testing.assert_allclose(interface, [[99.0, 109.0], [119.0, 129.0]])
    assert info["soil_layers"] == 3
    assert info["soil_depth_m"]["min"] == 1.0
    assert info["soil_depth_m"]["max"] == 1.0


def test_spatial_vic_soil_base_ignores_masked_parameter_fill(tmp_path: Path):
    parameter_file = tmp_path / "parameters.nc"
    with Dataset(parameter_file, "w") as dataset:
        dataset.createDimension("soil_layer", 1)
        dataset.createDimension("y", 1)
        dataset.createDimension("x", 2)
        dataset.createVariable("elev", "f8", ("y", "x"))[:] = [[100.0, -999.0]]
        dataset.createVariable("depth", "f8", ("soil_layer", "y", "x"))[:] = [
            [[1.0, -999.0]]
        ]

    interface, info = _load_vic_soil_base(
        parameter_file, (1, 2), Dataset, active=np.array([[True, False]])
    )

    assert interface[0, 0] == 99.0
    assert info["soil_depth_m"]["min"] == 1.0


def test_zero_elevation_disu_cell_is_the_surface(monkeypatch, tmp_path):
    flopy = pytest.importorskip("flopy")
    pytest.importorskip("shapely")
    from vicmf6.preprocess.exchange_builder.mf6_geometry import load_mf6_cells

    namefile = tmp_path / "mfsim.nam"
    namefile.touch()
    dtype = [("icell2d", "i8"), ("xc", "f8"), ("yc", "f8"), ("ncvert", "i8")]
    dtype.extend((f"icvert_{index}", "i8") for index in range(4))
    cells = np.array(
        [(0, 0.5, 0.5, 4, 0, 1, 2, 3), (1, 0.5, 0.5, 4, 4, 5, 6, 7)], dtype=dtype
    )
    disu = SimpleNamespace(
        **{
            key: SimpleNamespace(array=value)
            for key, value in {
                "nodes": 2,
                "idomain": np.ones(2),
                "top": np.array([0.0, -1.0]),
                "bot": np.array([-1.0, -2.0]),
                "area": np.ones(2),
                "cell2d": cells,
            }.items()
        }
    )
    grid = SimpleNamespace(
        get_cell_vertices=lambda node: [(0, 0), (0, 1), (1, 1), (1, 0)]
    )
    model = SimpleNamespace(
        model_type="gwf6",
        disu=disu,
        modelgrid=grid,
        get_package=lambda name: disu if name == "disu" else None,
    )
    simulation = SimpleNamespace(model_names=["GW"], get_model=lambda name: model)
    monkeypatch.setattr(flopy.mf6.MFSimulation, "load", lambda **kwargs: simulation)
    surface, _ = load_mf6_cells(namefile, mf6_crs="EPSG:5070")
    assert len(surface) == 1
    assert surface[0].node == 1
