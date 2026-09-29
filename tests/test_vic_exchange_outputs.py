from types import SimpleNamespace

import numpy as np
import pytest

from vicmf6.errors import ExchangeTableError
from vicmf6.exchange import ExchangeTable
from vicmf6.post.loaders import load_vic_exchange_fields
from vicmf6.vic import _read_window_outputs


@pytest.mark.parametrize("reader", ["runtime", "postprocessing"])
@pytest.mark.parametrize(
    "exchange_values", [[np.nan, np.nan], [1.0, np.nan], [0.0, 0.0], [-2.0, 3.0]]
)
def test_missing_coupled_exchange_is_rejected_and_inactive_cells_remain_masked(
    tmp_path, reader, exchange_values
):
    netcdf = pytest.importorskip("netCDF4")
    directory = tmp_path / "window-0000"
    directory.mkdir()
    with netcdf.Dataset(directory / "fluxes.test.nc", "w") as dataset:
        for dimension, size in (("time", 2), ("y", 1), ("x", 2)):
            dataset.createDimension(dimension, size)
        variable = dataset.createVariable(
            "OUT_GW_EXCHANGE", "f8", ("time", "y", "x"), fill_value=-9999.0
        )
        values = np.full((2, 1, 2), np.nan)
        values[:, 0, 0] = exchange_values
        variable[:] = np.ma.masked_invalid(values)

    if reader == "runtime":
        field, _ = _read_window_outputs(
            directory, prefix="fluxes", exchange_variable="OUT_GW_EXCHANGE"
        )
    else:
        config = SimpleNamespace(
            vic=SimpleNamespace(
                outputs_directory=tmp_path,
                exchange_output_prefix="fluxes",
                exchange_variable="OUT_GW_EXCHANGE",
            )
        )
        field = load_vic_exchange_fields(config, expected_windows=1)[0]["field_mm"]

    table = ExchangeTable.from_records(
        [
            {
                "vic_id": "A",
                "vic_row": 0,
                "vic_col": 0,
                "vic_area_m2": 1.0,
                "mf6_model": "GW",
                "mf6_node": 1,
                "overlap_area_m2": 1.0,
            }
        ]
    )
    assert np.isnan(field[0, 1])
    if np.isnan(exchange_values).any():
        with pytest.raises(ExchangeTableError, match="non-finite"):
            table.extract_vic_values(field)
    else:
        np.testing.assert_array_equal(
            table.extract_vic_values(field), [sum(exchange_values)]
        )
