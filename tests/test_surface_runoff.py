import numpy as np
import pytest

from vicmf6.errors import CouplingRuntimeError
from vicmf6.exchange import VicCell
from vicmf6.surface_runoff import SfrRunoffBoundary, SurfaceRunoffTable


def _cells():
    return (
        VicCell(0, "a", 0, 0, 1000.0, None, None, None),
        VicCell(1, "b", 0, 1, 2000.0, None, None, None),
    )


def _write(path, second_weight="1.0"):
    path.write_text(
        "vic_id,vic_row,vic_col,mf6_model,sfr_package,sfr_reach,weight\n"
        "a,0,0,GW,RIVER,1,0.25\n"
        "a,0,0,GW,RIVER,2,0.75\n"
        f"b,0,1,GW,RIVER,2,{second_weight}\n",
        encoding="utf-8",
    )


def test_runoff_mapping_preserves_volume(tmp_path):
    path = tmp_path / "runoff.csv"
    _write(path)
    table = SurfaceRunoffTable.from_csv(path, _cells())

    mapped = table.map_to_reaches(np.array([10.0, 20.0]))

    np.testing.assert_allclose(mapped.volume_by_reach_m3, [2.5, 47.5])
    assert mapped.signed_volume.net_m3 == pytest.approx(50.0)
    assert table.source_volume(np.array([10.0, 20.0])).net_m3 == pytest.approx(50.0)
    assert table.package_name == "RIVER"


def test_runoff_mapping_requires_complete_unit_weights(tmp_path):
    path = tmp_path / "runoff.csv"
    _write(path, second_weight="0.9")
    with pytest.raises(CouplingRuntimeError, match="sum to one"):
        SurfaceRunoffTable.from_csv(path, _cells())


class _Xmi:
    def __init__(self):
        self.runoff = np.full(3, -1.0)

    def get_var_address(self, name, model, package):
        assert name in {"RUNOFF", "SIMRUNOFF"}
        assert (model, package) == ("GW", "RIVER")
        return name

    def get_value_ptr(self, address):
        assert address in {"RUNOFF", "SIMRUNOFF"}
        return self.runoff


def test_sfr_boundary_writes_rates_and_clears_unmapped_reaches():
    xmi = _Xmi()
    boundary = SfrRunoffBoundary(xmi, "GW", "RIVER")
    boundary.write_rates(np.array([2.0, 3.0]))
    np.testing.assert_array_equal(xmi.runoff, [2.0, 3.0, 0.0])
