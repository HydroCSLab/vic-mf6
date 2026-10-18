import pytest

from vicmf6.errors import ExchangeTableError
from vicmf6.exchange import ExchangeTable


def test_duplicate_overlap_relationship_is_rejected() -> None:
    row = _overlap("A")
    with pytest.raises(ExchangeTableError, match="duplicate"):
        ExchangeTable.from_records([row, row])


def test_incomplete_vic_coverage_is_rejected_when_required() -> None:
    with pytest.raises(ExchangeTableError, match="does not reproduce"):
        ExchangeTable.from_records([_overlap("A", area=75)])


def _overlap(cell_id, area=100):
    return dict(
        vic_id=cell_id,
        vic_row=0,
        vic_col=0,
        vic_area_m2=100.0,
        mf6_model="GW",
        mf6_node=1,
        overlap_area_m2=area,
    )


def test_duplicate_physical_vic_cell_cannot_double_the_source():
    with pytest.raises(ExchangeTableError, match="unique vic_id"):
        ExchangeTable.from_records([_overlap("a"), _overlap("b")])
