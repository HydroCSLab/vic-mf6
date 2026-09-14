import pytest

from vicmf6.errors import ExchangeTableError
from vicmf6.exchange import ExchangeTable


def test_duplicate_overlap_relationship_is_rejected() -> None:
    row = {
        "vic_id": "A",
        "vic_row": 0,
        "vic_col": 0,
        "vic_area_m2": 2.0,
        "mf6_model": "GW",
        "mf6_node": 1,
        "overlap_area_m2": 1.0,
    }

    with pytest.raises(ExchangeTableError, match="duplicate"):
        ExchangeTable.from_records([row, row])


def test_incomplete_vic_coverage_is_rejected_when_required() -> None:
    with pytest.raises(ExchangeTableError, match="does not reproduce"):
        ExchangeTable.from_records(
            [
                {
                    "vic_id": "A",
                    "vic_row": 0,
                    "vic_col": 0,
                    "vic_area_m2": 4.0,
                    "mf6_model": "GW",
                    "mf6_node": 1,
                    "overlap_area_m2": 3.0,
                }
            ]
        )
