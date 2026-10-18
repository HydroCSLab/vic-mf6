import numpy as np

from vicmf6.exchange import ExchangeTable


def test_absolute_head_is_transformed_against_shared_interface_elevation() -> None:
    table = ExchangeTable.from_records(
        [
            {
                "vic_id": "A",
                "vic_row": 0,
                "vic_col": 0,
                "vic_area_m2": 1.0,
                "vic_interface_elevation_m": 1500.0,
                "mf6_model": "GW",
                "mf6_node": 1,
                "overlap_area_m2": 1.0,
            }
        ]
    )

    transformed = table.transform_head_for_vic(
        np.array([1425.5]), "pressure_head_from_interface_elevation"
    )

    np.testing.assert_array_equal(transformed, np.array([-74.5]))


def test_absolute_head_uses_overlap_weighted_mf6_interface_elevation() -> None:
    table = ExchangeTable.from_records(
        [
            {
                "vic_id": "A",
                "vic_row": 0,
                "vic_col": 0,
                "vic_area_m2": 4.0,
                "vic_interface_elevation_m": 1500.0,
                "mf6_model": "GW",
                "mf6_node": node,
                "mf6_interface_elevation_m": elevation,
                "overlap_area_m2": area,
            }
            for node, elevation, area in ((1, 1400.0, 1.0), (2, 1200.0, 3.0))
        ]
    )

    transformed = table.transform_head_for_vic(
        np.array([1180.0]), "pressure_head_from_mf6_interface_elevation"
    )

    np.testing.assert_array_equal(transformed, np.array([-70.0]))
