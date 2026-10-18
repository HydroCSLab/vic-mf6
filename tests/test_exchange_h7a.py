import numpy as np

from vicmf6.exchange import ExchangeTable, SignedVolume, assert_signed_volume_close


def _h7a_table() -> ExchangeTable:
    overlaps = [
        ("V0", 0, 0, 1, 1.5),
        ("V0", 0, 0, 2, 1.5),
        ("V0", 0, 0, 4, 0.5),
        ("V0", 0, 0, 5, 0.5),
        ("V1", 0, 1, 2, 1.5),
        ("V1", 0, 1, 3, 1.5),
        ("V1", 0, 1, 5, 0.5),
        ("V1", 0, 1, 6, 0.5),
        ("V2", 1, 0, 4, 2.0),
        ("V2", 1, 0, 5, 2.0),
        ("V3", 1, 1, 5, 2.0),
        ("V3", 1, 1, 6, 2.0),
    ]
    records = [
        {
            "vic_id": vic_id,
            "vic_row": row,
            "vic_col": col,
            "vic_area_m2": 4.0,
            "mf6_model": "GW",
            "mf6_node": node,
            "overlap_area_m2": area,
        }
        for vic_id, row, col, node, area in overlaps
    ]
    return ExchangeTable.from_records(records)


def test_h7a_signed_volume_mapping_matches_archived_values() -> None:
    table = _h7a_table()
    vic_exchange_mm = np.array([7.0, -4.0, 2.5, -1.5])

    mapped = table.map_vic_depth_to_model("GW", vic_exchange_mm, node_count=6)

    np.testing.assert_allclose(
        mapped.volume_by_node_m3,
        np.array([0.0105, 0.0045, -0.006, 0.0085, 0.0035, -0.005]),
        rtol=0.0,
        atol=1.0e-18,
    )
    expected = SignedVolume(positive_m3=0.038, negative_m3=-0.022, net_m3=0.016)
    assert_signed_volume_close(
        expected,
        mapped.signed_volume,
        absolute_tolerance_m3=1.0e-16,
        relative_tolerance=0.0,
        label="H7a",
    )
    # opposite-signed overlap contributions can land in the same MF6 node.
    # the API therefore receives one net boundary value per node, while the
    # overlap-scale diagnostics retain the original gross signed transfer.
    assert_signed_volume_close(
        SignedVolume(positive_m3=0.027, negative_m3=-0.011, net_m3=0.016),
        mapped.node_signed_volume,
        absolute_tolerance_m3=1.0e-16,
        relative_tolerance=0.0,
        label="H7a node aggregation",
    )


def test_h7a_reverse_head_mapping_matches_archived_values() -> None:
    table = _h7a_table()
    mf6_head_m = np.array([-100.0, -110.0, -120.0, -130.0, -115.0, -95.0])

    local = table.head_contribution_for_model("GW", mf6_head_m)
    mapped = table.finish_head_mapping(local.head_area_sum_m3, local.area_sum_m2)

    np.testing.assert_allclose(
        mapped,
        np.array([-109.375, -112.5, -122.5, -105.0]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_vic_id_is_not_used_as_flat_array_index() -> None:
    table = ExchangeTable.from_records(
        [
            {
                "vic_id": "900001",
                "vic_row": 1,
                "vic_col": 2,
                "vic_area_m2": 5.0,
                "mf6_model": "GW",
                "mf6_node": 1,
                "overlap_area_m2": 5.0,
            }
        ]
    )
    grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    extracted = table.extract_vic_values(grid)

    np.testing.assert_array_equal(extracted, np.array([6.0]))


def test_worker_partitions_preserve_global_vic_order_and_conservation():
    """Partitioning geometry must not change the transfer operator in either direction."""
    records = [
        {
            "vic_id": f"V{vic}",
            "vic_row": 0,
            "vic_col": vic,
            "vic_area_m2": 10.0,
            "mf6_model": model,
            "mf6_node": node,
            "overlap_area_m2": area,
        }
        for vic, model, node, area in (
            (0, "A", 1, 2.0),
            (0, "B", 1, 8.0),
            (1, "B", 1, 4.0),
            (1, "A", 2, 6.0),
            (2, "B", 2, 10.0),
        )
    ]
    table = ExchangeTable.from_records(records)
    depth = np.array([7.0, -4.0, 2.5])
    numerator = np.zeros(3)
    denominator = np.zeros(3)
    overlap_volumes = []
    for name, heads in [("A", [-10.0, -20.0]), ("B", [-30.0, -40.0])]:
        partition = table.for_model(name.lower())
        assert partition.vic_cells == table.vic_cells
        assert partition.model_names == (name,)
        expected = table.map_vic_depth_to_model(name, depth, node_count=2)
        actual = partition.map_vic_depth_to_model(name, depth, node_count=2)
        np.testing.assert_array_equal(
            actual.volume_by_node_m3, expected.volume_by_node_m3
        )
        assert actual.signed_volume == expected.signed_volume
        assert actual.node_signed_volume == expected.node_signed_volume
        overlap_volumes.append(actual.signed_volume)
        contribution = partition.head_contribution_for_model(name, heads)
        numerator += contribution.head_area_sum_m3
        denominator += contribution.area_sum_m2
    from vicmf6.exchange import combine_signed_volumes

    assert_signed_volume_close(
        table.source_volume_from_vic_depth(depth),
        combine_signed_volumes(overlap_volumes),
        absolute_tolerance_m3=1e-16,
        relative_tolerance=0.0,
        label="partitioned transfer",
    )
    np.testing.assert_array_equal(
        table.finish_head_mapping(numerator, denominator), [-26.0, -24.0, -40.0]
    )
