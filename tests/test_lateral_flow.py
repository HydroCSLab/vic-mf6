import numpy as np

from vicmf6.mf6 import lateral_flow_diagnostics


def test_two_node_flowja_closes_pair_and_domain() -> None:
    ia = np.array([1, 3, 5])
    ja = np.array([1, 2, 1, 2])
    flowja_volume = np.array([0.0, 3.0, -3.0, 0.0])

    diagnostics = lateral_flow_diagnostics(flowja_volume, ia, ja)

    np.testing.assert_array_equal(diagnostics.net_by_node_m3, np.array([3.0, -3.0]))
    assert diagnostics.domain_net_m3 == 0.0
    assert diagnostics.gross_pair_volume_m3 == 3.0
    assert diagnostics.maximum_pair_antisymmetry_m3 == 0.0
