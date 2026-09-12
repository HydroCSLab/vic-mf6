import numpy as np

from vicmf6.mf6 import api_rhs_from_rate


def test_api_rhs_sign_for_recharge_into_mf6() -> None:
    rhs = api_rhs_from_rate(np.array([0.2]))
    np.testing.assert_array_equal(rhs, np.array([-0.2]))


def test_api_rhs_sign_for_groundwater_to_vic_transfer() -> None:
    rhs = api_rhs_from_rate(np.array([-0.2]))
    np.testing.assert_array_equal(rhs, np.array([0.2]))
