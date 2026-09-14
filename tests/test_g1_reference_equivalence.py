import numpy as np


def test_scheidegger_and_head_forms_are_algebraically_equivalent() -> None:
    # this is the G1 algebra-to-code regression. it does not duplicate the VIC
    # runtime; it protects the sign and geometry identity documented by G1.
    psi_bot_m = np.array([-85.0, -105.328, -250.0])
    z_ref_m = -3.0
    gap_m = 105.328
    z_wt_m = gap_m + z_ref_m
    groundwater_head_m = -z_wt_m
    exchange_length_m = z_wt_m - z_ref_m
    ka_m_per_day = np.array([0.001, 0.002, 0.003])

    scheidegger = -ka_m_per_day * (-z_wt_m - (psi_bot_m - z_ref_m)) / (z_wt_m - z_ref_m)
    soil_head_m = psi_bot_m - z_ref_m
    head_form = ka_m_per_day / exchange_length_m * (soil_head_m - groundwater_head_m)

    np.testing.assert_array_equal(scheidegger, head_form)
