import pytest

from vicmf6.errors import Mf6RuntimeError
from vicmf6.mf6 import _next_tdis_time_step_days


def test_next_tdis_step_uses_model_schedule() -> None:
    boundaries = (0.25, 0.5, 1.0, 2.0)

    assert _next_tdis_time_step_days(
        boundaries, 0.0, tolerance_days=1.0e-10
    ) == pytest.approx(0.25)
    assert _next_tdis_time_step_days(
        boundaries, 0.25, tolerance_days=1.0e-10
    ) == pytest.approx(0.25)
    assert _next_tdis_time_step_days(
        boundaries, 0.5, tolerance_days=1.0e-10
    ) == pytest.approx(0.5)
    assert _next_tdis_time_step_days(
        boundaries, 1.0, tolerance_days=1.0e-10
    ) == pytest.approx(1.0)


def test_next_tdis_step_rejects_time_between_boundaries() -> None:
    with pytest.raises(Mf6RuntimeError, match="does not coincide"):
        _next_tdis_time_step_days(
            (1.0, 2.0, 3.0),
            1.5,
            tolerance_days=1.0e-10,
        )


def test_next_tdis_step_rejects_end_of_simulation() -> None:
    with pytest.raises(Mf6RuntimeError, match="final TDIS boundary"):
        _next_tdis_time_step_days(
            (1.0, 2.0, 3.0),
            3.0,
            tolerance_days=1.0e-10,
        )


@pytest.mark.parametrize("boundaries", [(), (1, 1), (2, 1), (0, 1), (float("nan"),)])
def test_persistent_schedule_rejects_invalid_boundaries(boundaries):
    from vicmf6.mf6.time_steps import TdisSchedule

    with pytest.raises(Mf6RuntimeError):
        TdisSchedule(boundaries)


def test_persistent_schedule_follows_irregular_substeps_with_roundoff():
    import numpy as np

    from vicmf6.mf6.time_steps import TdisSchedule

    # A long geometrically varying schedule exercises both early and late
    # boundaries, where an implementation assuming uniform dt would fail.
    boundaries = np.cumsum(np.geomspace(0.01, 2.0, 1000))
    schedule = TdisSchedule(boundaries)
    previous = 0.0
    for boundary in boundaries:
        for offset in (0.0, -2e-12, 2e-12):
            assert schedule.next_step_days(
                previous + offset, tolerance_days=1e-10
            ) == pytest.approx(boundary - previous, abs=1e-14)
        previous = boundary
