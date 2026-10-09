import pytest

from vicmf6.errors import Mf6RuntimeError
from vicmf6.mf6.time_steps import TdisSchedule


def test_native_steps_follow_irregular_boundaries_with_clock_roundoff():
    schedule = TdisSchedule((0.25, 0.75, 1.0, 2.0))
    for current, expected in ((0.0, 0.25), (0.25, 0.5), (0.75, 0.25), (1.0, 1.0)):
        for offset in (-2e-12, 2e-12):
            assert schedule.next_step_days(
                current + offset, tolerance_days=1e-10
            ) == pytest.approx(expected)


def test_native_clock_between_original_boundaries_is_rejected():
    with pytest.raises(Mf6RuntimeError, match="does not coincide"):
        TdisSchedule((1.0, 2.0, 3.0)).next_step_days(1.5, tolerance_days=1e-10)
