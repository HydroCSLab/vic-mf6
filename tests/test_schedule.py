from datetime import datetime

import pytest

from vicmf6.errors import ConfigurationError
from vicmf6.schedule import build_windows, vic_record_count


def test_windows_cover_the_period_without_losing_the_partial_final_day():
    windows = build_windows(datetime(2000, 1, 1), datetime(2000, 1, 2, 12), 1.0)
    assert [(window.start, window.end) for window in windows] == [
        (datetime(2000, 1, 1), datetime(2000, 1, 2)),
        (datetime(2000, 1, 2), datetime(2000, 1, 2, 12)),
    ]
    assert vic_record_count(24, windows[-1].duration_days) == 12


def test_nonintegral_vic_window_is_rejected():
    with pytest.raises(ConfigurationError, match="integer number of VIC model steps"):
        vic_record_count(24, 0.1)


def test_interval_that_rounds_to_zero_cannot_start_an_infinite_loop():
    with pytest.raises(ConfigurationError):
        build_windows(datetime(2000, 1, 1), datetime(2000, 1, 2), 1e-20)
