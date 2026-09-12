from datetime import datetime

import pytest

from vicmf6.schedule import build_windows, vic_record_count


def test_ten_day_half_open_calendar_builds_ten_daily_windows() -> None:
    windows = build_windows(
        datetime(1949, 1, 1),
        datetime(1949, 1, 11),
        1.0,
    )

    assert len(windows) == 10
    assert windows[0].start == datetime(1949, 1, 1)
    assert windows[-1].end == datetime(1949, 1, 11)


def test_twelve_hour_window_has_twelve_hourly_vic_records() -> None:
    assert vic_record_count(24, 0.5) == 12


def test_nonintegral_vic_window_is_rejected() -> None:
    with pytest.raises(Exception, match="integer number of VIC model steps"):
        vic_record_count(24, 0.1)


def test_single_day_half_open_calendar_builds_one_window() -> None:
    windows = build_windows(
        datetime(1949, 1, 1),
        datetime(1949, 1, 2),
        1.0,
    )

    assert len(windows) == 1
    assert windows[0].start == datetime(1949, 1, 1)
    assert windows[0].end == datetime(1949, 1, 2)
