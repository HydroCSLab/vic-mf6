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
