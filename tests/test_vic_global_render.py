from datetime import datetime
from pathlib import Path

from vicmf6.schedule import CouplingWindow
from vicmf6.vic import _render_global_parameter_file


def test_subdaily_global_file_owns_nrecs_restart_and_result_directory(
    tmp_path: Path,
) -> None:
    template = """MODEL_STEPS_PER_DAY 24\nSTARTYEAR 1900\nENDYEAR 1901\nNRECS 999\nRESULT_DIR old\nAGGFREQ NDAYS 1\nOUTFILE fluxes\nOUTVAR OUT_GW_EXCHANGE\n"""
    window = CouplingWindow(
        index=0,
        start=datetime(1949, 1, 1, 12, 0, 0),
        end=datetime(1949, 1, 1, 18, 0, 0),
    )

    rendered = _render_global_parameter_file(
        template,
        window=window,
        nrecs=6,
        output_directory=tmp_path / "output",
        state_prefix=tmp_path / "state" / "state",
        previous_state=tmp_path / "input-state.nc",
    )

    assert "STARTSEC    43200" in rendered
    assert "NRECS       6" in rendered
    assert "AGGFREQ     NHOURS 6" in rendered
    assert "STATESEC    64800" in rendered
    assert f"INIT_STATE  {tmp_path / 'input-state.nc'}" in rendered
    assert "NRECS 999" not in rendered
    assert "RESULT_DIR old" not in rendered
