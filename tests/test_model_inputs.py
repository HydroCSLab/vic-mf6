from pathlib import Path

import pytest

from vicmf6.errors import ConfigurationError
from vicmf6.model_inputs import parse_mf6_simulation, parse_vic_global


def test_vic_global_discovers_paths_calendar_and_exchange_stream(
    tmp_path: Path,
) -> None:
    (tmp_path / "domain.nc").write_text("x", encoding="utf-8")
    (tmp_path / "params.nc").write_text("x", encoding="utf-8")
    global_file = tmp_path / "model.global"
    global_file.write_text(
        """DOMAIN domain.nc
FORCING1 forcing.
PARAMETERS params.nc
MODEL_STEPS_PER_DAY 24
STARTYEAR 1949
STARTMONTH 1
STARTDAY 1
NRECS 240
OUTFILE history
OUTVAR OUT_RUNOFF
OUTFILE groundwater
OUTVAR OUT_GW_EXCHANGE
""",
        encoding="utf-8",
    )

    metadata = parse_vic_global(global_file)

    assert metadata.domain_file == (tmp_path / "domain.nc").resolve()
    assert metadata.parameters_file == (tmp_path / "params.nc").resolve()
    assert metadata.model_steps_per_day == 24
    assert metadata.nrecs == 240
    assert metadata.duration_days == 10.0
    assert metadata.exchange_output_prefix == "groundwater"
    assert metadata.end_time.isoformat() == "1949-01-11T00:00:00"


def test_vic_global_discovers_run_length_from_inclusive_end_date(
    tmp_path: Path,
) -> None:
    global_file = tmp_path / "model.global"
    global_file.write_text(
        """DOMAIN domain.nc
FORCING1 forcing.
PARAMETERS params.nc
MODEL_STEPS_PER_DAY 24
STARTYEAR 1949
STARTMONTH 1
STARTDAY 1
ENDYEAR 1949
ENDMONTH 1
ENDDAY 10
OUTFILE groundwater
OUTVAR OUT_GW_EXCHANGE
""",
        encoding="utf-8",
    )

    metadata = parse_vic_global(global_file)

    assert metadata.nrecs == 240
    assert metadata.duration_days == 10.0
    assert metadata.end_time.isoformat() == "1949-01-11T00:00:00"


def test_vic_global_rejects_both_nrecs_and_end_date(tmp_path: Path) -> None:
    global_file = tmp_path / "model.global"
    global_file.write_text(
        """DOMAIN domain.nc
FORCING1 forcing.
PARAMETERS params.nc
MODEL_STEPS_PER_DAY 24
STARTYEAR 1949
STARTMONTH 1
STARTDAY 1
NRECS 240
ENDYEAR 1949
ENDMONTH 1
ENDDAY 10
OUTFILE groundwater
OUTVAR OUT_GW_EXCHANGE
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="both NRECS and ENDYEAR"):
        parse_vic_global(global_file)


def test_vic_global_rejects_ambiguous_exchange_stream(tmp_path: Path) -> None:
    global_file = tmp_path / "model.global"
    global_file.write_text(
        """DOMAIN domain.nc
FORCING1 forcing.
PARAMETERS params.nc
MODEL_STEPS_PER_DAY 24
STARTYEAR 1949
STARTMONTH 1
STARTDAY 1
NRECS 24
OUTFILE a
OUTVAR OUT_GW_EXCHANGE
OUTFILE b
OUTVAR OUT_GW_EXCHANGE
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="multiple VIC OUTFILE streams"):
        parse_vic_global(global_file)


def test_mf6_input_discovers_model_api_solution_and_tdis(tmp_path: Path) -> None:
    (tmp_path / "mfsim.nam").write_text(
        """BEGIN TIMING
  TDIS6 time.tdis
END TIMING
BEGIN MODELS
  GWF6 gw.nam GW
END MODELS
BEGIN EXCHANGES
END EXCHANGES
BEGIN SOLUTIONGROUP 3
  IMS6 gw.ims GW
END SOLUTIONGROUP
""",
        encoding="utf-8",
    )
    (tmp_path / "gw.nam").write_text(
        """BEGIN PACKAGES
  API6 gw.api COUPLER
END PACKAGES
""",
        encoding="utf-8",
    )
    (tmp_path / "time.tdis").write_text(
        """BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS
BEGIN DIMENSIONS
  NPER 2
END DIMENSIONS
BEGIN PERIODDATA
  2 2 1
  3 3 1
END PERIODDATA
""",
        encoding="utf-8",
    )

    metadata = parse_mf6_simulation(tmp_path / "mfsim.nam")

    assert metadata.time_units == "DAYS"
    assert metadata.total_time_days == 5.0
    assert metadata.time_step_boundaries_days() == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert metadata.models[0].name == "GW"
    assert metadata.models[0].api_package == "COUPLER"
    assert metadata.models[0].solution_id == 3
