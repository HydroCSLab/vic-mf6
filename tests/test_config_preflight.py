import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vicmf6.config import create_fresh_run_output_directories, load_config
from vicmf6.errors import ConfigurationError
from vicmf6.model_inputs.mf6_simulation import _parse_mf6_solution_groups
from vicmf6.model_inputs.records import _Mf6ModelEntry
from vicmf6.preflight import run_preflight


def _write_minimal_models(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    mf6 = tmp_path / "mf6"
    vic = tmp_path / "vic"
    mf6.mkdir()
    vic.mkdir()

    library = tmp_path / "libmf6.so"
    executable = tmp_path / "vic_image.exe"
    library.write_text("placeholder", encoding="utf-8")
    executable.write_text("placeholder", encoding="utf-8")

    (vic / "domain.nc").write_text("placeholder", encoding="utf-8")
    (vic / "params.nc").write_text("placeholder", encoding="utf-8")
    (vic / "forcing.1949.nc").write_text("placeholder", encoding="utf-8")
    global_file = vic / "model.global"
    global_file.write_text(
        """DOMAIN domain.nc
FORCING1 forcing.
PARAMETERS params.nc
MODEL_STEPS_PER_DAY 24
STARTYEAR 1949
STARTMONTH 1
STARTDAY 1
NRECS 240
OUTFILE fluxes
OUTVAR OUT_GW_EXCHANGE
""",
        encoding="utf-8",
    )

    (mf6 / "mfsim.nam").write_text(
        """BEGIN OPTIONS
END OPTIONS
TDIS6 time.tdis
BEGIN MODELS
  GWF6 gw.nam GW
END MODELS
BEGIN EXCHANGES
END EXCHANGES
BEGIN SOLUTIONGROUP 1
  IMS6 gw.ims GW
END SOLUTIONGROUP
""",
        encoding="utf-8",
    )
    (mf6 / "time.tdis").write_text(
        """BEGIN OPTIONS
  TIME_UNITS DAYS
END OPTIONS
BEGIN DIMENSIONS
  NPER 1
END DIMENSIONS
BEGIN PERIODDATA
  10 10 1
END PERIODDATA
""",
        encoding="utf-8",
    )
    (mf6 / "gw.nam").write_text(
        """BEGIN OPTIONS
END OPTIONS
BEGIN PACKAGES
  API6 gw.api VICAPI
END PACKAGES
""",
        encoding="utf-8",
    )
    (mf6 / "gw.api").write_text(
        """BEGIN DIMENSIONS
  MAXBOUND 1
END DIMENSIONS
""",
        encoding="utf-8",
    )

    exchange_table = tmp_path / "exchange.csv"
    exchange_table.write_text(
        "vic_id,vic_row,vic_col,vic_area_m2,vic_lat,vic_lon,mf6_model,mf6_node,overlap_area_m2\n"
        "A,0,0,10,48,-120,GW,1,10\n",
        encoding="utf-8",
    )
    return global_file, mf6 / "mfsim.nam", library, executable


@pytest.fixture
def minimal_config_path(tmp_path: Path) -> Path:
    global_file, namefile, library, executable = _write_minimal_models(tmp_path)
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "mf6": {"namefile": str(namefile), "library": str(library)},
                "vic": {"global_file": str(global_file), "executable": str(executable)},
                "coupling": {
                    "exchange_table": "exchange.csv",
                    "interval_days": 1,
                    "exchange_length_m": 100,
                    "exchange_conductivity_scale": 0.001,
                    "head_transform": "identity",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("vic", "mpi_processes", 1.5),
        ("vic", "omp_threads", True),
        ("vic", "spawn_timeout_seconds", float("inf")),
        ("mf6", "max_solve_iterations", 0),
        ("coupling", "conservation_relative_tolerance", float("nan")),
        ("coupling", "api_absolute_tolerance_m3_per_day", float("inf")),
    ],
)
def test_invalid_settings_fail_before_launch(
    minimal_config_path, section, field, value
):
    raw = yaml.safe_load(minimal_config_path.read_text())
    raw[section][field] = value
    minimal_config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=rf"{section}\.{field} must"):
        load_config(minimal_config_path)


def test_fresh_output_directories_preserve_prepared_model_inputs(minimal_config_path):
    config = load_config(minimal_config_path)
    config.run_directory.mkdir()
    prepared_input = config.run_directory / "prepared-model.nam"
    prepared_input.write_text("input", encoding="utf-8")
    create_fresh_run_output_directories(config)
    assert prepared_input.read_text() == "input"
    assert config.vic.outputs_directory.is_dir()
    assert config.vic.exchange_directory.is_dir()
    assert config.coupling.diagnostics_directory.is_dir()
    with pytest.raises(ConfigurationError, match="already exists"):
        create_fresh_run_output_directories(config)


def test_cli_rejects_existing_outputs_before_opening_logs_or_models(
    minimal_config_path, monkeypatch, capsys
):
    from vicmf6 import cli

    config = load_config(minimal_config_path)
    directory = config.coupling.diagnostics_directory
    directory.mkdir(parents=True)
    previous_output = directory / "previous-output"
    previous_output.write_bytes(b"keep this result")
    previous_paths = set(config.run_directory.rglob("*"))

    def abort(code):
        raise SystemExit(code)

    def unexpected_log_creation(**kwargs):
        pytest.fail("existing run must be rejected before logs are opened")

    world = SimpleNamespace(Get_rank=lambda: 0, Abort=abort)
    monkeypatch.setitem(
        sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=world))
    )
    monkeypatch.setattr("vicmf6.diagnostics.build_logger", unexpected_log_creation)
    with pytest.raises(SystemExit) as failure:
        cli.main(["run", "-c", str(minimal_config_path)])
    assert failure.value.code == 1
    assert (
        f"run output directory already exists: {directory}" in capsys.readouterr().err
    )
    assert set(config.run_directory.rglob("*")) == previous_paths
    assert previous_output.read_bytes() == b"keep this result"


def test_minimal_config_discovers_model_owned_metadata(minimal_config_path):
    config = load_config(minimal_config_path)
    summary = run_preflight(config)
    assert config.vic.parameters_file == minimal_config_path.parent / "vic/params.nc"
    assert config.vic.exchange_output_prefix == "fluxes"
    assert config.mf6.api_package_for("GW") == "VICAPI"
    assert config.mf6.solution_id_for("GW") == 1
    assert config.mf6.time_step_boundaries_days == tuple(float(i) for i in range(1, 11))
    assert summary["mpi"]["world_ranks_required"] == 2
    assert summary["coupling"]["windows"] == 10


def test_vic_and_mf6_duration_mismatch_is_rejected(minimal_config_path):
    tdis = minimal_config_path.parent / "mf6/time.tdis"
    tdis.write_text(tdis.read_text().replace("10 10 1", "9 9 1"))
    with pytest.raises(ConfigurationError, match="durations do not match"):
        load_config(minimal_config_path)


def test_coupling_boundary_must_match_mf6_time_step(minimal_config_path):
    tdis = minimal_config_path.parent / "mf6/time.tdis"
    tdis.write_text(tdis.read_text().replace("10 10 1", "10 5 1"))
    with pytest.raises(ConfigurationError, match="MF6 time-step boundary"):
        run_preflight(load_config(minimal_config_path))


def test_reserved_groundwater_environment_override_is_rejected(minimal_config_path):
    raw = yaml.safe_load(minimal_config_path.read_text())
    raw["vic"]["environment"] = {"VIC_GW_REFERENCE_DEPTH": "node"}
    minimal_config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigurationError, match="coupler-owned groundwater controls"):
        load_config(minimal_config_path)


def test_solver_ids_are_not_solution_group_numbers():
    models = tuple(_Mf6ModelEntry(name, Path(name)) for name in ("A", "B"))
    lines = [
        "BEGIN SOLUTIONGROUP 1",
        "IMS6 a.ims A",
        "IMS6 b.ims B",
        "END SOLUTIONGROUP",
    ]
    assert _parse_mf6_solution_groups(lines, models, Path("mfsim.nam")) == {
        "A": 1,
        "B": 2,
    }
    with pytest.raises(ConfigurationError, match="exactly one"):
        _parse_mf6_solution_groups(
            lines[:-1] + ["IMS6 extra.ims A"], models, Path("mfsim.nam")
        )


def test_ambiguous_vic_exchange_stream_is_rejected(minimal_config_path):
    global_file = minimal_config_path.parent / "vic/model.global"
    with global_file.open("a") as stream:
        stream.write("OUTFILE another_stream\nOUTVAR OUT_GW_EXCHANGE\n")
    with pytest.raises(ConfigurationError, match="multiple VIC OUTFILE streams"):
        load_config(minimal_config_path)
