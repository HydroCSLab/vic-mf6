import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vicmf6.config import Mf6Config
from vicmf6.errors import Mf6RuntimeError
from vicmf6.mf6 import Mf6Runtime
from vicmf6.mf6.boundary import ApiFluxBoundary


class _FakeXmi:
    def __init__(self) -> None:
        self.current_time = 0.0
        self.time_step = 0.0
        self.values = {
            "X": np.array([-105.0], dtype=np.float64),
            "NBOUND": np.array([0], dtype=np.int32),
            "NODELIST": np.array([0], dtype=np.int32),
            "HCOF": np.array([0.0], dtype=np.float64),
            "RHS": np.array([0.0], dtype=np.float64),
            "SIMVALS": np.array([0.0], dtype=np.float64),
            "NODES": np.array([1]),
            "NODESUSER": np.array([1]),
        }

    def get_var_address(self, name, model_name, package_name=""):
        return name

    def get_current_time(self) -> float:
        return self.current_time

    def prepare_time_step(self, dt: float) -> None:
        self.time_step = float(dt)

    def get_time_step(self) -> float:
        return self.time_step

    def prepare_solve(self, solution_id: int) -> None:
        assert solution_id == 1

    def solve(self, solution_id: int) -> bool:
        assert solution_id == 1
        return True

    def finalize_solve(self, solution_id: int) -> None:
        assert solution_id == 1
        self.values["SIMVALS"][0] = -self.values["RHS"][0]

    def finalize_time_step(self) -> None:
        self.current_time += self.time_step

    def get_value_ptr(self, address: str) -> np.ndarray:
        return self.values[address.rsplit("/", 1)[-1]]

    def get_input_var_names(self):
        return [
            f"H8A/DIS/{name}"
            for name in ("NODES", "NODESUSER", "NODEUSER")
            if name in self.values
        ]

    get_output_var_names = get_input_var_names


def make_runtime(boundaries=(1.0,)) -> Mf6Runtime:
    config = Mf6Config(
        workspace=Path("."),
        library=Path("libmf6.so"),
        simulation_namefile="mfsim.nam",
        start_time=datetime(1949, 1, 1),
        api_packages={"H8A": "VICAPI"},
        solution_ids={"H8A": 1},
        total_time_days=boundaries[-1],
        time_step_boundaries_days=boundaries,
        max_solve_iterations=10,
    )
    runtime = Mf6Runtime(
        config,
        model_name="H8A",
        coupled_nodes=[1],
        logger=logging.getLogger(__name__),
    )
    runtime.xmi = _FakeXmi()
    runtime._head_address = "X"
    runtime.boundary = ApiFluxBoundary(
        runtime.xmi, "H8A", "VICAPI", runtime.coupled_nodes
    )

    return runtime


@pytest.mark.parametrize("volume", [2.0, -3.0])
def test_exchange_volume_is_integrated_across_native_substeps(volume):
    runtime = make_runtime((0.125, 0.375, 1.0, 1.5, 2.0))
    for target in (1.0, 2.0):
        result = runtime.advance_to(
            target, np.array([volume]), api_tolerance_m3_per_day=1e-12
        )
        assert result.applied.net_m3 == pytest.approx(volume)
        assert result.requested == result.applied
        assert runtime.current_time_days() == target
    assert result.nonlinear_iterations == 2


@pytest.mark.parametrize(
    "fault, message",
    [
        ("stalled_clock", "did not advance time"),
        ("wrong_step", "does not match TDIS"),
        ("nonconvergence", "did not converge"),
        ("wrong_rate", "API6 rate mismatch"),
    ],
)
def test_native_failures_raise_before_accepting_window(fault, message):
    runtime = make_runtime()
    if fault == "stalled_clock":
        runtime.xmi.finalize_time_step = lambda: None
    elif fault == "wrong_step":
        runtime.xmi.get_time_step = lambda: 0.5
    elif fault == "nonconvergence":
        runtime.xmi.solve = lambda solution_id: False
    elif fault == "wrong_rate":
        runtime.xmi.finalize_solve = lambda solution_id: None
    with pytest.raises(Mf6RuntimeError, match=message):
        runtime.advance_to(1.0, np.array([2.0]), api_tolerance_m3_per_day=1e-12)


def test_native_substep_cannot_cross_coupling_boundary():
    runtime = make_runtime((1.0, 2.0))
    with pytest.raises(Mf6RuntimeError, match="cross a coupling boundary"):
        runtime.advance_to(0.5, np.array([2.0]), api_tolerance_m3_per_day=1e-12)
    assert runtime.current_time_days() == 0.0


class _InitializationXmi(_FakeXmi):
    def __init__(self) -> None:
        super().__init__()
        self.sequential_calls = 0
        self.parallel_handles = []

    def initialize(self) -> None:
        self.sequential_calls += 1

    def initialize_mpi(self, handle: int) -> None:
        self.parallel_handles.append(handle)


@pytest.mark.parametrize("comm_size", [1, 2])
def test_initialization_selects_solver_path(monkeypatch, comm_size):
    wrapper = _InitializationXmi()
    monkeypatch.setitem(
        __import__("sys").modules,
        "xmipy",
        SimpleNamespace(XmiWrapper=lambda *args, **kwargs: wrapper),
    )
    runtime = make_runtime()
    runtime.xmi = None
    runtime._head_address = None
    runtime.boundary = None

    def communicator_from_handle(handle):
        assert handle == 17
        return SimpleNamespace(Get_size=lambda: comm_size)

    monkeypatch.setitem(
        __import__("sys").modules,
        "mpi4py",
        SimpleNamespace(
            MPI=SimpleNamespace(Comm=SimpleNamespace(f2py=communicator_from_handle))
        ),
    )
    runtime.initialize(17)

    assert wrapper.sequential_calls == (1 if comm_size == 1 else 0)
    assert wrapper.parallel_handles == ([] if comm_size == 1 else [17])
