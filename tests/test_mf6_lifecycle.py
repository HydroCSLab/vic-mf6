from datetime import datetime
from pathlib import Path

import numpy as np

from vicmf6.config import Mf6Config
from vicmf6.mf6 import Mf6Runtime


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
        }

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
        return self.values[address]


def test_api_simvals_are_sampled_after_finalize_solve() -> None:
    config = Mf6Config(
        workspace=Path("."),
        library=Path("libmf6.so"),
        simulation_namefile="mfsim.nam",
        start_time=datetime(1949, 1, 1),
        api_packages={"H8A": "VICAPI"},
        solution_ids={"H8A": 1},
        total_time_days=1.0,
        time_step_boundaries_days=(1.0,),
        max_solve_iterations=10,
    )
    runtime = Mf6Runtime(
        config,
        model_name="H8A",
        coupled_nodes=[1],
        logger=object(),
    )
    runtime.xmi = _FakeXmi()
    runtime._head_address = "X"
    runtime._api_addresses = {
        "NBOUND": "NBOUND",
        "NODELIST": "NODELIST",
        "HCOF": "HCOF",
        "RHS": "RHS",
        "SIMVALS": "SIMVALS",
    }

    result = runtime.advance_to(
        1.0,
        np.array([2.0], dtype=np.float64),
        api_tolerance_m3_per_day=1.0e-12,
    )

    assert result.maximum_api_error_m3_per_day == 0.0
    assert result.applied.net_m3 == 2.0
    assert runtime.current_time_days() == 1.0
