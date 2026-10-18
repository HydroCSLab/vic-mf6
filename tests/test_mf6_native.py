"""Opt-in checks against a real library: VICMF6_MF6_LIBRARY=/path/to/libmf6.so."""

import logging
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vicmf6.config import Mf6Config
from vicmf6.errors import Mf6RuntimeError
from vicmf6.mf6 import Mf6Runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("VICMF6_MF6_LIBRARY"),
        reason="set VICMF6_MF6_LIBRARY to test native MF6",
    ),
]


@pytest.mark.parametrize(
    "scenario", ["reduced-grid", "active-runoff", "inactive-runoff"]
)
def test_native_grid_numbering_and_applied_runoff(tmp_path, scenario):
    flopy = pytest.importorskip("flopy")
    pytest.importorskip("xmipy")
    simulation = flopy.mf6.MFSimulation(sim_ws=str(tmp_path), verbosity_level=0)
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", perioddata=[(1.0, 2, 1.0)])
    flopy.mf6.ModflowIms(simulation, outer_dvclose=1e-10, inner_dvclose=1e-10)
    model = flopy.mf6.ModflowGwf(simulation, modelname="GW", save_flows=True)
    reduced = scenario == "reduced-grid"
    flopy.mf6.ModflowGwfdis(
        model,
        nlay=1,
        nrow=1,
        ncol=3 if reduced else 1,
        top=100.0,
        botm=0.0,
        idomain=[[[0, 1, 1]]] if reduced else 1,
    )
    flopy.mf6.ModflowGwfic(model, strt=[[[0.0, 20.0, 40.0]]] if reduced else 20.0)
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=1e-6)
    flopy.mf6.ModflowGwfsto(model, ss=0.01, iconvert=0, transient={0: True})
    flopy.mf6.ModflowGwfapi(model, maxbound=1, pname="API", save_flows=True)
    flopy.mf6.ModflowGwfoc(
        model,
        head_filerecord="gw.hds",
        budget_filerecord="gw.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    if not reduced:
        flopy.mf6.ModflowGwfsfr(
            model,
            pname="RIVER",
            nreaches=1,
            packagedata=[
                (0, (0, 0, 0), 1.0, 1.0, 0.001, 90.0, 1.0, 0.0, 0.03, 0, 1.0, 0)
            ],
            connectiondata=[(0,)],
            perioddata={
                0: [
                    (
                        0,
                        "STATUS",
                        "INACTIVE" if scenario == "inactive-runoff" else "ACTIVE",
                    )
                ]
            },
        )
    simulation.write_simulation()
    config = Mf6Config(
        workspace=tmp_path,
        library=Path(os.environ["VICMF6_MF6_LIBRARY"]),
        simulation_namefile="mfsim.nam",
        start_time=datetime(2000, 1, 1),
        api_packages={"GW": "API"},
        solution_ids={"GW": 1},
        total_time_days=1.0,
        time_step_boundaries_days=(0.5, 1.0),
        max_solve_iterations=100,
    )
    runtime = Mf6Runtime(
        config,
        model_name="GW",
        coupled_nodes=[2 if reduced else 1],
        logger=logging.getLogger(__name__),
        sfr_runoff_package=None if reduced else "RIVER",
    )
    runtime.initialize(0, mpi_comm_size=1)
    try:
        if reduced:
            np.testing.assert_allclose(runtime.current_head(), [np.nan, 20.0, 40.0])
            result = runtime.advance_to(
                1.0, np.array([0.0, 1.0, 0.0]), api_tolerance_m3_per_day=1e-10
            )
            nodes = runtime.xmi.get_value_ptr(runtime.boundary.addresses["NODELIST"])
            assert nodes[0] == 1
            assert result.applied.net_m3 == pytest.approx(1.0)
            assert result.lateral is not None
        elif scenario == "inactive-runoff":
            with pytest.raises(Mf6RuntimeError, match="SFR runoff rate mismatch"):
                runtime.advance_to(
                    1.0,
                    np.zeros(1),
                    api_tolerance_m3_per_day=1e-10,
                    sfr_runoff_rates_m3_per_day=np.ones(1),
                )
        else:
            result = runtime.advance_to(
                1.0,
                np.zeros(1),
                api_tolerance_m3_per_day=1e-10,
                sfr_runoff_rates_m3_per_day=np.ones(1),
            )
            assert result.applied_runoff.net_m3 == pytest.approx(1.0)
    finally:
        runtime.finalize()

    if reduced:
        from vicmf6.postprocess.loaders.mf6_budgets import load_mf6_budget_data
        from vicmf6.postprocess.loaders.mf6_heads import load_mf6_head_series
        from vicmf6.postprocess.metrics.exchange import _aggregate_api_volume

        metadata = SimpleNamespace(
            mf6=config,
            mf6_source=SimpleNamespace(
                models=[SimpleNamespace(name="GW", api_package="API")],
                time_step_boundaries_days=lambda: (0.5, 1.0),
            ),
        )
        from contextlib import ExitStack

        with ExitStack() as resources:
            heads = load_mf6_head_series(metadata, simulation, resources)["GW"]
            assert all(
                np.isnan(heads.read_heads(i)[0]) for i in range(len(heads.times_days))
            )
            np.testing.assert_allclose(heads.read_heads(-1), result.head_m)
            budgets = load_mf6_budget_data(
                metadata,
                simulation,
                {"GW": SimpleNamespace(node=np.arange(1, 4))},
                resources,
                tmp_path,
            )
            assert budgets["GW"]["available"], budgets["GW"]
            np.testing.assert_allclose(
                _aggregate_api_volume(budgets["GW"], 0.0, 1.0, 3), [0.0, 1.0, 0.0]
            )
