#!/usr/bin/env python3
"""Create the small MODFLOW 6 decks used by the manuscript verification.

The model input files are generated with FloPy at run time.  They are kept out
of version control so the repository contains the model definition, rather
than a second copy of generated MODFLOW files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfapi,
    ModflowGwfdis,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfsto,
    ModflowIms,
    ModflowTdis,
)

CASES = (
    dict(
        directory="mf6_h1",
        stem="h1",
        model_name="H1",
        delr=1.0,
        delc=1.0,
        icelltype=1,
        iconvert=1,
        ss=1.0e-5,
        nper=1,
        perioddata=[(10.0, 10, 1.0)],
    ),
    dict(
        directory="mf6_h2a",
        stem="h2",
        model_name="H2",
        delr=1.0,
        delc=1.0,
        icelltype=0,
        iconvert=0,
        ss=1.0e-3,
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    ),
    dict(
        directory="mf6_h2b",
        stem="h2b",
        model_name="H2B",
        delr=45312.8039853623,
        delc=45312.8039853623,
        icelltype=0,
        iconvert=0,
        ss=1.0e-3,
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    ),
)

REQUIRED_FILES = {
    case["directory"]: {
        "mfsim.nam",
        f"{case['stem']}.nam",
        f"{case['stem']}.dis",
        f"{case['stem']}.ic",
        f"{case['stem']}.ims",
        f"{case['stem']}.npf",
        f"{case['stem']}.sto",
        f"{case['stem']}.api",
        f"{case['stem']}.oc",
        f"{case['stem']}.tdis",
    }
    for case in CASES
}


def _complete(output: Path) -> bool:
    return all(
        all((output / directory / filename).is_file() for filename in files)
        for directory, files in REQUIRED_FILES.items()
    )


def _create_case(output: Path, case: dict) -> None:
    case_dir = output / case["directory"]
    case_dir.mkdir(parents=True, exist_ok=True)
    stem = case["stem"]

    simulation = MFSimulation(
        sim_name=stem,
        version="mf6",
        exe_name="mf6",
        sim_ws=case_dir,
        verbosity_level=0,
        write_headers=False,
    )
    tdis = ModflowTdis(
        simulation,
        time_units="DAYS",
        nper=case["nper"],
        perioddata=case["perioddata"],
        filename=f"{stem}.tdis",
    )
    ims = ModflowIms(
        simulation,
        print_option="SUMMARY",
        complexity="SIMPLE",
        filename=f"{stem}.ims",
    )
    model = ModflowGwf(
        simulation,
        modelname=case["model_name"],
        model_nam_file=f"{stem}.nam",
        save_flows=True,
    )
    ModflowGwfdis(
        model,
        length_units="METERS",
        nlay=1,
        nrow=1,
        ncol=1,
        delr=case["delr"],
        delc=case["delc"],
        top=0.0,
        botm=-200.0,
        filename=f"{stem}.dis",
    )
    ModflowGwfic(model, strt=-105.328, filename=f"{stem}.ic")
    ModflowGwfnpf(
        model,
        save_flows=True,
        icelltype=case["icelltype"],
        k=1.0,
        filename=f"{stem}.npf",
    )
    ModflowGwfsto(
        model,
        save_flows=True,
        iconvert=case["iconvert"],
        ss=case["ss"],
        sy=0.20,
        transient={0: True},
        filename=f"{stem}.sto",
    )
    ModflowGwfapi(
        model,
        save_flows=True,
        maxbound=1,
        filename=f"{stem}.api",
        pname="VICAPI",
    )
    ModflowGwfoc(
        model,
        head_filerecord=f"{stem}.hds",
        budget_filerecord=f"{stem}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        filename=f"{stem}.oc",
    )
    simulation.register_solution_package(ims, [model.name])
    simulation.write_simulation()


def create_fixtures(output: Path, *, force: bool = False) -> None:
    output = output.expanduser().resolve()
    if _complete(output) and not force:
        print(f"[OK] MODFLOW 6 manuscript fixtures already exist: {output}")
        return
    if output.exists() and any(output.iterdir()):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        _create_case(output, case)
    print(f"[OK] created MODFLOW 6 manuscript fixtures with FloPy: {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    create_fixtures(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
