#!/usr/bin/env python3
"""Run the manuscript's numerical tests and process experiments in fresh folders."""
import argparse
import csv
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


STAGES = ["unit", "acceptance", "verification", "process", "reference", "robustness", "initialization"]


def _ansi(code, text, enabled):
    """Wrap text in an ANSI color when terminal output supports it."""
    return f"\033[{code}m{text}\033[0m" if enabled else text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, default=Path("/opt/vicmf6"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma-separated stages; default: all.")
    args = parser.parse_args()
    selected = args.stages.split(",")
    if args.workers < 1 or not selected or any(s not in STAGES for s in selected) or len(set(selected)) != len(selected):
        parser.error("Provide a positive worker count and unique valid stages: " + ",".join(STAGES))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("Output directory must be empty; previous runs are never overwritten.")
    scripts = Path(__file__).resolve().parent
    install = args.install_dir.resolve()
    coupler = install / "src/vic-mf6"
    logs = out / "logs"
    logs.mkdir()
    tables = out / "tables"
    tables.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env["PYTHONUNBUFFERED"] = "1"
    color_setting = os.environ.get("VICMF6_COLOR", "auto").lower()
    color = color_setting == "always" or (color_setting == "auto" and sys.stdout.isatty())
    blue = lambda text: _ansi(34, text, color)
    cyan = lambda text: _ansi(36, text, color)
    green = lambda text: _ansi(32, text, color)
    red = lambda text: _ansi(31, text, color)
    shutil.copy2(install / "share/component-revisions.txt", out / "component-revisions.txt")
    with (out / "python-packages.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["package", "version"])
        writer.writerows(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))
    with (out / "source-hashes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "sha256"])
        for root in [scripts.parent, coupler / "src", coupler / "tests", install / "examples/stehekin"]:
            for path in sorted(root.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    writer.writerow([str(path), hashlib.sha256(path.read_bytes()).hexdigest()])
        for path in [install / "bin/vic_image.exe", install / "lib/libmf6.so", install / "lib/libvic_parent_disconnect.so"]:
            writer.writerow([str(path), hashlib.sha256(path.read_bytes()).hexdigest()])
    records = []

    def run(name, command, cwd=None):
        print(blue(f"[manuscript] {name}"), flush=True)
        print(cyan("$ " + " ".join(shlex.quote(str(part)) for part in command)), flush=True)
        start = time.monotonic()
        with (logs / f"{name}.log").open("w") as stream:
            process = subprocess.Popen(
                list(map(str, command)),
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            returncode = process.wait()
        records.append([name, returncode, time.monotonic() - start])
        with (out / "execution.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stage", "exit_code", "seconds"])
            writer.writerows(records)
        if returncode:
            print(red(f"[FAIL] {name} (exit code {returncode})"), flush=True)
            raise RuntimeError(f"{name} failed; inspect {logs / (name + '.log')}")
        print(green(f"[OK] {name}"), flush=True)

    def script(name, *arguments):
        run(name, [sys.executable, scripts / f"{name}.py", *arguments])

    if "unit" in selected:
        run("unit", [coupler / "scripts/run_unit_tests.sh"], coupler)
        run("mpi-collectives", [coupler / "scripts/run_mpi_smoke.sh"], coupler)
        run("mpi-failure", [coupler / "scripts/run_mpi_failure_smoke.sh"], coupler)
    if "acceptance" in selected:
        run("acceptance", [install / "bin/container-entrypoint", "acceptance", out / "acceptance"])
        script("run-groundwater-control", "--acceptance-dir", out / "acceptance",
               "--output-dir", out / "groundwater-control", "--install-dir", install)
    if "verification" in selected:
        script("run-verification", "--output-dir", out / "verification", "--install-dir", install)
    if "process" in selected:
        run("process", [install / "bin/container-entrypoint", "feedback-campaign", out / "process"])
        script("create-data-process-figures", "--campaign-dir", out / "process", "--output-dir", tables)
    if "reference" in selected:
        script("run-coupled-reference", "--output-dir", out / "reference", "--install-dir", install)
        inputs = install / "examples/stehekin/input"
        script("create-data-nonlinear-reference", "--output-dir", out / "nonlinear-reference",
               "--parameter-file", inputs / "stehekin_parameters_20160327.nc",
               "--domain-file", inputs / "domain_stehekin_20151028.nc")
        script("run-nonlinear-interface-reference", "--output-dir", out / "nonlinear",
               "--install-dir", install, "--reference-dir", out / "nonlinear-reference")
    if "robustness" in selected:
        script("run-robustness-campaign", "--output-dir", out / "robustness", "--install-dir", install, "--workers", args.workers)
        script("analyze-robustness-campaign", "--campaign-dir", out / "robustness", "--output-dir", tables)
    if "initialization" in selected:
        script("run-review-campaign", "--output-dir", out / "initialization", "--install-dir", install, "--workers", args.workers)
        script("analyze-review-campaign", "--campaign-dir", out / "initialization", "--output-dir", tables)
        script("run-spatial-mapping-diagnostic", "--campaign-dir", out / "initialization", "--output-dir", tables, "--coupler-dir", coupler)
    script("collect-manuscript-tables", "--run-dir", out, "--output-dir", tables)
    (out / "completion.txt").write_text("Completed stages: " + ", ".join(selected) + "\n")
    print(f"[OK] completed {', '.join(selected)}; outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
