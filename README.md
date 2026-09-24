# VIC-MF6 (`vicmf6`)

`vicmf6` runs explicit two-way coupling between the VIC 5 Image Driver and
MODFLOW 6. This page is the short operator guide: it shows where things are and
how to verify a complete checkout. Scientific conventions and implementation
details are kept in [`docs/`](docs/README.md).

The earlier serial and one-way MPI implementations are preserved at the
`report-nm-hydro` and `one-way-mpi` tags.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/vicmf6/` | Installable Python package and command-line interface |
| `native/` | MPI parent/child disconnect helper used by VIC child jobs |
| `scripts/` | Environment, build, launch, provenance, and test wrappers |
| `tests/` | Unit, regression, and MPI smoke tests |
| `examples/stehekin/` | Small end-to-end coupled verification case |
| `docs/` | Scientific design, configuration, and developer documentation |

## Recommended build: complete Docker bundle

The `bundle/` directory is the recommended way for an examiner to build and
run the complete software stack.
It builds VIC, MODFLOW 6, and this coupler in one Linux container and includes
the Stehekin acceptance case.
The external model sources are pinned as submodules.

```bash
git clone --recurse-submodules https://github.com/mabdazzam/vic-mf6.git
cd vic-mf6
./bundle/scripts/build-image.sh
./bundle/scripts/run-acceptance.sh bundle/results/stehekin
```

Review the complete instructions in [`bundle/README.md`](bundle/README.md).

## Required runtime stack

The Python package orchestrates existing model builds. A complete coupled run
requires:

- Linux and an MPI implementation compatible with all compiled components;
- a modified VIC 5 Image Driver that reads the groundwater-head interface and
  writes signed `OUT_GW_EXCHANGE` output;
- a MODFLOW 6 shared library with XMI and API6 support; and
- Python 3.10 or newer.

The repository does not silently download or replace VIC or MODFLOW 6. Record
the exact model revisions and compiler/MPI stack used for a test.

## Build and run the software checks

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[runtime,preprocess,post,test]'

./native/build_disconnect.sh
./scripts/run_unit_tests.sh
./scripts/run_mpi_smoke.sh 3
./scripts/run_mpi_failure_smoke.sh
```

The failure smoke test intentionally aborts one worker. It passes only when the
failure reaches the launcher and the script prints an `[OK]` result.

## Run the end-to-end Stehekin acceptance test

First make a local configuration. Local configurations and generated run
products are ignored by Git.

```bash
cp examples/stehekin/config.example.yml \
   examples/stehekin/config.local.yml
${EDITOR:-vi} examples/stehekin/config.local.yml
```

For a native build, use the user-owned installation layout below.
It avoids requiring root access and gives the configuration stable paths.
The native build guide creates these directories and installs the compiled
components there.

```yaml
mf6:
  library: /home/USERNAME/usr/local/opt/vic-mf6/lib/libmf6.so

vic:
  global_file: /home/USERNAME/usr/local/src/vic-mf6/bundle/examples/stehekin/stehekin.global.txt
  executable: /home/USERNAME/usr/local/bin/vic_image.exe
```

Replace `USERNAME` with the output of `id -un`, or generate the file from the
shell variable used by the native guide:

```bash
export VICMF6_PREFIX="$HOME/usr/local/opt/vic-mf6"
mkdir -p "$HOME/usr/local/src" "$HOME/usr/local/bin" "$VICMF6_PREFIX"
sed -e "s|@MF6_LIBRARY@|$VICMF6_PREFIX/lib/libmf6.so|" \
    -e "s|@VIC_GLOBAL_FILE@|$PWD/bundle/examples/stehekin/stehekin.global.txt|" \
    -e "s|@VIC_EXECUTABLE@|$HOME/usr/local/bin/vic_image.exe|" \
    examples/stehekin/config.example.yml > examples/stehekin/config.local.yml
```

Then run the complete acceptance workflow:

```bash
./scripts/acceptance_stehekin.sh \
    examples/stehekin/config.local.yml
```

The wrapper rebuilds the synthetic MODFLOW 6 fixture and exchange table,
builds the native helper, inspects the configuration, runs the two-way coupled
case with two outer MPI ranks, checks its numerical signatures, and creates the
postprocessing report.

The acceptance case owns its generated files. Each run replaces
`examples/stehekin/run/mf6`, `run/vic`, `run/diagnostics`, and
`run/postprocessing`. It does not modify the configured VIC global file,
executable, or MODFLOW 6 shared library.

A passing run prints `[OK]` from the Stehekin result checker. Review:

```text
examples/stehekin/run/diagnostics/run_summary.json
examples/stehekin/run/postprocessing/acceptance_summary.txt
examples/stehekin/run/postprocessing/report.md
```

## Operate a configured case

When running directly from a repository checkout, use the bundled launcher so
the command does not depend on a separately installed `vicmf6` entry point:

```bash
./vicmf6 inspect -c config.yml
./vicmf6 preflight -c config.yml
mpirun -np <outer-ranks> ./vicmf6 run -c config.yml
./vicmf6 post all -c config.yml
```

Use one outer controller rank plus one rank for each coupled MODFLOW 6 GWF
model. `vic.mpi_processes` separately controls the VIC child-job size.

The complete, reproducible checkout-local command is the Stehekin acceptance
workflow above. It uses the tracked configuration template and example model
builder, while requiring only the local VIC executable, VIC input data, and
MODFLOW 6 shared library to be supplied in `config.local.yml`.

Capture the software environment alongside results:

```bash
./scripts/collect_provenance.sh > provenance.txt
```

See [Runtime and configuration](docs/runtime-and-configuration.md) for the YAML
contract and preprocessing commands. See [Coupling design](docs/coupling-design.md)
for the scientific ownership, signs, units, algorithm, and MPI topology.

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). The software is licensed
under GPL-3.0-or-later; see [`COPYING`](COPYING).

## Acknowledgments

This project is funded by the U.S. Geological Survey (USGS) Water Resources Research Act 104(b) grant [NM_2023_Cho](https://water.usgs.gov/wrri/grant-details.php?ProjectID=2023NM163B&Type=Annual) through the New Mexico Water Resources Research Institute (NM WRRI) under award GR0007017, as part of USGS Grant/Cooperative Agreement No. G21AP10635, along with an additional internal award from the NM WRRI.
