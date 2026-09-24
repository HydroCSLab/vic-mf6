# VIC-MF6

VIC-MF6 couples the VIC 5 Image Driver and MODFLOW 6 with explicit two-way
water exchange. The Docker bundle is the shortest reproducible path for a
complete build and test.

## Requirements

You need Git, Docker, two CPU cores, and about 10 GB of free disk space. Linux
is the supported host. Docker Desktop with Linux containers also works through
macOS or Windows WSL2.

### Linux

Install [Docker Engine](https://docs.docker.com/engine/install/), then check
the daemon:

```bash
# Check Docker.
docker --version
docker info
```

### macOS

Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/),
then run these commands in Terminal:

```bash
# Check Docker.
docker --version
docker info

# Use the Linux image on Apple silicon.
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

### Windows

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
with Linux containers and WSL2. In PowerShell, install or open Ubuntu WSL:

```powershell
# Check Docker.
docker --version
docker info

# Install and enter Ubuntu WSL if needed.
wsl --install -d Ubuntu
wsl
```

Run the Linux commands below inside the WSL terminal.

## Build and run the complete example

Run these commands from any shell supported above:

```bash
# Clone the framework and its pinned model sources.
git clone --recurse-submodules https://github.com/HydroCSLab/vic-mf6.git

# Enter the repository.
cd vic-mf6

# Build the VIC, MODFLOW 6, and VIC-MF6 image.
./bundle/scripts/build-image.sh

# Run the two-way Stehekin acceptance example.
./bundle/scripts/run-acceptance.sh bundle/results/stehekin
```

The acceptance command creates the small MODFLOW 6 fixture, builds the overlap
table, runs the MPI-coupled case, checks the water-transfer and budget
diagnostics, and writes the report. The result directory must be empty before
each run.

Inspect the result:

```bash
# Confirm the run passed.
cat bundle/results/stehekin/acceptance-status.txt

# Read the numerical summary and report.
cat bundle/results/stehekin/postprocessing/acceptance_summary.txt
less bundle/results/stehekin/postprocessing/report.md
```

The complete output is under `bundle/results/stehekin/`. It is ignored by Git.

## Repository map

| Path | Contents |
| --- | --- |
| `src/vicmf6/` | Python coupler and command-line interface |
| `native/` | MPI child-disconnect helper |
| `tests/` | Unit, regression, and MPI tests |
| `examples/stehekin/` | Small end-to-end acceptance case |
| `bundle/` | Pinned components, Docker build, and acceptance wrappers |
| `docs/` | Coupling design, configuration, and native-development details |

## Native development

Use the Docker workflow above for routine verification. If you need to modify
or debug a component outside Docker, follow [Native developer build](bundle/docs/native-build.md).
The YAML contract and coupling conventions are documented in
[`docs/`](docs/README.md).

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). The software is licensed
under GPL-3.0-or-later; see [`COPYING`](COPYING).
