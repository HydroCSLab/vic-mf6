# VIC-MF6 complete bundle

The `bundle/` directory builds and runs the complete VIC--MODFLOW 6 two-way coupling
stack as one OCI image.
It pins the two external model repositories, uses one MPI runtime, and includes the
small Stehekin example used for software acceptance and manuscript process
experiments.
Users do not need to build or connect VIC, MODFLOW 6, and `vicmf6` separately.

The validated platform is 64-bit Linux with Docker Engine and at least two
available CPU cores.
The image may run through Docker Desktop on macOS or Windows, but those hosts
have not been validated for MPI behavior, bind mounts, or performance.
Use a Linux host, or WSL2 with Docker's Linux containers, for the supported
workflow.

## Quick start: build and reproduce the reported workflows

Install Git and Docker Engine, then confirm that your account can run
`docker` without `sudo`.
Docker's official [Linux installation guide](https://docs.docker.com/engine/install/)
and [Docker Desktop guide](https://docs.docker.com/get-started/get-docker/)
cover supported host installations.
Plan for about 10 GB of free disk space for the image, build cache, and one
complete manuscript campaign.

Clone the parent repository and enter the checkout:

```bash
git clone --recurse-submodules https://github.com/mabdazzam/vic-mf6.git
cd vic-mf6
docker version
```

Build the complete runtime:

```bash
./bundle/scripts/build-image.sh
```

Run the three reviewable workflows:

```bash
# H1--H8: compact numerical-contract record.
./bundle/scripts/run-verification-evidence.sh

# Current two-way MPI Stehekin acceptance run.
./bundle/scripts/run-acceptance.sh bundle/results/stehekin

# P1--P7: ten 60-day snowmelt, withdrawal, aquitard, and time-step cases.
./bundle/scripts/run-feedback-campaign.sh bundle/results/manuscript-campaign
```

### Linux

Install Docker Engine using the [official Linux guide](https://docs.docker.com/engine/install/), then check the daemon:

```bash
docker --version
docker info
docker run --rm hello-world
```

From the repository root, run:

```bash
./bundle/scripts/build-image.sh
./bundle/scripts/run-verification-evidence.sh
./bundle/scripts/run-acceptance.sh bundle/results/stehekin
./bundle/scripts/run-feedback-campaign.sh bundle/results/manuscript-campaign
```

### macOS

Install and start [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/), then check Docker from Terminal:

```bash
docker --version
docker info
docker run --rm hello-world
```

On Apple silicon, set the x86-64 platform and run the same commands:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
./bundle/scripts/build-image.sh
./bundle/scripts/run-verification-evidence.sh
./bundle/scripts/run-acceptance.sh bundle/results/stehekin
./bundle/scripts/run-feedback-campaign.sh bundle/results/manuscript-campaign
```

### Windows

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) with Linux containers and WSL 2. In PowerShell:

```powershell
docker --version
docker info
docker run --rm hello-world
wsl --install -d Ubuntu
wsl
```

Run the Linux commands above inside the Ubuntu WSL terminal from the repository root. For the complete two-repository manuscript workflow, use [`../examples/manuscript/README.md`](../examples/manuscript/README.md).

The H1--H8 command checks the reported signs, conservative transfers,
temporal refinement, midpoint improvement, spatial mapping, and connected
groundwater budgets.
It does not recreate the original large development outputs.
The acceptance workflow runs the installed two-way MPI implementation.
The process campaign produces the ten manuscript cases, their audit tables,
and their figures.

The commands refuse to reuse a nonempty result directory.
Choose a new directory for another run, for example:

```bash
./bundle/scripts/run-acceptance.sh bundle/results/stehekin-repeat
```

Review the acceptance products under:

```text
bundle/results/stehekin/acceptance-status.txt
bundle/results/stehekin/diagnostics/run_summary.json
bundle/results/stehekin/postprocessing/acceptance_summary.txt
bundle/results/stehekin/postprocessing/report.md
bundle/results/stehekin/software-environment.txt
```

Review process-experiment results under:

```text
bundle/results/manuscript-campaign/analysis/
bundle/results/manuscript-campaign/figures/
bundle/results/manuscript-campaign/<case>/provenance.json
```

See [`bundle/examples/stehekin/experiments/README.md`](examples/stehekin/experiments/README.md)
for the ten-case matrix and the connection between cases and manuscript
questions.

## Repository structure

| Path | Purpose |
| --- | --- |
| `bundle/components/` | Exact external VIC and MODFLOW 6 Git revisions |
| `bundle/examples/stehekin/` | Public sample input, experiment definitions, and verification record |
| `bundle/scripts/` | Checkout verification, image build, acceptance, and campaign wrappers |
| `bundle/docs/` | Architecture, native-build, provenance, and release documentation |
| `bundle/components.lock` | Human-readable external component repositories and revisions |
| `bundle/Dockerfile` | Complete build and runtime environment |

Generated results belong under `bundle/results/` and are ignored by Git.

## Developer-native build

Docker is the supported reproducibility route.
Developers who need to modify or debug a component can build the pinned stack
on Linux with a shared compiler and MPI installation.
The required dependencies, commands, acceptance configuration, and limits are
documented in [Native developer build](docs/native-build.md).
The coupler operator guide remains in the parent repository's
[`README.md`](../README.md) and `docs/` directory.

## Image management and release

Use a different local image tag when reviewing changes:

```bash
./bundle/scripts/build-image.sh vic-mf6:review
VICMF6_IMAGE=vic-mf6:review ./bundle/scripts/run-acceptance.sh bundle/results/review
```

Inspect the installed source revisions:

```bash
docker run --rm vic-mf6:local versions
```

Open a diagnostic shell:

```bash
docker run --rm -it vic-mf6:local shell
```

Release procedure, GitHub Container Registry publication, version tags, and
immutable image digests are documented in [Releasing the bundle](docs/releasing.md).
See [Bundle architecture](docs/bundle-architecture.md) for component ownership
and reproducibility boundaries.
