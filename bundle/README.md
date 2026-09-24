# Complete Docker bundle

This directory builds the pinned VIC, MODFLOW 6, and VIC-MF6 components into
one Linux image and runs the Stehekin acceptance example. The parent
[`README.md`](../README.md) is the shortest start-to-finish guide.

## Check Docker

### Linux

Install [Docker Engine](https://docs.docker.com/engine/install/), then run:

```bash
# Check Docker.
docker --version
docker info
```

### macOS

Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/),
then run in Terminal:

```bash
# Check Docker.
docker --version
docker info

# Use the Linux image on Apple silicon.
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

### Windows

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
with Linux containers and WSL2. Run the build commands in Ubuntu WSL:

```powershell
# Check Docker.
docker --version
docker info

# Install and enter Ubuntu WSL if needed.
wsl --install -d Ubuntu
wsl
```

## Build and run

From the repository root:

```bash
# Build the complete image.
./bundle/scripts/build-image.sh

# Run the acceptance example.
./bundle/scripts/run-acceptance.sh bundle/results/stehekin
```

The result directory must be empty. Use a new directory for another run:

```bash
# Run a second acceptance case without replacing the first result.
./bundle/scripts/run-acceptance.sh bundle/results/stehekin-repeat
```

Inspect the output:

```bash
# Confirm the result.
cat bundle/results/stehekin/acceptance-status.txt

# Read the summary and report.
cat bundle/results/stehekin/postprocessing/acceptance_summary.txt
less bundle/results/stehekin/postprocessing/report.md
```

## Bundle contents

| Path | Contents |
| --- | --- |
| `components/` | Pinned VIC and MODFLOW 6 source trees |
| `components.lock` | Recorded component revisions |
| `examples/stehekin/` | Public sample inputs and acceptance fixture |
| `scripts/` | Build, component-check, and acceptance wrappers |
| `Dockerfile` | Complete build and runtime definition |
| `docs/` | Architecture, native development, and release details |

Generated results belong under `bundle/results/` and are ignored by Git.

For a native build, see [Native developer build](docs/native-build.md). For
component ownership and image boundaries, see [Bundle architecture](docs/bundle-architecture.md).
