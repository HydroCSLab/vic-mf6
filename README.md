# vicmf6

`vicmf6` is an MPI-parallel, explicit two-way coupler for the VIC 5 Image
Driver and MODFLOW 6. It keeps MODFLOW 6 groundwater state persistent through
XMI, launches VIC in restart-safe coupling windows, and transfers a signed
lower-boundary water flux through a conservative overlap-area map.

The earlier serial and one-way MPI implementations remain available in the
repository history. The `report-nm-hydro` and `one-way-mpi` tags identify the
archived milestones.

## Scientific contract

VIC owns atmosphere, canopy, snow, infiltration, root-zone and unsaturated
soil water, evapotranspiration, and surface runoff. MODFLOW 6 owns saturated
groundwater storage, hydraulic head, lateral groundwater flow, and groundwater
stresses. The ownership boundary is fixed at the bottom of the VIC soil
column.

The project sign convention is:

```text
q_gamma > 0 : VIC -> MODFLOW 6
q_gamma < 0 : MODFLOW 6 -> VIC
```

The MODFLOW 6 API pure-flux relation is:

```text
Q = HCOF * h - RHS
HCOF = 0
RHS = -Q
```

Positive project flux therefore enters groundwater. The implemented explicit
partitioned sequence is:

```text
H_mf6(t_n)
    -> conservative MF6-to-VIC head map
    -> VIC(t_n ... t_n+1)
    -> signed OUT_GW_EXCHANGE over the window
    -> conservative VIC-to-MF6 volume map
    -> API6 rate held over the matching MF6 interval
    -> H_mf6(t_n+1)
```

## Parallel architecture

The outer MPI job uses one controller plus one persistent worker per GWF model
in the split MODFLOW 6 simulation:

```text
MPI_COMM_WORLD
+-- rank 0              controller
+-- rank 1              persistent MF6 worker
+-- rank 2              persistent MF6 worker
+-- ...

rank 0 temporarily spawns VIC child ranks for every coupling window
```

The required outer world size is one controller plus the number of GWF models
declared in `mfsim.nam`.

## Runtime requirements

The validated development stack uses Linux, OpenMPI, a modified VIC 5 Image
Driver, MODFLOW 6 Extended with XMI support, `mpi4py`, `xmipy`, NumPy, NetCDF4,
and PyYAML. The VIC build must accept a spatial groundwater-head field and
write signed `OUT_GW_EXCHANGE`. Each coupled GWF model must contain an `API6`
package named `VICAPI`.

## Installation

Install against an existing VIC, MODFLOW 6, and MPI stack:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[runtime,test,preprocess]'
./native/build_disconnect.sh
./scripts/run_unit_tests.sh
```

The command-line entry point is:

```bash
vicmf6 --help
```

## Configuration

The YAML configuration points to the existing VIC and MODFLOW 6 model entry
files. Model metadata is discovered from those files rather than repeated in
the coupling configuration.

```yaml
run:
  directory: run

mf6:
  namefile: mf6/mfsim.nam
  library: /path/to/libmf6.so

vic:
  global_file: vic/global.txt
  executable: /path/to/vic_image.exe
  mpi_processes: 1
  omp_threads: 1

coupling:
  exchange_table: exchange_table.csv
  interval_days: 1.0
  scheme: explicit
  exchange_length_m: 100.0
  exchange_conductivity_scale: 0.001
  head_transform: identity
```

Inspect the resolved model and coupling contract before launching MPI:

```bash
vicmf6 inspect -c config.yml
```

Run the coupled model with one controller rank plus one rank per GWF model:

```bash
mpirun -np <outer-ranks> vicmf6 run -c config.yml
```

## Exchange-table preprocessing

Build an exact conservative overlap table from the VIC and MODFLOW 6 grids:

```bash
./scripts/build_exchange_table.sh \
    /path/to/vic.global.txt \
    /path/to/mfsim.nam \
    exchange_table.csv \
    EPSG:5070
```

The command refuses to replace an existing output unless `--force` is supplied
before the positional arguments.

## Tests

Run the deterministic source-level suite:

```bash
./scripts/run_unit_tests.sh
```

Run the MPI collective and intentional-failure smoke tests on a configured MPI
stack:

```bash
./scripts/run_mpi_smoke.sh 3
./scripts/run_mpi_failure_smoke.sh
```

## Repository layout

```text
src/vicmf6/    production Python package
native/        MPI parent/child disconnect helper
scripts/       build, launch, provenance, and test helpers
tests/         deterministic unit and MPI smoke tests
```

## Acknowledgments

This project is funded by the U.S. Geological Survey Water Resources Research
Act 104(b) grant NM_2023_Cho through the New Mexico Water Resources Research
Institute under award GR0007017, as part of USGS Grant/Cooperative Agreement
No. G21AP10635, along with an additional internal award from the NM WRRI.

The source is licensed under GPL-3.0-or-later. See `COPYING` and `CITATION.cff`.
