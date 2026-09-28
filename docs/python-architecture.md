# Reading and extending the Python framework

The framework implements one explicit coupling scheme. VIC receives groundwater
heads at the start of a window, runs once, and returns an accumulated exchange
amount. Persistent MODFLOW models then advance through that same window under a
constant exchange rate. The next window uses their updated heads.

Start with [`coupling/explicit.py`](../src/vicmf6/coupling/explicit.py). Its loop
contains the complete scientific sequence. Follow a stage into
[`coupling/stages.py`](../src/vicmf6/coupling/stages.py) when you want its equations
and conservation checks; you do not need to read MPI buffer packing or plotting
to understand that sequence.

## Where each responsibility lives

| Question | Module to read or change |
| --- | --- |
| What happens during one coupling window? | `coupling/explicit.py`, `coupling/stages.py` |
| Which rank owns a model, its files, and its lifetime? | `coupling/session.py` |
| What crosses MPI, and which reduction applies? | `coupling/parallel.py` |
| How are accepted results reported? | `coupling/reporting.py`, `diagnostics/` |
| How does YAML become a validated configuration? | `config/loading.py`, `config/validation.py`, `config/records.py` |
| Where do model calendars and package names come from? | `model_inputs/vic_global.py`, `model_inputs/mf6_simulation.py`, `model_inputs/mf6_time.py` |
| How is geometry checked and used to exchange water? | `exchange/loading.py`, `exchange/table.py`, `exchange/conservation.py` |
| How are VIC restarts and global files prepared? | `vic/runtime.py`, `vic/global_file.py`, `vic/restart_files.py` |
| How is VIC launched? | `mpi_spawn.py` |
| How does MF6 initialize and retain state? | `mf6/runtime.py` |
| What happens inside an MF6 substep? | `mf6/advance.py`, `mf6/time_steps.py` |
| What do API package arrays mean? | `mf6/boundary.py`, `mf6/variables.py` |
| How are native lateral flows interpreted? | `mf6/native_flow.py`, `mf6/lateral_flow.py` |
| How are missing VIC values handled? | `netcdf.py` |
| How is an overlap CSV generated offline? | `preprocess/exchange_builder/` |
| How are saved outputs loaded, analyzed, and plotted? | `post/loaders/`, `post/metrics/`, `post/figures/` |
| Where are output columns and acceptance rules defined? | `post/table_schema.py`, `post/acceptance.py` |

Files are grouped by responsibility, with plain functions for calculations and
small objects for resources that have a lifetime. `Mf6Runtime` owns an
`ApiFluxBoundary`, a `NativeLateralFlow` reader, and a `TdisSchedule`. There is no
plugin registry, inheritance hierarchy, or second configuration layer to learn.
The original public imports, such as `from vicmf6.mf6 import Mf6Runtime`, still
work through package exports. `driver.py` retains the execution entry point.

Record definitions use `records.py`, and logger setup uses `log_setup.py`.
Avoid standard-library module names such as `types.py` and `logging.py`: editors
and other Python tools can search the current directory during startup, even
when no framework command is being run. Package-qualified legacy imports are
retained as aliases, without conflicting files on disk.

## Follow the water through one window

1. Each groundwater worker supplies the heads for its own model. For each VIC
   cell, the controller computes `sum(head * overlap_area) / sum(overlap_area)`.
   Reducing extensive quantities before division preserves unequal overlap
   weights across worker partitions. The fixed denominator is reduced once at
   initialization.
2. The controller writes those heads, prepares one VIC global file, and spawns
   VIC. The previous window's restart is the initial state of this window. The
   accepted restart becomes the next window's initial state.
3. VIC's `OUT_GW_EXCHANGE` is accumulated over the window. The shared NetCDF
   reader preserves missing records as NaN. Coupled cells must be finite;
   inactive cells elsewhere on the VIC raster may remain masked.
4. An overlap transfers `exchange_mm * 0.001 * overlap_area_m2` cubic meters.
   Positive and negative overlap volumes are checked independently. After
   summing onto MF6 nodes, opposing overlaps may cancel; only net volume is
   invariant through this aggregation.
5. A node's constant rate is `volume_m3 / window_days`. Positive project exchange
   enters groundwater. API6 receives `HCOF = 0` and `RHS = -rate`. The adapter
   checks the converged `SIMVALS`, rather than assuming a written request was
   applied.
6. MF6 follows its original TDIS substeps. For each substep, the sequence is
   `prepare_time_step`, write API rates, `prepare_solve`, solve to convergence,
   `finalize_solve`, read API and FLOWJA values, then `finalize_time_step`.
   Sampling before `finalize_solve` would read stale flows. Stepping across a
   coupling boundary is rejected.
7. Domain diagnostics compare the node targets with the applied API volumes.
   The controller records the accepted window. There is no VIC–MF6 iteration or
   rollback in this algorithm; MF6's internal nonlinear solve still converges
   within each native timestep.

The source comments explain units, ownership, and ordering constraints at these
boundaries. The explanatory style was informed by
[Torrent.jl's simulation runner](https://github.com/DOE-ICoM/Torrent.jl/blob/main/src/simulation_runner.jl).

## MPI ownership and failure handling

The outer MPI world contains one controller plus one rank per GWF model.
Groundwater workers share a separate communicator passed to MF6. Rank zero owns
the complete exchange table, VIC files, and domain diagnostics. Each worker
receives only its model's overlap rows while retaining the common VIC cell order.

All ranks enter the coupling collectives in the same order. A rank with no local
contribution sends the neutral value: zero for sums, positive infinity for a
minimum, and negative infinity for a maximum. Related scalar diagnostics are
packed by reduction operation. A barrier after mapping prevents workers from
starting a native solve before the controller accepts conservation.

On success, each worker finalizes MF6 and frees its communicator. A rank-local
failure propagates to the CLI's `MPI.COMM_WORLD.Abort(1)` path. Do not put native
finalization in an unconditional `finally` block: it can wait for peers already
blocked in another collective, preventing the failing rank from reaching abort.
Partial outputs remain diagnostic evidence; they are not accepted restart points.

### Why the small C helper remains

Python already spawns VIC through `MPI.COMM_SELF.Spawn`. The helper in
[`native/mpi_finalize_disconnect.c`](../native/mpi_finalize_disconnect.c) runs
inside each VIC child. It intercepts native `MPI_Finalize`, disconnects from the
parent if there is one, then calls `PMPI_Finalize`.

```text
Python controller                       VIC child ranks
COMM_SELF.Spawn ----------------------> MPI_Init and VIC window
intercommunicator.Disconnect <--------> helper: disconnect parent communicator
continue with completed VIC outputs    helper: PMPI_Finalize
```

`MPI_Comm_disconnect` requires matching participation on both sides. A Python
parent cannot make the child execute that operation. Replacing the helper would
require changing VIC's native shutdown or adopting and validating a different
launch arrangement. Keeping it preserves the tested VIC binary and shutdown
behavior; this refactor adds no compiled component. The helper does not spawn or
kill ranks. See the Open MPI documentation for
[MPI_Comm_disconnect](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Comm_disconnect.3.html)
and [MPI_Finalize](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Finalize.3.html).

## Scaling changes and remaining limits

The refactor removes repeated work without changing the explicit scheme:

- Overlap CSV validation occurs on the controller during session initialization;
  workers receive their own overlap rows. Preflight independently validates the
  complete table before the run starts.
- Model overlap indices occupy space proportional to overlap count, replacing a
  full boolean mask for every model. VIC row, column, and area arrays are cached.
- The static head-map denominator is reduced once. Related scalar reductions
  are packed, reducing the explicit window's outer collective calls from 20 to 8.
- Native TDIS boundaries are validated once, then searched in logarithmic time.
- API volumes accumulate in one boundary-sized vector. Memory no longer grows
  with the number of MF6 substeps in a window.
- The completed MF6 advance supplies the head snapshot for diagnostics, avoiding
  another native head copy. The static node count is cached as well.

This is still a centralized explicit coordinator. Each worker retains the global
coupled VIC cell order and uses VIC-sized communication buffers. Rank zero holds
the full geometry and constructs the initial worker partitions. VIC restarts,
MPI dynamic spawning, and file I/O remain costs; postprocessing loads saved
series into memory. These limits require measurements on the target cluster
before claiming large-domain speedups. Changing them would be a separate design
change, not a consequence of merely adding more modules.

## Make an extension in the owning module

To add a diagnostic, compute it where the relevant model state is available,
pass a named result through the coupling stages, and update the diagnostic/output
schema deliberately. A worker statistic also needs its mathematical reduction
specified in `parallel.py`. Optional lateral diagnostics remain absent when only
some workers provide topology, so a partial sum is not reported as a whole-domain
quantity.

To support another grid geometry, extend the offline preprocessing loader and
keep the same validated overlap columns. The runtime transfer operator should
not need polygon operations or a special branch for each grid type.

To change model startup or a native pointer, work in the relevant adapter. Keep
model advancement independent of reporting and keep sign conversion in the API
boundary adapter. An iterative scheme would additionally require tested model
checkpoint/restore and convergence rules; the current modules do not imply that
those capabilities already exist.

## Verification

Run the ordinary suite and lint from the repository root:

```bash
PYTHONPATH=src python -m pytest tests -q
python -m ruff check src/vicmf6 tests
```

The ordinary suite covers mapping, input parsing, output
protection, missing data, native solve ordering, irregular schedules, substep
integration, injected native errors, and Python startup from each source
directory. Three additional MPI scenarios are
opt-in on a system with the supported Open MPI/mpi4py stack:

```bash
VICMF6_RUN_MPI_TESTS=1 PYTHONPATH=src python -m pytest tests/test_mpi_coupling.py -q
```

Those scenarios exercise two groundwater workers through the actual explicit
driver and collectives, using deterministic model substitutes: complete topology,
partial topology, and a worker exception that must abort promptly. Native VIC and
MF6 execution is covered by the existing full Stehekin acceptance workflow.

For this refactor, the complete 10-window Stehekin run passed, producing all 17
tables and 34 figures. A comparison against the saved pre-refactor run checked
7,778 numeric values across the runtime CSV and postprocessing tables with
`rtol=1e-12, atol=1e-10`; none differed beyond those tolerances. Timing columns and
run-specific file paths were excluded. This is evidence for the tested fixture,
not a guarantee for every MPI implementation or model configuration.
