# Reproducing the manuscript

The `manuscript-2026-09-19` tag freezes the code and inputs for the VIC–MODFLOW 6
framework paper. The paper repository uses the same tag. The preparation branch
is `manuscript`; use the tag when reproducing the published snapshot.

VIC and MODFLOW 6 are pinned by `bundle/components.lock`. These are the submitted
framework versions. The separate Rio Grande development campaign is not needed.

## Build and run

On Linux with Docker, Git, and sufficient disk space for retained outputs:

```sh
git clone --recurse-submodules --branch manuscript-2026-09-19 \
  https://github.com/mabdazzam/vic-mf6.git vic-mf6-manuscript
cd vic-mf6-manuscript
VICMF6_VERSION=manuscript-2026-09-19 ./bundle/scripts/build-image.sh vic-mf6:manuscript
./bundle/scripts/run-manuscript.sh /absolute/path/to/new-results --workers 2
```

The results directory must be empty. Failed runs are retained for inspection;
the workflow never overwrites an earlier run. Python packages, compilers, MPI,
VIC, and MODFLOW are provided by the image. The additional SciPy dependency
integrates the independent nonlinear reference; pytest runs the software tests.
Neither is a dependency of the coupling algorithm itself.
`bundle/requirements-container.lock` records the full tested Python environment
and constrains the image build. See [validation.md](validation.md) for the
completed release checks and measured output size.

If Docker's default build network cannot resolve package servers on your host,
set `VICMF6_BUILD_NETWORK=host` for the build command. This does not change the
model inputs or numerical methods.

The default workflow runs these stages in order:

| Stage | Experiments and retained evidence |
| --- | --- |
| `unit` | Software unit tests, three-rank collective test, intentional worker-failure propagation |
| `acceptance` | RC distributed Stehekin example, cell and domain budgets, postprocessing, matching groundwater-only control |
| `verification` | A–G prescribed-boundary tests and H1–H8 numerical experiments, including hourly references and midpoint iteration |
| `process` | Ten 60-day cases for P1–P7: snowmelt, pumping, prescribed exchange, aquitard conductivity, ARNO, and six-hour repeats |
| `reference` | Signed API tests, the exact two-store solution, 12 compiled VIC-kernel checks, and nonlinear time refinement |
| `robustness` | Fifteen runs varying interface conductance and initial soil water; 105 analysis checks |
| `initialization` | Twelve runs varying initial groundwater head or prepared state, two MPI-controller repeats, 128 analysis checks, and within-cell mapping diagnostics |

For a shorter installation check, run a subset in another empty directory:

```sh
./bundle/scripts/run-manuscript.sh /absolute/path/to/smoke-results --stages unit,acceptance,reference
```

Only a run with every stage represents the complete manuscript suite.
`execution.csv` records subprocess exit codes and wall times. Stage logs, source
hashes, component revisions, installed Python versions, input files, and native
model outputs remain together below the selected directory. Each stage stops
on failure. `completion.txt` lists the stages that finished.

## Data and figures

`tables/` contains the CSV inputs for the paper. It includes numerical summaries,
reference traces, sensitivity curves, and the selected process fields and grid
vertices needed to regenerate the figures. Values are retained at full precision;
rounding belongs in the manuscript and plotting labels.

The matching paper snapshot is
[vic-mf6-paper, manuscript-2026-09-19](https://github.com/mabdazzam/vic-mf6-paper/tree/manuscript-2026-09-19).
Its `scripts/README.md` explains figure regeneration and the source-only LaTeX
build. Copy reviewed `tables/data-*.csv` into the paper's flat `manuscript/`
directory, run the named `create-fig-*.py` and `create-tab-*.py` scripts, and
compile the manuscript and supplement. Review the resulting Git diff before
replacing an existing paper snapshot: small residuals can differ at floating-point
precision between environments.

The comparison uses only the Python standard library:

```sh
python3 examples/manuscript/scripts/compare-manuscript-tables.py \
  --tables /absolute/path/to/new-results/tables \
  --paper-dir /absolute/path/to/vic-mf6-paper \
  --output /absolute/path/to/table-comparison.csv
```

It checks corresponding table dimensions, identifiers, and values. Its rounding
tolerances are separate from the experiment suite's water-budget, convergence,
and paired-control checks.

Commit generated figure TeX, compact plotting CSV, and their scripts to the paper.
Keep raw NetCDF/NPZ files, MODFLOW binary outputs, JSON metadata, restart files,
logs, compiled libraries, PDFs, and local result archives outside the paper repo.
The professor can compile the committed TeX without Docker, Python, or model runs.

## Original numerical experiments

`verification/` preserves the original H1–H8 calculations recovered from the
development archive. `source-provenance.csv` records their original hashes.
Adaptations replace local absolute paths, record installed-source provenance,
and initialize the parallel MODFLOW library on `MPI.COMM_SELF`. They do not
change the exchange equations, solver settings, or experimental parameters.
`run-verification.py` stages writable fixtures and runs the cases in dependency
order. A–G parameter loops are reproduced from the original experiment decks;
G1's algebraic identity is also checked by the software unit tests.

The lower-boundary zero-exchange control retains the coupled boundary with a
conductivity multiplier of `1e-300`. This satisfies the positive-conductivity
driver check and gives exactly zero in saved exchange output. Both saved
exchange and baseflow are checked to be zero. Turning coupling off would instead
restore ARNO baseflow and would change the control's soil-water trajectory.

`bundle/examples/stehekin/verification/` contains the earlier compact historical
evidence audit. That command inspects stored results; it does not replace the
fresh numerical experiments above.
