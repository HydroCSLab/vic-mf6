# Staged verification evidence

This directory ships the compact numerical record behind the H1--H8 Stehekin
verification campaign reported in the manuscript.
It is separate from the production acceptance workflow:

- `reference-results.json` records the numerical outcomes and source-test
  scope in a small, reviewable form.
- `verify_reference_results.py` checks the reported signs, conservation,
  temporal refinement, midpoint improvement, nonmatching-grid mapping, and
  connected-groundwater budgets.
- `../../../scripts/run-acceptance.sh` rebuilds and runs the current
  installed two-way MPI implementation.  It is the executable acceptance test
  for the released software.

Run the evidence audit from the parent repository root after building the image:

```bash
./bundle/scripts/run-verification-evidence.sh
```

The wrapper uses the installed image, so this command has no host Python
dependency.

The reference table is intentionally compact.
The original development campaign includes transient NetCDF restart files,
MODFLOW binary budgets, logs, and scripts that contain the developer's local
source paths.  Its raw outputs are approximately 695 MB and are retained as
external provenance rather than copied into the distribution repository.
The ten fully reproducible 60-day process cases remain in
`examples/stehekin/experiments/`.
