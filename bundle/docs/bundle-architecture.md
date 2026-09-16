# Bundle architecture

The bundle is a distribution layer around the parent `vic-mf6` coupler and two
independently useful scientific model sources. VIC owns land-surface physics,
MODFLOW 6 owns groundwater flow, and the parent checkout owns their explicit
two-way exchange, MPI control, and acceptance diagnostics. The external model
repositories remain submodules so each change can be reviewed and tested in
the project where it belongs.

## Source and build boundary

Git submodules are the machine-readable source lock. `bundle/components.lock` repeats
the same revisions in a compact form that reviewers can inspect and that the
checkout verification script checks before every image build. The source trees
and their license files are also retained under `/opt/vicmf6/src` in the image.

The image builds every native component against one OpenMPI installation:

1. the modified VIC Image Driver executable;
2. the parallel MODFLOW 6 executable and XMI shared library;
3. the MPI child-disconnect preload library; and
4. the installable `vicmf6` Python package and its pinned direct dependencies.

Using one MPI implementation inside the image avoids host-library ABI mixing.
Only Docker communicates with the host; the coupled executables communicate
inside the container.
The three public wrappers run the compact H1--H8 record, production acceptance,
and the ten P-series process cases inside that same installed image.
Consequently Git and Docker are the only host-side requirements for the
supported workflow.

## Acceptance boundary

The included Stehekin inputs are a small public VIC sample. The groundwater
model is generated as a numerical fixture and is not a calibrated Stehekin
groundwater model. This distinction keeps the example focused on software
contracts: time-window completion, conservative spatial mapping, API volume
application, signed exchange, restart continuity, water balance, and
groundwater-budget closure.

The runtime works in a disposable directory. It copies only run products,
diagnostics, the postprocessing report, and software provenance to the mounted
output directory. It refuses a nonempty destination so an earlier result is
never silently replaced.

## Reproducibility boundary

The component commits and direct Python dependencies are pinned. The generated
`software-environment.txt` records the complete resolved Python environment and
the installed native revisions for each run. The Ubuntu base tag and Ubuntu
package repositories remain external build inputs; a release image should also
be published by immutable digest so an examiner can run the tested binary image
without resolving those packages again.

## Project boundary

The bundle repository contains reusable packaging, tests, documentation, and a
small example. The tracked Stehekin experiment definitions under
`bundle/examples/stehekin/experiments/` reproduce the ten process cases reported in
the manuscript; their generated runs remain external to Git. The compact H1--H8
numerical evidence table and its read-only contract audit live under
`bundle/examples/stehekin/verification/`. It preserves the reported numerical record
without adding roughly 695 MB of local-path raw campaign output to the release.
Manuscript files and project-specific simulation products belong
in their own `vic-mf6-paper` and workflow directories. Generated model inputs,
run directories, and analysis products remain outside version control.

The [native developer build](native-build.md) documents the separate advanced
route for changing or debugging a pinned component outside the image.
