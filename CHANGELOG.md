# Changelog

## Unreleased

- Stream saved heads, VIC windows, and API rates; keep large intermediate tables
  on disk and align lateral cell budgets without repeatedly scanning the history.
- Resolve each model's own binary grid and remove implicit archived geometry;
  set controller/worker thread defaults before importing numerical libraries.
- Remove unused eager head-row construction, hide the retired reference option
  from help, and identify missing solver IDs accurately.

- Reduced tests to focused scientific, failure-handling, and regression scenarios;
  reused input fixtures and removed routine format checks and redundant cases.
- Removed unused private re-exports, obsolete postprocessing helpers and ignored
  acceptance arguments; model runtimes now use standard logger methods directly.
- Preserved communicator-handle-only MF6 initialization used by reference scripts,
  deriving the communicator size when it is not supplied.
- Fixed native acceptance failure reporting and rejected fractional, boolean, or
  infinite values for integer configuration settings.
- Added optional volume-conservative routing of VIC `OUT_RUNOFF` to MODFLOW 6
  SFR `RUNOFF` arrays. The mapping supports weighted cell-to-reach relations and
  model partitions, validates complete unit weights, clears unmapped reaches,
  and is covered by conservation and XMI-array tests.
- Made DISU exchange preprocessing select the highest node for each repeated
  horizontal footprint before constructing polygons, which avoids expanding all
  vertically stacked cells in regional and national models.
- Bounded the interface-forcing overview to 500 plotted node bins. Detailed
  FLOW-JA-FACE tables now use temporary disk storage instead of a node-time cap.
- distinguish overlap-scale signed transfer from node-aggregated MF6 API boundary transfer; mixed-sign overlaps may cancel within one groundwater node while net volume remains conserved.

- Replaced the redundant public configuration with model-entry-point discovery.
- Added VIC global-file parsing for paths, calendar, native step, restart, and exchange output stream.
- Added MF6 discovery for GWF models, API6 package names, solution groups, TDIS duration, and time-step boundaries.
- Added `vicmf6 inspect` for human-readable resolved-model preflight output.
- Changed internal coupling calendars to explicit half-open `[start, end)` intervals.

## 0.1.0rc1

- Replaced one-way RCHA mapping with a signed, volume-conservative overlap mapper.
- Preserved the controller plus persistent MODFLOW 6 worker MPI topology.
- Added reverse MF6-head-to-VIC mapping using overlap-area reduction.
- Added API6 signed boundary application with explicit `RHS = -Q` handling.
- Added restart-owned VIC coupling windows and signed `OUT_GW_EXCHANGE` retrieval.
- Added positive, negative, and net exchange diagnostics and conservation checks.
- Added deterministic G/H-series numerical regression tests and MPI smoke tests.
- Added standalone controls, provenance output, and package metadata.

## Postprocessing milestone

- added `vicmf6 post summarize|figures|accept|all` for completed coupled runs;
- added canonical VIC exchange, MF6 head, node-boundary, mapping, budget, lateral-flow, and connected-cell budget tables;
- added numerical-contract acceptance plus optional archived H8c reference comparison;
- added publication-oriented PNG/PDF/SVG figure generation and a Markdown run report;
- postprocessing reads VIC NetCDF through `netCDF4` and MF6 model/binary outputs through FloPy;
- added `post` optional dependencies and postprocessing documentation;
- unit suite now includes postprocessing metrics, acceptance, report-schema, and CLI coverage.
