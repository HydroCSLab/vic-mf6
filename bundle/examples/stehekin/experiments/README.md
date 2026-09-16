# Stehekin process experiments

This directory contains the reproducible experiment definitions used in the
EMS manuscript. The campaign uses the bundled VIC sample inputs, the
production `vicmf6` adapter, and a synthetic three-layer MODFLOW 6 fixture.
It does not claim calibrated Stehekin groundwater properties.

The staged H1--H8 numerical evidence is documented separately in
[`../verification/`](../verification/).  Run its compact contract audit before
or after this longer process campaign with
`./bundle/scripts/run-verification-evidence.sh` from the parent repository root.

Run the complete campaign into a new empty directory from the parent repository root:

```bash
./bundle/scripts/run-feedback-campaign.sh bundle/results/manuscript-campaign
```

The wrapper runs the installed image, so the host needs only Docker.
It mounts the requested output directory and retains every model product,
audit table, figure, and provenance record there.

The script creates ten cases and then runs the audit/plotter:

| Case | Purpose |
| --- | --- |
| `baseline` | Two-way reference with 80 mm cold-phase snow input |
| `pumped` | Reference aquitard with 1 mm/day withdrawal |
| `tight-baseline` | Tenfold lower aquitard conductivity, no withdrawal |
| `tight-pumped` | Low-conductivity aquitard with withdrawal |
| `pumped-3mm` | Threefold withdrawal sensitivity |
| `baseline-6h` | Reference case with six-hour coupling windows |
| `pumped-6h` | Withdrawal case with six-hour coupling windows |
| `high-snow` | 240 mm cold-phase snow input and melt reversal |
| `stock` | VIC-only ARNO lower-boundary control |
| `replay` | Prescribed reference exchange with groundwater withdrawal |

Each case contains its forcing, MODFLOW 6 input deck, overlap geometry,
VIC restart chain, model logs, budgets, time series, and `provenance.json`.
The campaign directory also receives `analysis/` tables and `figures/`.
The runner refuses to overwrite a nonempty destination.

The two Python files are intentionally shipped with the bundle. The runner
records an exact copy of its source in each run, while the plotter records the
checks used to produce the manuscript figures. To run one case manually, call
`run-feedback-experiment.py` with a new `--run-dir`; the campaign wrapper is
the recommended path because it prepares the compact bundle input names and
uses the installed VIC and MODFLOW 6 binaries automatically.

The generated outputs are large and should remain outside Git. Commit this
directory's scripts and README, not a completed campaign directory.
