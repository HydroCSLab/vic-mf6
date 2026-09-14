# Documentation

The root [`README.md`](../README.md) is intentionally limited to repository
navigation and the shortest complete build-and-test workflow. This directory
contains the details needed to understand, configure, review, and extend the
coupler.

## Current documentation

- [Coupling design](coupling-design.md) defines model ownership, the exchange
  sign and units, the explicit coupling sequence, and the MPI process layout.
- [Runtime and configuration](runtime-and-configuration.md) documents the YAML
  configuration, preflight checks, exchange-table generation, launch rules,
  outputs, and routine software-management commands.

The runnable Stehekin case has its own focused README under
`examples/stehekin/`. It explains the verification fixture and the evidence
produced by its acceptance workflow.

## Documentation boundary

Put these items in the root README:

- the purpose and directory layout;
- the supported installation path;
- the complete acceptance command and pass criteria; and
- brief commands for operating and recording a run.

Put scientific rationale, equations, sign and unit conventions, configuration
reference, architecture, implementation notes, debugging guidance, and
experiment interpretation in `docs/` or the relevant example README.

## Project acknowledgment

This project is funded by the U.S. Geological Survey Water Resources Research
Act 104(b) grant NM_2023_Cho through the New Mexico Water Resources Research
Institute under award GR0007017, as part of USGS Grant/Cooperative Agreement
No. G21AP10635, along with an additional internal award from the NM WRRI.
