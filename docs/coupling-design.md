# Coupling design

## Scientific ownership

VIC owns atmospheric forcing, canopy and snow processes, infiltration,
root-zone and unsaturated soil water, evapotranspiration, and surface runoff.
MODFLOW 6 owns saturated groundwater storage, hydraulic head, lateral
groundwater flow, and groundwater stresses. The coupling interface is the
bottom of the VIC soil column.

The configured VIC and MODFLOW 6 models remain independently recognizable.
The coupler exchanges only the interface quantities needed to advance them:

- MODFLOW 6 sends groundwater head to VIC; and
- VIC returns the signed lower-boundary water exchange accumulated during the
  coupling window.

## Sign and API convention

The project sign convention is:

```text
q_gamma > 0 : VIC -> MODFLOW 6
q_gamma < 0 : MODFLOW 6 -> VIC
```

VIC reports exchange as a signed depth. The coupler multiplies that depth by
the exact overlap area to obtain volume and aggregates overlap contributions to
MODFLOW 6 nodes.

The MODFLOW 6 API pure-flux relation is:

```text
Q = HCOF * h - RHS
HCOF = 0
RHS = -Q
```

Positive project flux therefore enters groundwater. Sign conversion is made
at the API boundary rather than being distributed across the code.

## Explicit two-way sequence

For a coupling window from `t_n` to `t_n+1`, the implemented sequence is:

```text
H_mf6(t_n)
    -> conservative MF6-to-VIC head map
    -> VIC(t_n ... t_n+1)
    -> signed OUT_GW_EXCHANGE over the window
    -> conservative VIC-to-MF6 volume map
    -> API6 rate held over the matching MF6 interval
    -> H_mf6(t_n+1)
```

This is an explicit partitioned scheme: VIC uses the groundwater state at the
start of the window, and MODFLOW 6 then applies VIC's integrated response over
that same interval. Coupling-interval sensitivity is therefore part of a
scientific application assessment even when every mapping and budget check
passes.

## Spatial mapping

The exchange table contains one row per nonzero intersection between a VIC
cell and a MODFLOW 6 cell. An overlap area is immutable geometry and is used for
both directions:

```text
V_vic_to_mf6 = q_vic_mm * 1e-3 * overlap_area_m2
```

The forward map preserves signed volume. The reverse head map uses an
area-weighted reduction over each VIC cell. VIC identifiers and MODFLOW 6 node
numbers are explicit identifiers; neither should be treated as an implicit
array position.

For absolute-head coupling, the interface elevation must share the MODFLOW 6
vertical datum. With
`head_transform: pressure_head_from_interface_elevation`, the value passed to
VIC is:

```text
groundwater head - VIC soil-base interface elevation
```

## MPI topology

The persistent outer MPI job contains one controller and one worker for each
coupled GWF model:

```text
MPI_COMM_WORLD
+-- rank 0              controller
+-- rank 1              persistent MF6 worker
+-- rank 2              persistent MF6 worker
+-- ...

rank 0 temporarily spawns the configured VIC child ranks for each window
```

MODFLOW 6 state remains live through XMI for the complete run. VIC advances in
restart-linked child jobs. Outer ranks and VIC child ranks are separate
parallelism controls.

## Verification evidence

For every window, diagnostics compare the signed transfer through successive
stages:

```text
VIC full-cell source volume
overlap-row mapped volume
MODFLOW 6 node target volume
requested API integrated volume
solved API integrated volume
```

The first failed equality identifies the stage where conservation was lost.
Connected groundwater diagnostics separately account for storage and internal
lateral flow, including FLOW-JA-FACE antisymmetry and cancellation.

A passing software fixture verifies these coupling contracts for that fixture.
Hydraulic parameters, forcing, boundary conditions, and calibration require
their own scientific evaluation for each application.

## Application checks beyond conservation

Use coupling intervals `dt`, `dt/2`, and `dt/4` aligned with both native model
calendars. Compare head trajectories, upward and downward exchange separately,
soil moisture, ET, and streamflow where available. Exact transferred-volume
closure does not bound the time-lag error of an explicit scheme, particularly
in low-storage aquifers or during rapid wetting and drying.

The current VIC exchange law represents an effective hydraulic resistance
between the soil base and groundwater. The gap is not an explicitly resolved
vadose column with its own water storage. Application tests should cover deep
water tables, groundwater approaching the soil column, and upward extraction
from a nearly dry aquifer. Groundwater donor availability and storage ownership
need a defined physical treatment before adding an automatic flux limiter;
clipping only the MF6 side would break the water transfer already applied by VIC.

Check process ownership when retaining native groundwater stresses: VIC-driven
recharge or ET should not also be imposed on the same area by RCH, UZF, or EVT
unless those terms deliberately represent different water sources or sinks.
The coupled VIC path already suppresses its conceptual ARNO baseflow.

A complete catchment ledger would include precipitation, ET, soil/snow/canopy
storage, groundwater storage, channel storage, outlet discharge, and external
stresses. Groundwater exchange and optional VIC-to-SFR runoff are internal
transfers in that ledger. The present groundwater CBC closure and interface
checks verify only their stated portions of this larger balance.
