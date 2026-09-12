"""human-readable model inspection output for configuration and preflight work."""

from __future__ import annotations

from .config import ApplicationConfig


def format_inspection(config: ApplicationConfig, summary: dict[str, object]) -> str:
    """format discovered model metadata without exposing implementation details."""

    vic = summary["vic"]
    mf6 = summary["mf6"]
    coupling = summary["coupling"]
    mpi = summary["mpi"]

    model_lines: list[str] = []
    for model in mf6["models"]:
        model_lines.append(
            f"    {model['name']}: API={model['api_package']} solution={model['solution_id']}"
        )

    forcing_lines = "\n".join(
        f"    {path}" for path in config.vic_source.forcing_prefixes
    )

    lines = [
        "VIC model",
        f"  global file       : {vic['global_file']}",
        f"  working directory : {config.vic.working_directory}",
        f"  domain            : {vic['domain_file']}",
        f"  parameters        : {vic['parameters_file']}",
        f"  forcing streams   : {vic['forcing_streams']}",
        forcing_lines,
        f"  model steps/day   : {vic['model_steps_per_day']}",
        f"  records           : {vic['nrecs']}",
        f"  simulation        : {vic['start_time']} -> {vic['end_time']} [end exclusive]",
        f"  exchange output   : {vic['exchange_output_prefix']} / {vic['exchange_variable']}",
        "",
        "MODFLOW 6",
        f"  namefile          : {mf6['namefile']}",
        f"  TDIS              : {mf6['tdis_file']}",
        f"  time units        : {mf6['time_units']}",
        f"  duration          : {mf6['total_time_days']:.17g} days",
        f"  GWF models        : {len(mf6['models'])}",
        *model_lines,
        "",
        "Coupling",
        f"  exchange table    : {config.coupling.exchange_table}",
        f"  overlaps          : {coupling['overlap_rows']}",
        f"  VIC cells         : {coupling['vic_cells']}",
        f"  coupled area      : {coupling['total_overlap_area_m2']:.17g} m2",
        f"  interval          : {coupling['interval_days']:.17g} days",
        f"  windows           : {coupling['windows']}",
        f"  head transform    : {config.vic.head_transform}",
        "",
        "MPI",
        f"  controller ranks  : {mpi['controller_ranks']}",
        f"  MF6 worker ranks  : {mpi['mf6_worker_ranks']}",
        f"  outer ranks       : {mpi['world_ranks_required']}",
        f"  VIC child ranks   : {mpi['vic_child_ranks']}",
        "",
        "preflight: PASS",
    ]
    return "\n".join(line for line in lines if line is not None)
