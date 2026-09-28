"""Resolve GWF models, API packages, and solution groups from MF6 input.

Names in this module are native model identifiers, not MPI ranks. The coupling
execution layer assigns ranks only after this metadata has been validated."""

from __future__ import annotations

from pathlib import Path

from ..errors import ConfigurationError
from .mf6_time import _parse_tdis
from .records import Mf6ModelMetadata, Mf6SimulationMetadata, _Mf6ModelEntry
from .text import _resolve_model_path, _strip_comment


def parse_mf6_simulation(path: str | Path) -> Mf6SimulationMetadata:
    """discover GWF models, API packages, solution groups, and TDIS metadata."""

    namefile = Path(path).expanduser().resolve()
    if not namefile.is_file():
        raise ConfigurationError(f"MF6 simulation name file was not found: {namefile}")

    workspace = namefile.parent
    lines = namefile.read_text(encoding="utf-8").splitlines()
    model_entries = _parse_mf6_model_entries(lines, workspace, namefile)
    solution_ids = _parse_mf6_solution_groups(lines, model_entries, namefile)
    tdis_file = _parse_tdis_path(lines, workspace, namefile)
    time_units, periods = _parse_tdis(tdis_file)

    models: list[Mf6ModelMetadata] = []
    for entry in model_entries:
        api_package = _parse_single_api_package(entry.namefile, entry.name)
        models.append(
            Mf6ModelMetadata(
                name=entry.name,
                namefile=entry.namefile,
                api_package=api_package,
                solution_id=solution_ids[entry.name],
            )
        )

    return Mf6SimulationMetadata(
        namefile=namefile,
        workspace=workspace,
        tdis_file=tdis_file,
        time_units=time_units,
        periods=periods,
        models=tuple(models),
    )


def _parse_mf6_model_entries(
    lines: list[str],
    workspace: Path,
    source: Path,
) -> tuple[_Mf6ModelEntry, ...]:
    models: list[_Mf6ModelEntry] = []
    inside_models = False
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        upper = [token.upper() for token in tokens]
        if upper[:2] == ["BEGIN", "MODELS"]:
            inside_models = True
            continue
        if upper[:2] == ["END", "MODELS"]:
            inside_models = False
            continue
        if not inside_models or not upper[0].startswith("GWF"):
            continue
        if len(tokens) < 2:
            raise ConfigurationError(f"invalid GWF model entry in {source}: {raw_line}")
        model_namefile = _resolve_model_path(workspace, tokens[1])
        model_name = (
            tokens[2].upper() if len(tokens) >= 3 else model_namefile.stem.upper()
        )
        models.append(_Mf6ModelEntry(name=model_name, namefile=model_namefile))

    if not models:
        raise ConfigurationError(f"no GWF models were declared in {source}")
    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ConfigurationError(f"duplicate GWF model names in {source}: {names}")
    for model in models:
        if not model.namefile.is_file():
            raise ConfigurationError(
                f"GWF name file was not found for model {model.name}: {model.namefile}"
            )
    return tuple(models)


def _parse_mf6_solution_groups(
    lines: list[str],
    models: tuple[_Mf6ModelEntry, ...],
    source: Path,
) -> dict[str, int]:
    known = {model.name for model in models}
    assignments: dict[str, list[int]] = {name: [] for name in known}
    current_group: int | None = None

    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        upper = [token.upper() for token in tokens]
        if upper[:2] == ["BEGIN", "SOLUTIONGROUP"]:
            if len(tokens) < 3:
                raise ConfigurationError(
                    f"SOLUTIONGROUP is missing its identifier in {source}: {raw_line}"
                )
            try:
                current_group = int(tokens[2])
            except ValueError as exc:
                raise ConfigurationError(
                    f"invalid SOLUTIONGROUP identifier in {source}: {tokens[2]}"
                ) from exc
            continue
        if upper[:2] == ["END", "SOLUTIONGROUP"]:
            current_group = None
            continue
        if current_group is None or not upper[0].startswith("IMS"):
            continue
        for model_name in upper[2:]:
            if model_name in assignments:
                assignments[model_name].append(current_group)

    resolved: dict[str, int] = {}
    for model_name in sorted(known):
        groups = assignments[model_name]
        if not groups:
            raise ConfigurationError(
                f"GWF model {model_name} is not assigned to a SOLUTIONGROUP in {source}"
            )
        unique = sorted(set(groups))
        if len(unique) != 1:
            raise ConfigurationError(
                f"GWF model {model_name} is assigned to multiple solution groups in {source}: {unique}"
            )
        resolved[model_name] = unique[0]
    return resolved


def _parse_tdis_path(lines: list[str], workspace: Path, source: Path) -> Path:
    candidates: list[Path] = []
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        if tokens[0].upper().startswith("TDIS") and len(tokens) >= 2:
            candidates.append(_resolve_model_path(workspace, tokens[1]))
    if len(candidates) != 1:
        raise ConfigurationError(
            f"expected exactly one TDIS entry in {source}, found {len(candidates)}"
        )
    tdis_file = candidates[0]
    if not tdis_file.is_file():
        raise ConfigurationError(f"TDIS file was not found: {tdis_file}")
    return tdis_file


def _parse_single_api_package(model_namefile: Path, model_name: str) -> str:
    packages: list[str] = []
    for raw_line in model_namefile.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens = line.split()
        if not tokens[0].upper().startswith("API"):
            continue
        if len(tokens) < 3 or not tokens[2].strip():
            raise ConfigurationError(
                f"API6 package in {model_namefile} must have an explicit package name for XMI addressing"
            )
        packages.append(tokens[2].upper())

    if not packages:
        raise ConfigurationError(
            f"GWF model {model_name} does not declare an API6 package in {model_namefile}"
        )
    if len(packages) > 1:
        raise ConfigurationError(
            f"GWF model {model_name} declares multiple API6 packages in {model_namefile}: {packages}"
        )
    return packages[0]
