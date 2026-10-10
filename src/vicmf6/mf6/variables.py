"""Resolve native XMI addresses, including optional discretization diagnostics.

Required variables fail with model and package context. Optional FLOWJA
connectivity may be absent in an otherwise usable native build.
"""

from __future__ import annotations

import re

from ..errors import Mf6RuntimeError


def resolve_variable_address(
    xmi, var_name: str, component_name: str, subcomponent_name: str = ""
) -> str:
    try:
        return str(xmi.get_var_address(var_name, component_name, subcomponent_name))
    except Exception as exc:
        suffix = (
            f"/{subcomponent_name}/{var_name}" if subcomponent_name else f"/{var_name}"
        )
        raise Mf6RuntimeError(
            f"failed to resolve XMI variable {component_name}{suffix}: {exc}"
        ) from exc


def optional_variable_address(
    xmi, var_name: str, component_name: str, subcomponent_name: str = ""
) -> str | None:
    try:
        address = resolve_variable_address(
            xmi, var_name, component_name, subcomponent_name
        )
        _ = xmi.get_value_ptr(address)
        return address
    except Exception:
        return None


def find_model_variable_by_suffix(
    xmi: object, model_name: str, variable_name: str
) -> str | None:
    target = variable_name.upper()
    names: list[str] = []
    for method_name in ("get_input_var_names", "get_output_var_names"):
        method = getattr(xmi, method_name, None)
        if not callable(method):
            continue
        try:
            names.extend(str(name) for name in method())
        except Exception:
            continue
    candidates = [
        name
        for name in dict.fromkeys(names)
        if name.upper().endswith("/" + target)
        and model_name.upper() in name.upper().split("/")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # discretization arrays are preferred when multiple packages expose
        # similarly named values.
        dis = [name for name in candidates if re.search(r"/DIS[UV]?/", name.upper())]
        if len(dis) == 1:
            return dis[0]
    return None
