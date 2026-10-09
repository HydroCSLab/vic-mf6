"""Configure a headless plotting backend and save requested figure formats."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...errors import PostprocessingError


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PostprocessingError(
            "figure generation requires matplotlib; install vicmf6[post]"
        ) from exc
    return plt


def _save(
    fig: Any, plt: Any, directory: Path, stem: str, formats: Iterable[str], dpi: int
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    created: list[str] = []
    for extension in formats:
        normalized = extension.strip().lower()
        if normalized not in {"png", "pdf", "svg"}:
            raise PostprocessingError(f"unsupported figure format: {extension}")
        path = directory / f"{stem}.{normalized}"
        kwargs = {"bbox_inches": "tight"}
        if normalized == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        created.append(path.name)
    plt.close(fig)
    return created
