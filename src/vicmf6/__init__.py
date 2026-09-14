"""mpi-parallel two-way coupling between vic and modflow 6."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["ExchangeTable", "SignedVolume"]
__version__ = "0.1.0rc1"

if TYPE_CHECKING:
    from .exchange import ExchangeTable, SignedVolume


def __getattr__(name: str) -> Any:
    """load optional public objects only when they are requested."""
    if name == "ExchangeTable":
        from .exchange import ExchangeTable

        return ExchangeTable
    if name == "SignedVolume":
        from .exchange import SignedVolume

        return SignedVolume
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
