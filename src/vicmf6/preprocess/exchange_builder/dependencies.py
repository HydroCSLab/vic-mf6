"""Load optional geospatial dependencies only when preprocessing is requested."""

from __future__ import annotations

from .records import ExchangeBuildError


def _optional_imports():
    try:
        import flopy
        from netCDF4 import Dataset
        from pyproj import CRS, Transformer
        from shapely.geometry import Polygon
        from shapely.ops import transform
        from shapely.strtree import STRtree
    except ImportError as exc:
        raise ExchangeBuildError(
            "exchange-table preprocessing requires vicmf6[preprocess] "
            "(netCDF4, FloPy, pyproj, and shapely)"
        ) from exc
    return Dataset, flopy, CRS, Transformer, Polygon, transform, STRtree
