"""Read-only discovery of model-owned metadata."""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .mf6_simulation import parse_mf6_simulation as parse_mf6_simulation
from .records import Mf6ModelMetadata as Mf6ModelMetadata
from .records import Mf6Period as Mf6Period
from .records import Mf6SimulationMetadata as Mf6SimulationMetadata
from .records import VicGlobalMetadata as VicGlobalMetadata
from .records import VicOutputStream as VicOutputStream
from .vic_global import parse_vic_global as parse_vic_global

_sys.modules[__name__ + ".types"] = types
