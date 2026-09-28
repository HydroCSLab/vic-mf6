"""Read-only discovery of model-owned metadata.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .mf6_simulation import _parse_mf6_model_entries as _parse_mf6_model_entries
from .mf6_simulation import _parse_mf6_solution_groups as _parse_mf6_solution_groups
from .mf6_simulation import _parse_single_api_package as _parse_single_api_package
from .mf6_simulation import _parse_tdis_path as _parse_tdis_path
from .mf6_simulation import parse_mf6_simulation as parse_mf6_simulation
from .mf6_time import _parse_tdis as _parse_tdis
from .records import Mf6ModelMetadata as Mf6ModelMetadata
from .records import Mf6Period as Mf6Period
from .records import Mf6SimulationMetadata as Mf6SimulationMetadata
from .records import VicGlobalMetadata as VicGlobalMetadata
from .records import VicOutputStream as VicOutputStream
from .records import _Mf6ModelEntry as _Mf6ModelEntry
from .records import _period_time_steps as _period_time_steps
from .text import _nonnegative_int_text as _nonnegative_int_text
from .text import _positive_int_text as _positive_int_text
from .text import _resolve_model_path as _resolve_model_path
from .text import _strip_comment as _strip_comment
from .vic_global import _find_exchange_output_prefix as _find_exchange_output_prefix
from .vic_global import _forcing_sort_key as _forcing_sort_key
from .vic_global import _parse_vic_output_streams as _parse_vic_output_streams
from .vic_global import _vic_entries as _vic_entries
from .vic_global import _vic_run_length as _vic_run_length
from .vic_global import _vic_single as _vic_single
from .vic_global import _vic_start_time as _vic_start_time
from .vic_global import parse_vic_global as parse_vic_global

_sys.modules[__name__ + ".types"] = types
