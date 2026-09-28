"""Conservative exchange mapping and signed-volume contracts.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

# Preserve the qualified legacy import without a stdlib-shadowing filename.
import sys as _sys

from . import records as types
from .conservation import assert_net_volume_close as assert_net_volume_close
from .conservation import assert_signed_volume_close as assert_signed_volume_close
from .conservation import combine_signed_volumes as combine_signed_volumes
from .records import HeadContribution as HeadContribution
from .records import MappingResult as MappingResult
from .records import SignedVolume as SignedVolume
from .records import VicCell as VicCell
from .table import ExchangeTable as ExchangeTable
from .validation import _OPTIONAL_FLOAT_COLUMNS as _OPTIONAL_FLOAT_COLUMNS
from .validation import _REQUIRED_COLUMNS as _REQUIRED_COLUMNS
from .validation import _finite_float as _finite_float
from .validation import _immutable as _immutable
from .validation import _integer as _integer
from .validation import _merge_vic_metadata as _merge_vic_metadata
from .validation import _optional_float_value as _optional_float_value
from .validation import _parse_row as _parse_row
from .validation import _reject_duplicate_relations as _reject_duplicate_relations
from .validation import _require_finite as _require_finite
from .validation import _vic_metadata_from_row as _vic_metadata_from_row
from .validation import _vic_vector as _vic_vector

_sys.modules[__name__ + ".types"] = types
