"""VIC adapter and stable window-result types.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

from .global_file import _aggregation_line as _aggregation_line
from .global_file import _is_whole_day as _is_whole_day
from .global_file import _read_positive_int_parameter as _read_positive_int_parameter
from .global_file import _render_global_parameter_file as _render_global_parameter_file
from .global_file import (
    _resolve_exchange_output_prefix as _resolve_exchange_output_prefix,
)
from .global_file import _seconds_since_midnight as _seconds_since_midnight
from .outputs import _netcdf_array as _netcdf_array
from .outputs import _read_window_outputs as _read_window_outputs
from .outputs import _sum_time_axis as _sum_time_axis
from .restart_files import _state_candidates as _state_candidates
from .restart_files import _state_path as _state_path
from .restart_files import _write_head_file as _write_head_file
from .runtime import VicRuntime as VicRuntime
from .runtime import _log as _log
from .types import PreparedVicWindow as PreparedVicWindow
from .types import VicWindowResult as VicWindowResult
