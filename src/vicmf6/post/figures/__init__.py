"""Figure generation separated from numerical validation and file loading.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

from .output import _matplotlib as _matplotlib
from .output import _save as _save
from .spatial import _plot_mapping_connectivity as _plot_mapping_connectivity
from .spatial import _plot_mf6_field as _plot_mf6_field
from .spatial import _plot_mf6_head_maps as _plot_mf6_head_maps
from .spatial import _plot_node_boundary_heatmap as _plot_node_boundary_heatmap
from .spatial import _plot_vic_exchange_maps as _plot_vic_exchange_maps
from .spatial import _plot_vic_field as _plot_vic_field
from .suite import create_all_figures as create_all_figures
from .time_series import _plot_conservation_errors as _plot_conservation_errors
from .time_series import _plot_cumulative_exchange as _plot_cumulative_exchange
from .time_series import _plot_daily_net_exchange as _plot_daily_net_exchange
from .time_series import _plot_daily_signed_exchange as _plot_daily_signed_exchange
from .time_series import _plot_head_statistics as _plot_head_statistics
from .time_series import _plot_lateral_flow as _plot_lateral_flow
from .time_series import _plot_node_head_timeseries as _plot_node_head_timeseries
from .time_series import _plot_runtime as _plot_runtime
