"""Pure numerical postprocessing grouped by the quantity being assessed.

The implementation is organized by responsibility in the sibling modules.
Existing imports remain supported here; new code can import the owning module.
"""

from .exchange import _aggregate_api_volume as _aggregate_api_volume
from .exchange import build_node_boundary_table as build_node_boundary_table
from .exchange import build_vic_exchange_tables as build_vic_exchange_tables
from .groundwater import build_budget_term_table as build_budget_term_table
from .groundwater import build_cell_budget_table as build_cell_budget_table
from .groundwater import build_lateral_tables as build_lateral_tables
from .groundwater import build_mf6_head_tables as build_mf6_head_tables
from .mapping import build_mapping_tables as build_mapping_tables
from .mapping import build_vic_cell_table as build_vic_cell_table
from .mass_balance import build_mf6_mass_balance_table as build_mf6_mass_balance_table
from .mass_balance import enrich_mass_balance_summary as enrich_mass_balance_summary
from .summary import build_post_summary as build_post_summary
from .summary import enrich_post_summary as enrich_post_summary
from .time_series import _direction as _direction
from .time_series import _duration_days as _duration_days
from .time_series import _elapsed_days as _elapsed_days
from .time_series import _finite_or_none as _finite_or_none
from .time_series import _head_at_time as _head_at_time
from .time_series import _is_final_head_row as _is_final_head_row
from .time_series import _signed_from_rows as _signed_from_rows
from .time_series import _signed_from_values as _signed_from_values
from .time_series import cumulative_by_step as cumulative_by_step
