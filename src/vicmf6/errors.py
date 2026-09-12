"""project-specific exceptions used to provide direct failure messages."""


class VicMf6Error(RuntimeError):
    """base class for expected coupling failures."""


class ConfigurationError(VicMf6Error):
    """raised when run configuration is missing or internally inconsistent."""


class ExchangeTableError(VicMf6Error):
    """raised when spatial coupling metadata cannot be trusted."""


class ConservationError(VicMf6Error):
    """raised when a signed interface transfer fails a conservation check."""


class VicRuntimeError(VicMf6Error):
    """raised when a vic coupling window cannot be prepared or accepted."""


class Mf6RuntimeError(VicMf6Error):
    """raised when modflow 6 cannot be initialized, advanced, or diagnosed."""


class CouplingRuntimeError(VicMf6Error):
    """raised when the cross-model orchestration fails."""


class PostprocessingError(VicMf6Error):
    """raised when completed-run outputs cannot be read or summarized safely."""
