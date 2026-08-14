class MultistageError(RuntimeError):
    """Base error for the reproducible multi-stage workflow."""


class ConfigurationError(MultistageError):
    pass


class DataValidationError(MultistageError):
    pass


class GeometryError(MultistageError):
    pass


class MemoryBudgetError(MultistageError):
    pass


class ToughnessSensitivityConfigurationError(ConfigurationError):
    pass
