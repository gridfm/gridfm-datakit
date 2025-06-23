"""gridfm-datakit - A library for generating synthetic power grid data."""

from gridfm_datakit.generate import (
    generate_power_flow_data,
    generate_power_flow_data_distributed,
)

__version__ = "0.1.0"

__all__ = ["generate_power_flow_data", "generate_power_flow_data_distributed"]
