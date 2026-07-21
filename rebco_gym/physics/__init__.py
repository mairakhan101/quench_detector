"""Physics models for REBCO tape thermal, electromagnetic, and sensor response."""

from .thermal import ThermalRunawayModel, Ic_T, lambda_brass
from .quench_antenna import QuenchAntennaModel
from .current_redistribution import CurrentRedistributionMesh
from .electromagnetic import self_field, AnisotropicJcModel, IcDropoutField
from .spatial_tape import SpatialTape, SpatialTapeParams

__all__ = [
    "ThermalRunawayModel",
    "Ic_T",
    "lambda_brass",
    "QuenchAntennaModel",
    "CurrentRedistributionMesh",
    "self_field",
    "AnisotropicJcModel",
    "IcDropoutField",
    "SpatialTape",
    "SpatialTapeParams",
]
