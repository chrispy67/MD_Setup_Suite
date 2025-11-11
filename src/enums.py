"""Enumerations for parameter types and categories."""

from enum import Enum


class ParameterType(str, Enum):
    """Supported parameter types."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "bool"
    LIST = "list"
    DICT = "dict"


class ParameterCategory(str, Enum):
    """Parameter categories for organization."""
    CONTROL = "control"  # control for individual simulations (no consistencies necessary)
    GENERAL = "general"  # CRITICAL parameters that must be consistent through workup

    THERMOSTAT = "thermostat"
    BAROSTAT = "barostat"
    RESTRAINT = "restraint"
    OUTPUT = "output"
    
    # Workflow control categories
    WORKFLOW = "workflow"

