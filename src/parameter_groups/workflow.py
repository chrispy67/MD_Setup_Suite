"""Workflow control parameter group factory."""

from src.enums import ParameterType, ParameterCategory
from src.models.validation import ParameterValidation
from src.models.parameter import AmberParameter
from src.models.group import ParameterGroup


def create_workflow_parameter_group() -> ParameterGroup:
    """Create workflow control parameter group."""
    group = ParameterGroup(
        name="workflow",
        description="Parameters for workflow control and directory management"
    )

    # Global workflow parameters
    group.add_parameter(AmberParameter(
        yaml_key="windows",
        amber_flag=None,  # Workflow parameter
        description="Number of umbrella sampling windows: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(min_value=1, max_value=100),
        default_value=10
    ))


    # Universal system settings that should NOT change
    group.add_parameter(AmberParameter(
        yaml_key="force_field",
        amber_flag=None,  # Workflow parameter
        description="Force field being used for simulation ensemble: {value}",
        param_type=ParameterType.STRING,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(valid_values=["amber", "charmm", "gromos"]),
        default_value=None
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="water_model",
        amber_flag=None,  # Workflow parameter
        description="Water model being used for simulation ensemble: {value}",
        param_type=ParameterType.STRING,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(valid_values=["TIP3P", "TIP4P", "SPCE"]),
        default_value="TIP3P"
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="hmass_repart",
        amber_flag=None,  # Workflow parameter
        description="Whether to repartition hydrogen masses: {value}",
        param_type=ParameterType.BOOLEAN, # True/False
        category=ParameterCategory.WORKFLOW,
        default_value=False
    ))
    return group

