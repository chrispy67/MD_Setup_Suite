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
    # Heating control parameters
    group.add_parameter(AmberParameter(
        yaml_key="ramped_heating",
        amber_flag=None,  # Workflow parameter
        description="Enable ramped heating protocol",
        param_type=ParameterType.BOOLEAN,
        category=ParameterCategory.WORKFLOW,
        default_value=False
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="ramps",
        amber_flag=None,  # Workflow parameter
        description="Number of heating ramps",
        param_type=ParameterType.INT,
        category=ParameterCategory.WORKFLOW,
        default_value=0
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="windows",
        amber_flag=None,  # Workflow parameter
        description="Number of umbrella sampling windows",
        param_type=ParameterType.INT,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(min_value=1, max_value=100),
        default_value=10
    ))


    # Universal system settings that should NOT change
    group.add_parameter(AmberParameter(
        yaml_key="force_field",
        amber_flag=None,  # Workflow parameter
        description="Force field name",
        param_type=ParameterType.STRING,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(valid_values=["amber", "charmm", "gromos"]),
        default_value="amber"
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="water_model",
        amber_flag=None,  # Workflow parameter
        description="Water model",
        param_type=ParameterType.STRING,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(valid_values=["TIP3P", "TIP4P", "SPCE"]),
        default_value="tip3p"
    ))
    
    return group

