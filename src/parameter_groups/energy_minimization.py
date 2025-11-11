"""Energy minimization parameter group factory."""

from src.enums import ParameterType, ParameterCategory
from src.models.validation import ParameterValidation
from src.models.parameter import AmberParameter
from src.models.group import ParameterGroup


def create_em_parameter_group() -> ParameterGroup:
    """Create energy minimization parameter group."""
    group = ParameterGroup(
        name="energy_minimization",  # Grouped parameters
        description="Parameters for energy minimization protocol for simulations"
    )
    
    # Add parameters
    group.add_parameter(AmberParameter(
        yaml_key="min_method",
        amber_flag="ntmin",
        description="Minimization method",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3, 4, 5]),
        # notes="0=Molecular Dynamics, 1=Energy Minimization, 5=CG, 6=SD+CG, 7=SD+CG+MD"
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="maxcyc",
        description="Maximum number of minimization cycles",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(min_value=1, max_value=10000000),
        default_value=1000
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="restraint",
        amber_flag="ntr",
        description="Enable positional restraints",
        param_type=ParameterType.BOOLEAN,
        category=ParameterCategory.RESTRAINT,
        default_value=True
    ))

    group.add_parameter(AmberParameter(
        yaml_key="max_force",
        amber_flag="restraint_wt",
        description="Max force of restraint applied to indicated atoms (kcal/mol)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.RESTRAINT,
        default_value=10.0,
    ))

    group.add_parameter(AmberParameter(
        yaml_key="output_frequency",
        amber_flag="ntpr",
        description="Output frequency in simulation steps",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=50
    ))

    # This is one of the things that should stay consistent through each simulation?
    group.add_parameter(AmberParameter(
        yaml_key="nonbonded_cut",
        amber_flag="cut",
        description="Nonbonded Cutoff off for VdW interactions (Å)",
        param_type=ParameterType.FLOAT, 
        category=ParameterCategory.GENERAL,
        default_value=10.0
    ))

    return group

