"""Energy minimization parameter group factory."""

from src.enums import ParameterType, ParameterCategory
from src.models.validation import ParameterValidation
from src.models.parameter import AmberParameter
from src.models.group import ParameterGroup


# Using amber_flag
# description="The simulation will be run for a max of {maxcyc} cycles"

# # Using yaml_key
# description="The simulation will be run for a max of {steps} cycles"

# # Using {value} placeholder
# description="Output will be written every {value} steps"

# # Multiple placeholders
# description="Using {yaml_key} ({amber_flag}) with value {value}"

# {value} - the formatted parameter value
# {yaml_key} - the YAML key name
# {amber_flag} - the AMBER flag name

def create_em_parameter_group() -> ParameterGroup:
    """Create energy minimization parameter group."""
    group = ParameterGroup(
        name="energy_minimization",  # Grouped parameters
        description="Parameters for energy minimization protocol for simulations"
    )
    
    group.add_parameter(AmberParameter(
        yaml_key="MD_method",
        amber_flag="imin",
        description="Method to be used for production simulations",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 5, 6, 7]),
        default_value=1, # for minimization this is true
        notes="""0=Molecular Dynamics, 
        1=Energy Minimization, 
        5=Read in trajectory for analysis using minimization algorithms, 
        6=Read in trajectory for molecular dynamics driver, 
        7=????"""
    ))

    # Add parameters
    group.add_parameter(AmberParameter(
        yaml_key="min_method",
        amber_flag="ntmin",
        description="Minimization method",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3, 4, 5]),
        default_value=1,
        notes=f"""0=Full conjugate gradient minimization 
        1=For cycles, steepest descent method is used, then conjugate gradient is switched on
        2=Steepest Descent only
        3=XMIN method is used 
        4=LMOD method is used
        5=DL-Find module is used"""
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="maxcyc",
        description="The minimization will be run for a maximum of {maxcyc} cycles",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(min_value=1, max_value=1000000),
        default_value=1000
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="restraint", #this boolean will translate when filling out files
        amber_flag="ntr",
        description="Positional restraints: {restraint}",
        param_type=ParameterType.BOOLEAN,
        category=ParameterCategory.RESTRAINT,
        default_value=True
    ))

    group.add_parameter(AmberParameter(
        yaml_key="max_force",
        amber_flag="restraint_wt",
        description="Max force of restraint applied to indicated atoms: {value} (kcal/mol)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.RESTRAINT,
        default_value=10.0,
    ))

    group.add_parameter(AmberParameter(
        yaml_key="output_frequency",
        amber_flag="ntpr",
        description="Data will be written to AMBER files every {value} steps",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=50
    ))

    # This is one of the things that should stay consistent through each simulation?
    group.add_parameter(AmberParameter(
        yaml_key="nonbonded_cut",
        amber_flag="cut",
        description="Nonbonded Cutoff off for VdW interactions: {value} (Å)",
        param_type=ParameterType.FLOAT, 
        category=ParameterCategory.GENERAL,
        default_value=10.0
    ))

    return group

