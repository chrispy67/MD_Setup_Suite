"""NPT ensemble parameter group factory."""

from src.enums import ParameterType, ParameterCategory
from src.models.validation import ParameterValidation
from src.models.parameter import AmberParameter
from src.models.group import ParameterGroup
from src.models.dependency import ParameterDependency


def create_npt_parameter_group() -> ParameterGroup:
    """Create NPT ensemble parameter group."""
    group = ParameterGroup(
        name="npt_ensemble",
        description="Parameters for NPT ensemble equilibrations"
    )

    group.add_parameter(AmberParameter(
        yaml_key="MD_method",
        amber_flag="imin",
        description="MD Method to be used for NPT ensemble: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 5, 6, 7]),
        notes="""0=Molecular Dynamics, 
        1=Energy Minimization, 
        5=Read in trajectory for analysis using minimization algorithms, 
        6=Read in trajectory for molecular dynamics driver, 
        7=????"""    
        ))

    group.add_parameter(AmberParameter(
        yaml_key="PBC_treatment",
        amber_flag="ntb",
        description="Periodic boundary condition is set to {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.GENERAL,
        notes="0=No periodicity, 1=Constant Volume, 2=Constant Pressure",
        default_value=2 # intrinsic to this ensemble
    ))

    group.add_parameter(AmberParameter(
        yaml_key="timestep",
        amber_flag="dt",
        description="Timestep for NVT ensemble: {value} (ps)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.GENERAL,  # NEEDS TO BE CONSISTENT THROUGHOUT SIMULATION ENSEMBLE
        default_value=0.002  # BUT NOT FOR HMASS REPARTITIONING
    ))

    group.add_parameter(AmberParameter(
        yaml_key="Force_calculation",
        amber_flag="ntf",
        description="Forces to be calculated: {value}",  # this should change between equil -> prod
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        notes=""""1=all interactions calculated, 
        2=bond interactions including H omitted(NTC=2) 
        3=all bond interactions are omitted (NTC=3) 
        4=Angles involving H-atom and all bonds omitted
        5=Bond and Angle interactions omitted 
        6=Dihedrals involving H-atoms omitted 
        7=Bond, Angle and Dihedral interactions omitted
        8=Bond, Angle, Dihedral, AND nonbonded interactions ommitted"""
    ))

    group.add_parameter(AmberParameter(
        yaml_key="SHAKE_param",
        amber_flag="ntc",
        description="SHAKE constraints for equilibrations: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,  # Generally turned off during production?
        default_value=2,  # Is this default for heating sims??
        notes="""1=No SHAKE constraints, 
        2=Hydrogen bonds constrained 
        3=All bonds constrainted"""
    ))

    group.add_parameter(AmberParameter(
        yaml_key="nonbonded_cut",
        amber_flag="cut",
        description="Nonbonded Cutoff off for VdW interactions: {value} (Å)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.CONTROL,  # THIS MUST BE SET IN EM
        default_value=10.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="nstlim",
        description="Simulation steps for NPT ensemble: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(min_value=0),
        default_value=100
    ))

    group.add_parameter(AmberParameter(
        yaml_key="restraint",
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
        yaml_key="restraint_string",
        amber_flag="restraintmask",
        description="Atom selection rules for restraints: {value}",
        param_type=ParameterType.RESTRAINT_STRING_ARRAY,
        category=ParameterCategory.RESTRAINT,
        default_value=[],
        notes="Array of AMBER atom selection strings (e.g., [':1-NUMRES@CA,C,N,O,PA,PB,Mg,MG'])"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="read_prev_coordinates",
        amber_flag="ntx",
        description="Read in previous coordinates from input file: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=1,
        notes="""1=Read coordinates only, 
        5= Read coordinates AND velocities"""
    ))

    group.add_parameter(AmberParameter(
        yaml_key="restart_sim",
        amber_flag="irest",
        description="Restart Simulation from provided input file: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=0,
        notes="""0=Do not Restart simulation, 
        1=read coordinates AND velocities to continue simulation (NTX=5)"""
    ))


    # THIS IS THE MAIN TREATMENT THAT DEFINES THE NPT ENSEMBLE, NOT NECESSARILY THE BAROSTAT TYPE
    group.add_parameter(AmberParameter(
        yaml_key="pressure_control",
        amber_flag="ntp",
        description="Pressure control method (NOT THE BARSOSTAT): {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3, 4]),
        notes="""0=No pressure control, 
        1=Isotropic, 
        2=Anisotropic, 
        3=Semi-isotropic,
        4=MD to target volume (REMD)"""
    ))

    group.add_parameter(AmberParameter(
        yaml_key="barostat",
        amber_flag="barostat",
        description="Barostat type: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(valid_values=[0, 1]),
        notes="0=Berendsen, 1=Monte Carlo"
    ))

# When ntp > 0
    group.add_parameter(AmberParameter(
        yaml_key="target_pressure",
        amber_flag="pres0",
        description="Target pressure for NPT ensemble: {value} (bar)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(min_value=0.1, max_value=10.0),
        default_value=1.01325 # 1 atm in bars
    ))

# when ntp > 0
    group.add_parameter(AmberParameter(
        yaml_key="pressure_scaling_factor",
        amber_flag="taup",
        description="Pressure Relaxation Time for NPT ensemble: {value} (ps)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(min_value=0.1, max_value=10.0),
        # default_value=1.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="Anisotropy_Direction",
        amber_flag="baroscalingdir",
        description="Direction in which to compress or expand box to reach target pressure",
        param_type=ParameterType.INT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3]),
        default_value=0,
        notes="""0=box size scales randomly (x, y, z) each scaling step, 
        1=x-direction (y, z) fixed, 
        2=y-direction (x, z) fixed, 
        3=z-direction (x, y) fixed"""
    ))

    # ADD BAROSTAT CATEGORY DEPENDECY RULES USING ParameterDependency
    # Declarative approach: Easy to add, modify, or remove dependencies
    # These are for parameteres WITHIN THE SAME GROUP THAT MUST BE CONSISTENT, EASY!

    group.add_dependency(ParameterDependency(
        condition_param="pressure_control",
        condition_value=[1, 2, 3, 4],  # required when controlling pressure at all
        required_param="target_pressure",
        required_condition="required",
        error_message="Target pressure is required when controlling pressure at all"
    ))


    # Pressure Scaling Factor is required and implied in NPT ensemble 
    group.add_dependency(ParameterDependency(
        condition_param="pressure_control",
        condition_value=[1, 2, 3, 4],  # Scaling Factor
        required_param="pressure_scaling_factor",
        required_condition="required",
        error_message="Pressure relaxation time (in ps) is required when controlling pressure at all"
    ))

    # If isotropically controling pressure, the direction of the box is necessary
    group.add_dependency(ParameterDependency(
        condition_param="pressure_control",
        condition_value=2,  # Only for use in anisotropic pressure control
        required_param="Anisotropy_Direction",
        required_condition="required",
        error_message="Anisotropy direction is required when controlling pressure at all"
    ))

    # # MUST USE MC BAROSTAT
    group.add_dependency(ParameterDependency(
        condition_param="pressure_control",
        condition_value=3,  # Only for use in semi-isotropic pressure control
        required_param="barostat",
        required_condition="required",
        error_message="Anisotropy direction is required when controlling pressure at all"
    ))

    return group