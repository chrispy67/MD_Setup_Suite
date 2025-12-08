"""NVT ensemble parameter group factory."""

from src.enums import ParameterType, ParameterCategory
from src.models.validation import ParameterValidation
from src.models.parameter import AmberParameter
from src.models.group import ParameterGroup
from src.models.dependency import ParameterDependency


def create_nvt_parameter_group() -> ParameterGroup:
    """Create NVT ensemble parameter group."""
    group = ParameterGroup(
        name="nvt_ensemble",
        description="Parameters for NVT ensemble equilibrations"
    )

    group.add_parameter(AmberParameter(
        yaml_key="MD_method",
        amber_flag="imin",
        description="MD Method to be used for NVT ensemble: {value}",
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
        default_value=1 # intrinsic to this ensemble
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
        yaml_key="thermostat",
        amber_flag="ntt",
        description="Temperature control method: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.THERMOSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3]),
        notes="""0=Constant energy classical dynamics, 
        1=Constant temperature (weak coupling), 
        2=Andersen, 
        3=Langevian, 
        9=Optimized Isokinetic Nose-Hoover chain ensemble (OIN), 
        10=Stochastic Isokinetic Nose-Hoover RESPA integrator, 
        11=Stochastic Berendsen (Bussi)"""
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="temperature",
        amber_flag="temp0",
        description="Target temperature for NVT ensemble: {value} (K)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.GENERAL,  # heating_window[-1] == prod target temperature!
        validation=ParameterValidation(min_value=0.0, max_value=1000.0),
        default_value=300.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="nstlim",
        description="Simulation steps for NVT ensemble: {value}",
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
        yaml_key="Collision_frequency",
        amber_flag="gamma_ln",
        description="Collision frequency for Langevian thermostat: {value} (ps^-1)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.THERMOSTAT,
        default_value=0.0,
        notes="Required with Langevian thermostat! ntt=3"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="heat_bath_coupling_constant",
        amber_flag="tautp",
        description="Time constant for heat bath coupling for the system {value} (ps))",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.THERMOSTAT,
        default_value=0.0,
        notes="Required with Consntant Temperaure, weak coupling! ntt=1"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="read_prev_coordinates",
        amber_flag="ntx",
        description="Read in previous coordinates from input file: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=1,
        notes="""1=Read coordinates only, 
        5=Read coordinates AND velocities"""
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

    # Add THERMOSTAT category dependency rules using ParameterDependency
    # Declarative approach: Easy to add, modify, or remove dependencies
    # These are for parameteres WITHIN THE SAME GROUP THAT MUST BE CONSISTENT, EASY!
    group.add_dependency(ParameterDependency(
        condition_param="thermostat",
        condition_value=3,  # Langevin thermostat
        required_param="Collision_frequency",
        required_condition="required_gt_zero",
        error_message="THERMOSTAT dependency: 'Collision_frequency' (gamma_ln) must be > 0 when using Langevin thermostat (thermostat=3)"
    ))
    
    group.add_dependency(ParameterDependency(
        condition_param="thermostat",
        condition_value=1,  # Weak coupling thermostat
        required_param="heat_bath_coupling_constant",
        required_condition="required_gt_zero",
        error_message="THERMOSTAT dependency: 'heat_bath_coupling_constant' (tautp) must be > 0 when using weak coupling thermostat (thermostat=1)"
    ))

    return group

