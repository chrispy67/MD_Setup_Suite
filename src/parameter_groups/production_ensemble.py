from src.enums import ParameterType, ParameterCategory
from src.models.validation import ParameterValidation
from src.models.parameter import AmberParameter
from src.models.group import ParameterGroup
from src.models.dependency import ParameterDependency


def create_production_parameter_group() -> ParameterGroup:
    """Create production parameter group."""
    group = ParameterGroup(
        name="production_ensemble",
        description="Parameters for production ensemble selected by the user"
    )

    group.add_parameter(AmberParameter(
        yaml_key="MD_method",
        amber_flag="imin",
        description="MD Method to be used for production ensemble: {value}",
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
        # default_value=2 # production could be either nvt or npt
    ))

    group.add_parameter(AmberParameter(
        yaml_key="timestep",
        amber_flag="dt",
        description="Timestep for production ensemble: {value} (ps)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.GENERAL,  # NEEDS TO BE CONSISTENT THROUGHOUT SIMULATION ENSEMBLE
        default_value=0.002  # BUT NOT FOR HMASS REPARTITIONING
    ))

    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="nstlim",
        description="Simulation steps for production ensemble: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(min_value=0),
        default_value=100
    ))

    group.add_parameter(AmberParameter(
        yaml_key="nonbonded_cut",
        amber_flag="cut",
        description="Nonbonded Cutoff off for VdW interactions (Å)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.CONTROL,  # THIS MUST BE SET IN EM
        default_value=10.0
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
        description="Target pressure for production ensemble: {value} (bar)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(min_value=0.1, max_value=10.0),
        default_value=1.01325 # 1 atm in bars
    ))

# when ntp > 0
    group.add_parameter(AmberParameter(
        yaml_key="pressure_scaling_factor",
        amber_flag="taup",
        description="Pressure Relaxation Time for production ensemble: {value} (ps)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(min_value=0.1, max_value=10.0),
        # default_value=1.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="Anisotropy_Direction",
        amber_flag="baroscalingdir",
        description="Direction in which to compress or expand box to reach target pressure: {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3]),
        default_value=0,
        notes="""0=box size scales randomly (x, y, z) each scaling step, 
        1=x-direction (y, z) fixed, 
        2=y-direction (x, z) fixed, 
        3=z-direction (x, y) fixed"""
    ))

# THIS DEFINES THE NVT ENSEMBLE
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