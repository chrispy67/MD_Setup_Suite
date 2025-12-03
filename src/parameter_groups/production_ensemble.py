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
        description="Method to be used for production simulations",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 5, 6, 7]),
        notes="0=Molecular Dynamics, 1=Energy Minimization, 5=CG, 6=SD+CG, 7=SD+CG+MD"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="PBC_treatment",
        amber_flag="ntb",
        description="Periodic boundary condition",
        param_type=ParameterType.INT,
        category=ParameterCategory.GENERAL,
        notes="0=No periodicity, 1=Constant Volume, 2=Constant Pressure",
        default_value=1
    ))

    group.add_parameter(AmberParameter(
        yaml_key="timestep",
        amber_flag="dt",
        description="Timestep, in ps, of simulation",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.GENERAL,  # NEEDS TO BE CONSISTENT THROUGHOUT SIMULATION ENSEMBLE
        default_value=0.002  # BUT NOT FOR HMASS REPARTITIONING
    ))

    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="nstlim",
        description="Simulation steps",
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
        description="Read in previous coordinates from input file",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=1,
        notes="1=Read coordinates, but not velocities, 5= Read coordinates AND velocities"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="restart_sim",
        amber_flag="irest",
        description="Restart Simulation from provided input file?",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=0,
        notes="0=Do not Restart simulation, 1=read coordinates AND velocities to continue simulation"
    
    ))

    # THIS IS THE MAIN TREATMENT THAT DEFINES THE NPT ENSEMBLE, NOT NECESSARILY THE BAROSTAT TYPE
    group.add_parameter(AmberParameter(
        yaml_key="pressure_control",
        amber_flag="ntp",
        description="Pressure control method (NOT THE BARSOSTAT)",
        param_type=ParameterType.INT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3, 4]),
        notes="0=No pressure control, 1=Isotropic, 2=Anisotropic, 3=Semi-isotropic, 4=MD to target volume (REMD)"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="barostat",
        amber_flag="barostat",
        description="Barostat type",
        param_type=ParameterType.INT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(valid_values=[0, 1]),
        notes="0=Berendsen, 1=Monte Carlo"
    ))

# When ntp > 0
    group.add_parameter(AmberParameter(
        yaml_key="target_pressure",
        amber_flag="pres0",
        description="Target pressure (bar)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(min_value=0.1, max_value=10.0),
        # default_value=1.01325 # 1 atm in bars
    ))

# when ntp > 0
    group.add_parameter(AmberParameter(
        yaml_key="pressure_scaling_factor",
        amber_flag="taup",
        description="Pressure Relaxation Time (ps)",
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
        # default_value=0,
        notes="0=box size scales randomly (x, y, z) each scaling step, 1=x-direction (y, z) fixed, 2=y-direction (x, z) fixed, 3=z-direction (x, y) fixed"
    ))


# THIS DEFINES THE NVT ENSEMBLE
    group.add_parameter(AmberParameter(
        yaml_key="thermostat",
        amber_flag="ntt",
        description="Temperature control method",
        param_type=ParameterType.INT,
        category=ParameterCategory.THERMOSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3]),
        notes="0=Constant energy classical dynamics, 1=Constant temperature (weak coupling), 2=Andersen, 3=Langevian, 9=Optimized Isokinetic Nose-Hoover chain ensemble (OIN), 10=Stochastic Isokinetic Nose-Hoover RESPA integrator, 11=Stochastic Berendsen (Bussi) "
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="temperature",
        amber_flag="temp0",
        description="Target temperature (K)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.GENERAL,  # heating_window[-1] == prod target temperature!
        validation=ParameterValidation(min_value=0.0, max_value=1000.0),
        default_value=300.0
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



    return group