"""
Hydra-based simulation setup script that creates properly labeled directories
with formatted input files and global variables.
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.simulation import SimulationSetup, BuildInputFiles
from src.models import ParameterRegistry
from src.parameter_groups import (
    create_em_parameter_group,
    create_nvt_parameter_group,
    create_workflow_parameter_group
)

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Legacy class definitions removed - now imported from src.simulation
# SimulationSetup and BuildInputFiles are now in src/simulation/setup.py and src/simulation/input_builder.py


@hydra.main(
    version_base="1.2",
    config_path="./config/",
    config_name="simulation_config.yaml"
)
def main(cfg):

    # Create simulation setup instance by building recurisve directories and distributing input cards
    setup = SimulationSetup(cfg)

    # Initialize input files instance to build input files according to registry parameters, dependencies, and cross dependencies 
    input_files = BuildInputFiles(cfg)

    # # HOW THE SCRIPT SHOULD BE RUN TO CREATE NEW DIRECTORY STRUCTURE
    base_path = Path(cfg.directories.base_path)
    
    # # Example usage - build directories for a system
    system_name = "my_protein" 
    optional_string = ""  # Leave empty string if you don't want the optional part in the directory names

    # # Build directories for ALL windows (umbrella sampling)
    print(f"Creating directories for {cfg['global']['windows']} windows...")
    # created_dirs = setup.build_directories( # This will throw an error if the direcotries already exist TODO: Error handle for if a user wants to overwrite, startover, edit, etc. 
    #     system_name=system_name,
    #     window_num=None,  # None means create all windows
    #     optional=optional_string # 
    # )
    # print(f"\nCreated simulation directory structures:")
    # for i, dir_path in enumerate(created_dirs, 0):
    #     print(f"  Window {i}: {dir_path}")


    # # Build registry and validate configuration
    registry = ParameterRegistry()
    registry.add_group(create_em_parameter_group())
    registry.add_group(create_nvt_parameter_group())
    registry.add_group(create_workflow_parameter_group())
    
    # # Validate configuration before proceeding
    # print("\nValidating configuration...")
    
    # # Validate EM parameters
    em_group = registry.get_group("energy_minimization")
    em_config = OmegaConf.to_container(cfg.simulations.em, resolve=True)
    is_valid, errors = em_group.validate_config(em_config)
    
    if not is_valid:
        print("❌ EM configuration errors:")
        for error in errors:
            print(f"  - {error}")
        # return
    else:
        print("✅ EM configuration valid")

    # Simple example of a cross-group dependency for a method to be correct
    registry.add_cross_group_dependency(
        condition_group="workflow", # groups are workflow, energy_minimization, nvt_ensemble SO FAR
        condition_param="water_model",
        condition_value="tip3p",  # Note: lowercase to match validation
        target_group="nvt_ensemble",
        required_params={"Force_calculation": 2, "SHAKE_param": 2},
        error_message="TIP3P Water Model requires NTF=NTC=2!"
    )


    # To enforce that nvt_ensemble.cut matches energy_minimization.nonbonded_cut
    # (i.e., the 'cut' parameter in nvt_ensemble equals 'nonbonded_cut' in energy_minimization),
    # you should fetch the configured value for energy_minimization.nonbonded_cut and use it as condition_value.

    # Example: Dynamically set the condition_value to match current value in config
    
    registry.add_cross_group_dependency(
        condition_group="energy_minimization",
        condition_param="nonbonded_cut",
        condition_value=em_config.get("nonbonded_cut"),
        target_group="nvt_ensemble",
        required_params={"nonbonded_cut": em_config.get("nonbonded_cut")},
        error_message="NVT ensemble nonbonded_cut must match energy minimization nonbonded_cut!"
    )

    # # Validate NVT parameters
    nvt_group = registry.get_group("nvt_ensemble")
    nvt_config = OmegaConf.to_container(cfg.simulations.NVT_ensemble, resolve=True)
    is_valid, errors = nvt_group.validate_config(nvt_config)

    if not is_valid:
        print("❌ NVT configuration errors:")
        for error in errors:
            print(f"  - {error}")
        # return
    else:
        print("✅ Workflow configuration valid")
    

    # GROUPS: 
    # - energy_minimization
    # - nvt_ensemble
    # Generate input files with validated parameters
    # print(registry.get_group("energy_minimization"))
    # print(registry.get_parameter(yaml_key= "method" , group_name="energy_minimization"))

    # print(registry.get_parameter(yaml_key="thermostat", group_name="nvt_ensemble"))


    # print("\nGenerating input files...")
    # input_files.build_em(registry)
    # input_files.build_nvt_equil()



if __name__ == "__main__":
    main()

