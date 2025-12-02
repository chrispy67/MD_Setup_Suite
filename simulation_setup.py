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
    create_workflow_parameter_group,
    create_npt_parameter_group
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
    registry.add_group(create_npt_parameter_group())
    registry.add_group(create_workflow_parameter_group())


    workflow_group = registry.get_group("workflow")
    workflow_config = OmegaConf.to_container(cfg["global"], resolve=True)
    
    is_valid, errors = workflow_group.validate_config(workflow_config)
    if not is_valid:
        print("❌ Workflow configuration errors:")
        for error in errors:
            print(f"  - {error}")
        # return
    else:
        print("✅ Workflow configuration valid")


    # Validate EM parameters
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
        print("✅ NVT configuration valid")


    npt_group = registry.get_group("npt_ensemble")
    npt_config = OmegaConf.to_container(cfg.simulations.NPT_ensemble, resolve=True)
    is_valid, errors = npt_group.validate_config(npt_config)

    if not is_valid:
        print("❌ NPT configuration errors:")
        for error in errors:
            print(f"  - {error}")
        # return
    else:
        print("✅ NPT configuration valid")

    # Simple example of a cross-group dependency for a method to be correct
    # auto_apply_defaults=True means: show warning and automatically apply required values
    # many of these can and will be caught by AMBER, but this is a better way to catch errors before you do a bunch of simulations
    registry.add_cross_group_dependency(
        condition_group="workflow", # groups are workflow, energy_minimization, nvt_ensemble SO FAR
        condition_param="water_model",
        condition_value="tip3p", 
        target_group="nvt_ensemble",
        required_params={"Force_calculation": 2, "SHAKE_param": 2},
        error_message="TIP3P Water Model requires NTF=NTC=2!",
        auto_apply_defaults=True  # Auto-apply defaults and show warnings instead of errors
    )


    # To enforce that nvt_ensemble.cut matches energy_minimization.nonbonded_cut
    # (i.e., the 'cut' parameter in nvt_ensemble equals 'nonbonded_cut' in energy_minimization),
    # you should fetch the configured value for energy_minimization.nonbonded_cut and use it as condition_value.

    # This would be an example of BAD SCIENCE that AMBER would be completely complacent in doing. 
    # what else can we add here?
    # Is there an easier way to check consistencies between ALL groups? This is what it takes to check timestep consistency through an entire workup
    registry.add_cross_group_dependency(
        condition_group="energy_minimization",
        condition_param="nonbonded_cut",
        condition_value=em_config.get("nonbonded_cut"),
        target_group="nvt_ensemble",
        required_params={"nonbonded_cut": em_config.get("nonbonded_cut"),},
        error_message="All simulations must use the same nonbonded cutoff(Å)!!"
    )

    registry.add_cross_group_dependency(
        condition_group="nvt_ensemble",
        condition_param="nonbonded_cut",
        condition_value=nvt_config.get("nonbonded_cut"),
        target_group="npt_ensemble",
        required_params={"nonbonded_cut": nvt_config.get("nonbonded_cut"),},
        error_message="All simulations must use the same nonbonded cutoff(Å)!!"
    )

    registry.add_cross_group_dependency(
        condition_group="npt_ensemble",
        condition_param="nonbonded_cut",
        condition_value=npt_config.get("nonbonded_cut"),
        target_group="production",
        required_params={"nonbonded_cut": npt_config.get("nonbonded_cut"),},
        error_message="All simulations must use the same nonbonded cutoff(Å)!!"
    )

    configs = {
        "energy_minimization": em_config,
        "workflow": workflow_config,
        "nvt_ensemble": nvt_config,
        "npt_ensemble": npt_config,
    }
    
    cross_group_errors, cross_group_warnings = registry.validate_cross_group_dependencies(configs)
    
    if cross_group_warnings:
        print("⚠️  Cross-group dependency warnings (auto-applied defaults):")
        for warning in cross_group_warnings:
            print(f"  - {warning}")
    
    if cross_group_errors:
        print("❌ Cross-group dependency errors:")
        for error in cross_group_errors:
            print(f"  - {error}")
        # return  # Uncomment to stop execution on errors
    else:
        if not cross_group_warnings:  # Only show success if there are no warnings either
            print("✅ All cross-group dependencies satisfied")


    # GROUPS: 
    # - energy_minimization
    # - nvt_ensemble
    # - npt_ensemble
    # - workflow
    # Generate input files with validated parameters
    # print(registry.get_group("energy_minimization"))
    # print(registry.get_parameter(yaml_key= "method" , group_name="energy_minimization"))

    # print(registry.get_parameter(yaml_key="thermostat", group_name="nvt_ensemble"))


    # print("\nGenerating input files...")
    # input_files.build_em(registry)
    # input_files.build_nvt_equil()



if __name__ == "__main__":
    main()

