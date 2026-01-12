"""
Hydra-based simulation setup script that creates properly labeled directories
with formatted input files and global variables.
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.simulation import SimulationSetup, BuildInputFiles
from src.models import ParameterRegistry
from src.parameter_groups import create_workflow_parameter_group
from src.utils.parameter_display import (
    display_parameter_summary,
    display_simulation_order_summary
)
from src.utils.simulation_mappings import (
    SIMULATION_GROUP_MAPPING,
    get_primary_group_name,
    get_simulation_order
)
from src.utils.registry_helpers import register_simulation_groups

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Legacy class definitions removed - now imported from src.simulation
# SimulationSetup and BuildInputFiles are now in src/simulation/setup.py and src/simulation/input_builder.py

#TODO: Handle 'global' parameter as workflow; the Python reserved word is making this difficult
#TODO: Better error handling with restraint: True/False and string values.
#TODO: Think about different ensembles and extensibility for different simulations and ensembles


@hydra.main(
    version_base="1.2",
    config_path="./config/",
    config_name="simulation_config.yaml"
)
def main(cfg):

    # # Example usage - build directories for a system
    system_name = "1pdb" 
    optional_string = ""  # Leave empty string if you don't want the optional part in the directory names

    # Create simulation setup instance by building recurisve directories and distributing input cards
    setup = SimulationSetup(cfg)

    # Initialize input files instance to build input files according to registry parameters, dependencies, and cross dependencies 
    input_files = BuildInputFiles(cfg, system_name=system_name)
    

    # # Build directories for ALL windows (umbrella sampling)
    print(f"Creating directories for {cfg['global']['windows']} windows...")
    created_dirs = setup.build_directories(
        system_name=system_name,
        window_num=None,  # None means create all windows
        optional=optional_string
    )
    
    if created_dirs:
        print(f"\nCreated simulation directory structures:")
        if isinstance(created_dirs, list):
            for i, dir_path in enumerate(created_dirs, 0):
                print(f"  Window {i}: {dir_path}")
        else:
            print(f"  {created_dirs}")
    else:
        print("\n⚠️  No directories were created (skipped by user or already exist).")


    # # Build registry and validate configuration
    registry = ParameterRegistry()
    
    registry.add_group(create_workflow_parameter_group()) # this is standard
    
    # Get simulation order from YAML configuration
    try:
        simulation_order, yaml_to_canonical = get_simulation_order(cfg)
    except ValueError as e:
        print(f"\n❌ Configuration Error:")
        print(f"  {str(e)}")
        return
    
    # Display simulation order summary
    global_config = OmegaConf.to_container(cfg["global"], resolve=True) if "global" in cfg else {}
    windows = global_config.get("windows", 1)
    display_simulation_order_summary(simulation_order, cfg, windows, yaml_to_canonical)
    
    # Dynamically register simulation parameter groups based on YAML
    registered_groups = register_simulation_groups(registry, simulation_order)
    
    # Validate workflow parameters
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

    display_parameter_summary(
        group=workflow_group, # registry group with associated metadata, defaults, and validation
        config=workflow_config, #config dictionary with stored, RUNTIME values passed by user
        show_only_set=True,
        group_by_category=True,
        title="Workflow Parameters"
    )

    # Dynamically validate and display parameters for each simulation type
    group_configs = {"workflow": workflow_config}
    
    for canonical_key in simulation_order:
        if canonical_key not in SIMULATION_GROUP_MAPPING:
            # This should not happen if get_simulation_order() is called first,
            # but handle it gracefully
            logger.warning(f"Unknown simulation type: {canonical_key}, skipping...")
            continue
        
        mapping = SIMULATION_GROUP_MAPPING[canonical_key]
        group_name_list = mapping["group_name"]
        display_name = mapping["display_name"]
        
        # Get primary group name (first in list, or the string itself)
        primary_group_name = get_primary_group_name(group_name_list)
        
        # Find the original YAML key to access the config
        original_yaml_key = None
        for yaml_key, canon in yaml_to_canonical.items():
            if canon == canonical_key:
                original_yaml_key = yaml_key
                break
        
        # Get the group and config (use original YAML key if found, otherwise canonical key)
        sim_group = registry.get_group(primary_group_name)
        if original_yaml_key and original_yaml_key in cfg.simulations:
            sim_config = OmegaConf.to_container(cfg.simulations[original_yaml_key], resolve=True)
        elif canonical_key in cfg.simulations:
            sim_config = OmegaConf.to_container(cfg.simulations[canonical_key], resolve=True)
        else:
            logger.warning(f"Could not find config for {canonical_key}, skipping...")
            continue
        
        group_configs[primary_group_name] = sim_config
        
        # Validate
        is_valid, errors = sim_group.validate_config(sim_config)
        
        if not is_valid:
            print(f"❌ {display_name} configuration errors:")
            for error in errors:
                print(f"  - {error}")
            # return
        else:
            print(f"✅ {display_name} configuration valid")
        
        # Display parameter summary
        title = f"{display_name} Parameters"
        if display_name == "EM":
            title = "Energy Minimization Parameters"
        elif display_name == "NVT":
            title = "NVT Ensemble Parameters"
        elif display_name == "NPT":
            title = "NPT Ensemble Parameters"
        elif display_name == "Production":
            title = "Production Ensemble Parameters"
        

        ## SUMMARY DISPLAYED FOR EACH SIMULATION TYPE 
        ## SIMULATION ORDER IS IMPLIED BY THE YAML CONFIGURATION AND REFLECTED HERE
        display_parameter_summary(
            group=sim_group,
            config=sim_config,
            show_only_set=True,
            group_by_category=True,
            title=title
        )


    # Convert YAML keys to primary group names for cross-group dependency checking
    # The order is now dynamically determined from the YAML configuration
    selected_simulation_ensemble = [
        get_primary_group_name(SIMULATION_GROUP_MAPPING[yaml_key]["group_name"])
        for yaml_key in simulation_order
        if yaml_key in SIMULATION_GROUP_MAPPING
    ]
    
    # group_configs is already built dynamically above during validation
    # It includes workflow and all simulation groups in the order they appear in YAML

    # Simple example of a cross-group dependency for a method to be correct
    # auto_apply_defaults=True means: show warning and automatically apply required values
    # many of these can and will be caught by AMBER, but this is a better way to catch errors before you do a bunch of simulations
    # Only add if nvt_ensemble is in the selected simulations
    if "nvt_ensemble" in selected_simulation_ensemble:
        registry.add_cross_group_dependency(
            condition_group="workflow", 
            condition_param="water_model",
            condition_value="tip3p", 
            target_group="nvt_ensemble",
            required_params={"Force_calculation": 2, "SHAKE_param": 2},
            error_message="TIP3P Water Model requires NTF=NTC=2!",
            auto_apply_defaults=True  # Auto-apply defaults and show warnings instead of errors
        )

    # While the timestep is checked for consistency throughout the ensemble, this would be the first opprotunity to check for hmass repartitioning
    # Only add if energy_minimization is in the selected simulations
    if "energy_minimization" in selected_simulation_ensemble:
        registry.add_cross_group_dependency(
            condition_group="workflow",
            condition_param="hmass_repart",
            condition_value=True,
            target_group="energy_minimization",
            required_params={"timestep": 0.004},
            error_message="Hydrogen mass repartitioning is primarily used with a 4 femtosecond timestep"
        )

    # To enforce that nvt_ensemble.cut matches energy_minimization.nonbonded_cut
    # (i.e., the 'cut' parameter in nvt_ensemble equals 'nonbonded_cut' in energy_minimization),
    # you should fetch the configured value for energy_minimization.nonbonded_cut and use it as condition_value.

    # This would be an example of BAD SCIENCE that AMBER would be completely complacent in doing. 

    # More elegant approach to check consistencies between all groups in a sequence
    for i in range(len(selected_simulation_ensemble) - 1):
        src = selected_simulation_ensemble[i]
        tgt = selected_simulation_ensemble[i + 1]

        tgt_config = group_configs.get(tgt, {})
        src_config = group_configs.get(src, {})
        cutoff = src_config.get("nonbonded_cut")

        if cutoff is not None:
            registry.add_cross_group_dependency(
                condition_group=src,
                condition_param="nonbonded_cut",
                condition_value=src_config.get("nonbonded_cut"),
                target_group=tgt,
                required_params={"nonbonded_cut": src_config.get("nonbonded_cut")},
                error_message=f"Inconsistencies with nonbonded cutoff used to calculate nonbonded interactions! |  {src} is {src_config.get('nonbonded_cut')}Å and {tgt} is {tgt_config.get('nonbonded_cut')}Å"
            )

    for i in range(len(selected_simulation_ensemble) - 1):
        src = selected_simulation_ensemble[i]
        tgt = selected_simulation_ensemble[i + 1]

        tgt_config = group_configs.get(tgt, {})
        src_config = group_configs.get(src, {})
        timestep = src_config.get("timestep")

        if timestep is not None:
            registry.add_cross_group_dependency(
                condition_group=src,
                condition_param="timestep",
                condition_value=src_config.get("timestep"),
                target_group=tgt,
                required_params={"timestep": src_config.get("timestep")},
                error_message=f"All simulations must use the same timestep (ps)! |  {src} is {src_config.get('timestep')}ps and {tgt} is {tgt_config.get('timestep')}ps"
            )


    cross_group_errors, cross_group_warnings = registry.validate_cross_group_dependencies(group_configs)
    
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
    input_files.build_em(registry)
    input_files.build_nvt_equil(registry)
    input_files.build_npt_equil(registry)



if __name__ == "__main__":
    main()

