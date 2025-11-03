"""
Hydra-based simulation setup script that creates properly labeled directories
with formatted input files and global variables.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from src.pydantic_parameter_models import ParameterType, ParameterCategory, ParameterValidation, AmberParameter, ParameterGroup, ParameterRegistry, create_em_parameter_group, create_nvt_parameter_group, create_workflow_parameter_group

# Set up logging
logger = logging.getLogger(__name__)

class SimulationSetup:
    """Class to handle simulation directory setup and file generation/modification."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    def build_directories(self, system_name: str, window_num: int = None, optional: str = "") -> Union[Path, List[Path]]:
        """Build directories for a single window or all windows if window_num is None."""
        
        # Get base path and naming pattern from config
        base_path = Path(self.cfg.directories.base_path)
        naming_pattern = self.cfg.directories.naming_pattern
        subdirectories = self.cfg.directories.subdirectories
        
        created_dirs = []
        
        # If window_num is None, create directories for all windows
        if window_num is None:
            num_windows = self.cfg["global"]["windows"]
            for i in range(0, num_windows):
                dir_path = self._create_single_window_directories(
                    system_name, i, optional, base_path, naming_pattern, subdirectories
                )
                created_dirs.append(dir_path)
        else:
            # Create directory for a specific window
            dir_path = self._create_single_window_directories(
                system_name, window_num, optional, base_path, naming_pattern, subdirectories
            )
            created_dirs.append(dir_path)
        
        return created_dirs if len(created_dirs) > 1 else created_dirs[0]
    
    def _create_single_window_directories(self, system_name: str, window_num: int, optional: str,
                                        base_path: Path, naming_pattern: str, subdirectories: list):
        
        # Build main directory name using the pattern
        # Handle optional underscore: only add it if optional string is not empty
        if optional:
            main_dir_name = naming_pattern.format(
                system_name=system_name,
                window_num=f"window_{window_num}",
                optional=optional
            )

        else:
            # Create pattern without optional part and its underscore
            pattern_without_optional = naming_pattern.replace("_{optional}", "")
            main_dir_name = pattern_without_optional.format(
                system_name=system_name,
                window_num=f"window_{window_num}"
            )
        
        # Create main directory path
        main_dir = base_path / main_dir_name
        
        # Create the main directory | ADD LOGIC FOR RESTARTING/REBUILDING/OVERWRITING SIMULATION DIRECTORY?
        try:
            main_dir.mkdir(parents=True, exist_ok=False)

        except FileExistsError:
            print(f"Error! {main_dir} already exists!")
        
        # Create subdirectories for each simulation type
        for sim_type, sim_config in self.cfg.simulations.items():
            # Get the subdirectory path from config
            
            sim_subdir = main_dir / Path(sim_config.subdirectory).name
            
            # Create simulation-specific directory
            try:
                sim_subdir.mkdir(parents=True, exist_ok=False)
            
            except FileExistsError:
                print(f"Error! {sim_subdir} already exists!")
            
            # Create standard subdirectories within each simulation window
            for subdir in subdirectories:

                try:
                    (sim_subdir / subdir).mkdir(exist_ok=False)
                except FileExistsError:
                    print(f"Error! {subdir} already exists!")
        return main_dir

    def _distribute_input_cards(self, base_path: Path):

        # Source directory containing prepared input files
        input_files_dir = Path(__file__).parent / "input_files"

        if not input_files_dir.exists():
            logger.error(f"Input files directory not found: {input_files_dir}")
            return

        # Define simple distribution rules based on filename prefixes
        # - min_*.in   -> em/
        # - heat_*.in  -> NVT/
        # - equil_*.in -> NPT/
        # - prod_*.in -> prod/ (often in NVT ensemble)
        distribution_rules = [
            ("min_", "em"),
            ("heat_", "NVT"),
            ("equil_", "NPT"),
            ("prod_", "prod")
        ]

        # Iterate over each window directory inside base_path
        for window_dir in base_path.iterdir():
            if not window_dir.is_dir():
                continue

            for prefix, target_subdir_name in distribution_rules:
                destination_dir = window_dir / target_subdir_name # simulations/my_protein_window0/em/min_*...
                if not destination_dir.exists():
                    # Skip silently if the target subdir is not present in this window
                    continue

                for src_file in input_files_dir.glob(f"{prefix}*.in"):
                    dest_file = destination_dir / src_file.name
                    try:
                        shutil.copy2(src_file, dest_file)
                        # logger.info(f"Copied {src_file.name} -> {destination_dir}")
                    except Exception as exc:
                        logger.error(
                            f"Failed to copy {src_file} to {dest_file}: {exc}"
                        )
        return

class BuildInputFiles():

    # Same notation we hvae in SimulationSetup, taking in .yaml file with all parameters
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # each input file is renamed, filled out, and copied to the input_files_dir from JSON config
        self.input_files_dir = Path(__file__).parent / cfg["global"]["input_files_dir"]

    def build_em(self, registry: ParameterRegistry = None):        
        """Build EM input file using validated parameters from registry."""
        
        if registry:
            # Use validated parameters from registry
            em_group = registry.get_group("energy_minimization")
            em_config = OmegaConf.to_container(self.cfg.simulations.em, resolve=True)
            
            # Validate again before building
            # is_valid, errors = em_group.validate_config(em_config)
            # if not is_valid:
            #     logger.error(f"EM validation failed: {errors}")
            #     return
            
            # Use registry parameters for mapping
            EM_yaml_to_amber = []
            for param in em_group.parameters:
                if param.amber_flag:  # Skip workflow parameters
                    EM_yaml_to_amber.append((param.yaml_key, param.amber_flag))
        else:
            # Fallback to hardcoded mapping | This is already in pydantic_parameter_models.py
            EM_yaml_to_amber = [
                ("method", "ntmin"),
                ("restart", "ntx"), 
                ("steps", "maxcyc"), 
                ("output_frequency", "ntpr"),
                ("nonbonded_cut", "cut"),
                ("restraint", "ntr"),
                ("max_force", "restraint_wt"),
                ("restraint_string", "restraintmask")
            ]
        
        # Read the template file
        template_path = self.input_files_dir / "min_BLANK.in"
        
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            return
            
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Get EM configuration from YAML
        em_config = self.cfg.simulations.em
        
        # Enumerate through the mapping list and replace placeholders in _BLANK.in file
        for yaml_key, amber_key in EM_yaml_to_amber:
            # Get the value from the YAML config
            if hasattr(em_config, yaml_key):
                value = getattr(em_config, yaml_key)
                
                # Convert boolean values to appropriate format for AMBER
                if isinstance(value, bool):
                    value = 1 if value else 0
                
                # Replace the placeholder in the template (using YAML key name)
                placeholder = f"{{{yaml_key}}}"
                template_content = template_content.replace(placeholder, str(value))
                
                # logger.info(f"Replaced {placeholder} with {value} (from {yaml_key})")
            else:
                logger.warning(f"Key '{yaml_key}' not found in EM configuration")
        
        # Write the populated content to a new file
        output_path = self.input_files_dir / "min_populated.in"
        with open(output_path, 'w') as f:
            f.write(template_content)
            
        # logger.info(f"Generated populated EM input file: {output_path}")
        return template_content

    def build_nvt_equil(self):

        NVT_yaml_to_amber = [
            ("thermostat", "tcoupl"), # tcoupl is for GROMACS, where is thermostat indicated for amber?
            ("steps", "nstlim"),
            ("timestep", "dt"),
            ("initial_temperature", "temp0"),
            ("final_temperature", "tempi"),
            # ("hmass_repart", "VOID"), # What does this change in the amber inputs?
            ("nonbonded_cut", "cut"),
            ("restraint", "ntr"),
            ("restraint_weight", "restraint_wt"),
            ("restraint_string", "restraintmask")
        ]

        ramped_heat = self.cfg.simulations.NVT_ensemble.ramped_heating

        if ramped_heat:

            # Parse JSON file for necessary info
            heat_windows = self.cfg.simulations.NVT_ensemble.ramps
            temp_i = self.cfg.simulations.NVT_ensemble.initial_temperature
            temp_f = self.cfg.simulations.NVT_ensemble.final_temperature
            
            print(f"***NVT equilibration will be done in {heat_windows} steps***")

            # Set arrays from stored values in JSON config
            temp_gradient = (float(temp_f) - float(temp_i)) / int(heat_windows) # Deg (K) per window

            temp_windows = []
            temp_prev = float(temp_i)

            ## Create temperature gradient pairs for directories heat0, heat1, heat2...
            for i in range(heat_windows): # [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0), (180.0, 240.0), (240.0, 300.0)]
                temp_next = temp_prev + temp_gradient
                temp_windows.append((temp_prev, temp_next))
                temp_prev = temp_next
                

            # Example variables you'd actually draw from config/environment
            base_sim_dir = "./simulations"
            window_heat_dirs = []

            # Enumerate through each value pair in temp_windows and generate a subdirectory in each NVT/ folder in simulations/my_protein_window_{i}/NVT/heat1, heat2, etc.
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NVT"
                )
                for idx, (t_start, t_end) in enumerate(temp_windows, start=0):

                    heat_dir = os.path.join(window_folder, f"heat{idx}")
                    try:
                        os.makedirs(heat_dir, exist_ok=False) # FLOW CONTROL FOR OVERWRITING/CREATING NEW FOLDER HIREARCHIES STARTS HERE 
                        print(f"Heating directory created!: {heat_dir}")
                    except FileExistsError as e:
                        window_heat_dirs.append(heat_dir)
                        print(f"Failed to create directory, {heat_dir} already exists!")


            ## Build input files for each temperature window
            # Read the template file
            template_path = self.input_files_dir / "heat_BLANK.in"
            
            if not template_path.exists():
                logger.error(f"Template file not found: {template_path}")
                return
            
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            nvt_config = self.cfg.simulations.NVT_ensemble
            
            # Create input files for each temperature window in each simulation window
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NVT"
                )
                
                for heat_idx, (temp_prev, temp_next) in enumerate(temp_windows):
                    heat_dir = os.path.join(window_folder, f"heat{heat_idx}")
                    
                    # Create a copy of the template for this specific heat window
                    current_template = template_content

                    # Process each mapping
                    for yaml_key, amber_key in NVT_yaml_to_amber:
                        if yaml_key == "initial_temperature":
                            value = temp_prev
                        elif yaml_key == "final_temperature":
                            value = temp_next
                        elif hasattr(nvt_config, yaml_key):
                            value = getattr(nvt_config, yaml_key)
                        else:
                            logger.warning(f"Key '{yaml_key}' not found in NVT configuration")
                            continue
                        
                        # Convert boolean values to appropriate format for AMBER (WIP)
                        if isinstance(value, bool):
                            value = 1 if value else 0
                        
                        # Replace the placeholder in the template
                        placeholder = f"{{{yaml_key}}}"
                        current_template = current_template.replace(placeholder, str(value))
                    
                    # Also replace the temperature placeholders directly OUTSIDE of the the loop since this depends on directories
                    current_template = current_template.replace("{initial_temperature}", str(temp_prev))
                    current_template = current_template.replace("{final_temperature}", str(temp_next))
                    
                    # Write the populated content to the specific heat directory
                    output_path = os.path.join(heat_dir, "heat.in")
                    with open(output_path, 'w') as f:
                        f.write(current_template)


        else:
            print("**NVT equilibration will be done in one step**")
            
            # Read the template file for single-step NVT
            template_path = self.input_files_dir / "heat_BLANK.in"
            
            if not template_path.exists():
                logger.error(f"Template file not found: {template_path}")
                return
            
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            nvt_config = self.cfg.simulations.NVT_ensemble
            
            # Create input files for each simulation window (single NVT step)
            base_sim_dir = "./simulations"
            system_name = getattr(self.cfg, "system_name", "my_protein")
            num_windows = self.cfg["global"]["windows"]
            
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NVT"
                )
                
                # Create a copy of the template for this window
                current_template = template_content
                
                # Process each mapping
                for yaml_key, amber_key in NVT_yaml_to_amber:
                    if hasattr(nvt_config, yaml_key):
                        value = getattr(nvt_config, yaml_key)
                        
                        # Convert boolean values to appropriate format for AMBER
                        if isinstance(value, bool):
                            value = 1 if value else 0
                        
                        # Replace the placeholder in the template
                        placeholder = f"{{{yaml_key}}}"
                        current_template = current_template.replace(placeholder, str(value))
                    else:
                        logger.warning(f"Key '{yaml_key}' not found in NVT configuration")
                
                # Also replace the temperature placeholders directly for single-step NVT
                current_template = current_template.replace("{initial_temperature}", str(nvt_config.initial_temperature))
                current_template = current_template.replace("{final_temperature}", str(nvt_config.final_temperature))
                
                # Write the populated content to the NVT directory
                output_path = os.path.join(window_folder, "heat.in")
                with open(output_path, 'w') as f:
                    f.write(current_template)
                
                logger.info(f"Generated single-step NVT input file for window {window_idx}: {output_path}")

        return output_path

    def build_npt_equil():

        NPT_yaml_to_amber = [
            ("barostat", "barostat"),
            ("restart", "irest"),
            ("steps", "nstlim"),
            ("timestep", "dt"),
            ("temperature", "temp0"),
            ("SHAKE", ("ntf", "ntc")), # if shake is on, NTF=NTC=2 is pretty standard. this will complicate the loop I have to write later
            ("restraint", "ntr"),
            ("restraint_weight", "restraint_wt"),
            ("restraint_string", "restraintmask")
        ]
        return
    
    def build_prod():
        return


@hydra.main(
    version_base="1.2",
    config_path="./config/",
    config_name="simulation_config.yaml"
)
def main(cfg):
    # Create simulation setup instance
    setup = SimulationSetup(cfg)

    # The order in which I call these functions is really important
    input_files = BuildInputFiles(cfg)

    ## HOW THE SCRIPT SHOULD BE RUN TO CREATE NEW DIRECTORY STRUCTURE
    base_path = Path(cfg.directories.base_path)
    
    # Example usage - build directories for a system
    system_name = "my_protein"
    optional = ""  # Leave empty string if you don't want the optional part in the directory names

    # Build directories for ALL windows (umbrella sampling)
    print(f"Creating directories for {cfg['global']['windows']} windows...")
    created_dirs = setup.build_directories(
        system_name=system_name,
        window_num=None,  # None means create all windows
        optional=optional
    )
    print(f"\nCreated simulation directory structures:")
    for i, dir_path in enumerate(created_dirs, 0):
        print(f"  Window {i}: {dir_path}")
    
    print(f"\nEach window contains:")
    for sim_type, sim_config in cfg.simulations.items():
        print(f"  - {Path(sim_config.subdirectory).name}/ (with analysis/ subdirectory)")
    
    print(f"\nTotal windows created: {len(created_dirs)}")

    # Build registry and validate configuration
    registry = ParameterRegistry()
    registry.add_group(create_em_parameter_group())
    registry.add_group(create_nvt_parameter_group())
    registry.add_group(create_workflow_parameter_group())
    
    # Validate configuration before proceeding
    print("\nValidating configuration...")
    
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
    
    # Validate NVT parameters
    nvt_group = registry.get_group("nvt_ensemble")
    nvt_config = OmegaConf.to_container(cfg.simulations.NVT_ensemble, resolve=True)
    is_valid, errors = nvt_group.validate_config(nvt_config)
    
    if not is_valid:
        print("❌ NVT configuration errors:")
        # These should include issues with NVT config issues IF there are any. That is how I want to use this function
        for error in errors:
            print(f"  - {error}")
        # return
    else:
        print("✅ NVT configuration valid")
    
    # Validate workflow parameters
    workflow_group = registry.get_group("workflow_control")
    workflow_config = OmegaConf.to_container(cfg["global"], resolve=True)
    is_valid, errors = workflow_group.validate_config(workflow_config)
    
    if not is_valid:
        print("❌ Workflow configuration errors:")
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
    print(registry.get_parameter(yaml_key= "method" , group_name="energy_minimization"))

    print(registry.get_parameter(yaml_key="thermostat", group_name="nvt_ensemble"))


    print("\nGenerating input files...")
    # input_files.build_em(registry)
    # input_files.build_nvt_equil()



if __name__ == "__main__":
    main()

