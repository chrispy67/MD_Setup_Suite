"""AMBER input file builder."""

import os
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, List, Tuple, Any
from omegaconf import DictConfig, OmegaConf
import logging

from src.models.registry import ParameterRegistry
from src.models.group import ParameterGroup
from src.models.parameter import AmberParameter

# Set up logging
logger = logging.getLogger(__name__)


class BuildInputFiles:
    """Build AMBER input files from templates and configuration."""

    # Same notation we have in SimulationSetup, taking in .yaml file with all parameters
    def __init__(self, cfg: DictConfig, registry: Optional[ParameterRegistry] = None, system_name: Optional[str] = None):
        self.cfg = cfg
        self.registry = registry
        
        # Store system_name - use from parameter, config, or default
        if system_name is not None:
            self.system_name = system_name
        else:
            self.system_name = getattr(self.cfg, "system_name", None) or getattr(self.cfg.directories, "system_name", "my_protein")

        # each input file is renamed, filled out, and copied to the input_files_dir from JSON config
        # input_files_dir is relative to the project root (where simulation_setup.py is)
        project_root = Path(__file__).parent.parent.parent
        self.input_files_dir = project_root / cfg["global"]["input_files_dir"]
    
    def _get_parameter_mapping(self, group: ParameterGroup) -> List[Tuple[str, str]]:
        """
        Get YAML key to AMBER flag mapping from a parameter group.
        Returns list of (yaml_key, amber_flag) tuples for parameters with amber_flag.
        """
        mapping = []
        for param in group.parameters:
            if param.amber_flag:  # Skip workflow parameters (those without amber_flag)
                mapping.append((param.yaml_key, param.amber_flag))
        return mapping
    
    def _get_parameter_value(self, param: AmberParameter, config_value: Any) -> Any:
        """
        Get the formatted AMBER value for a parameter.
        Uses the parameter's get_amber_value method for proper formatting.
        """
        if config_value is None and param.default_value is not None:
            config_value = param.default_value
        
        if config_value is None:
            return None
        
        return param.get_amber_value(config_value)
    
    def _replace_template_placeholders(
        self, 
        template_content: str, 
        mapping: List[Tuple[str, str]], 
        config: DictConfig,
        group: ParameterGroup,
        special_values: Optional[dict] = None
    ) -> str:
        """
        Replace placeholders in template using registry parameters.
        
        Args:
            template_content: The template string
            mapping: List of (yaml_key, amber_flag) tuples
            config: Configuration dict/config object
            group: ParameterGroup for accessing parameter definitions
            special_values: Optional dict of special values to override (e.g., temperature windows)
        """
        special_values = special_values or {}
        
        for yaml_key, amber_flag in mapping:
            # Check for special override values first
            if yaml_key in special_values:
                value = special_values[yaml_key]
            elif hasattr(config, yaml_key):
                value = getattr(config, yaml_key)
            else:
                logger.warning(f"Key '{yaml_key}' not found in {group.name} section of config file!")
                continue
            
            # Get parameter object to use its formatting methods
            param = group.get_parameter(yaml_key)
            if param:
                # Use parameter's get_amber_value for proper formatting
                formatted_value = self._get_parameter_value(param, value)
            else:
                # Fallback formatting if parameter not found
                if isinstance(value, bool):
                    formatted_value = 1 if value else 0
                else:
                    formatted_value = value
            
            # Replace the placeholder in the template
            placeholder = f"{{{yaml_key}}}"
            template_content = template_content.replace(placeholder, str(formatted_value))
        
        return template_content

    def build_em(self, registry: Optional[ParameterRegistry] = None):        
        """Build EM input file using validated parameters from registry."""
        
        # Use registry from instance or parameter
        registry = registry or self.registry
        if not registry:
            logger.error("Registry is required to build EM input file")
            return None
        
        # Get parameter group from registry
        em_group = registry.get_group("energy_minimization")
        if not em_group:
            logger.error("energy_minimization group not found in registry")
            return None
        
        # Get parameter mapping from registry
        mapping = self._get_parameter_mapping(em_group)
        
        # Read the template file
        template_path = self.input_files_dir / "min_BLANK.in"
        
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            return None
            
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Get EM configuration from YAML
        em_config = self.cfg.simulations.em
        
        # Replace placeholders using registry-based method
        template_content = self._replace_template_placeholders(
            template_content, mapping, em_config, em_group
        )
        
        # Write the populated content to a new file
        output_path = self.input_files_dir / "min_populated.in"
        with open(output_path, 'w') as f:
            f.write(template_content)
            
        return template_content

    def build_nvt_equil(self, registry: Optional[ParameterRegistry] = None):
        """Build NVT equilibration input files using validated parameters from registry."""
        
        # Use registry from instance or parameter
        registry = registry or self.registry
        if not registry:
            logger.error("Registry is required to build NVT input file")
            return None
        
        # Get parameter group from registry
        nvt_group = registry.get_group("nvt_ensemble")
        if not nvt_group:
            logger.error("nvt_ensemble group not found in registry")
            return None
        
        # Get parameter mapping from registry
        mapping = self._get_parameter_mapping(nvt_group)
        
        ramped_heat = self.cfg.simulations.NVT_ensemble.ramped_heating

        if ramped_heat:

            # Parse JSON file for necessary info
            heat_windows = self.cfg.simulations.NVT_ensemble.ramps
            restraint_strings = self.cfg.simulations.NVT_ensemble.restraint_string
            temp_i = self.cfg.simulations.NVT_ensemble.initial_temperature
            temp_f = self.cfg.simulations.NVT_ensemble.final_temperature
            
            print(f"***NVT equilibration will be done in {heat_windows} steps***")

            # Set arrays from stored values in JSON config
            temp_gradient = (float(temp_f) - float(temp_i)) / int(heat_windows)  # Deg (K) per window

            temp_windows = []
            temp_prev = float(temp_i)

            ## Create temperature gradient pairs for directories heat0, heat1, heat2...
            for i in range(heat_windows):  # [(0.0, 60.0), (60.0, 120.0), (180.0, 240.0), (240.0, 300.0)]
                temp_next = temp_prev + temp_gradient
                temp_windows.append((temp_prev, temp_next))
                temp_prev = temp_next
                

            # Example variables you'd actually draw from config/environment
            base_sim_dir = "./simulations"
            window_heat_dirs = []
            num_windows = self.cfg["global"]["windows"]
            system_name = self.system_name  # Use instance variable instead of getattr

            # Enumerate through each value pair in temp_windows and generate a subdirectory in each NVT/ folder in simulations/my_protein_window_{i}/NVT/heat1, heat2, etc.
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NVT"
                )
                for idx, (t_start, t_end) in enumerate(temp_windows, start=0):

                    heat_dir = os.path.join(window_folder, f"heat{idx}")
                    try:
                        os.makedirs(heat_dir, exist_ok=False)  # FLOW CONTROL FOR OVERWRITING/CREATING NEW FOLDER HIREARCHIES STARTS HERE 
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
            # Ensure restraint_string is a list of same length as temp_windows. If not, handle gracefully.
            restraint_strings = getattr(nvt_config, "restraint_string", [])
            if not isinstance(restraint_strings, (list, tuple)):
                restraint_strings = [restraint_strings]
            if len(restraint_strings) != len(temp_windows):
                print(
                    f"⚠️  restraint_string count ({len(restraint_strings)}) does not match temp_windows count ({len(temp_windows)})."
                    " The last restraint will be used for all remaining windows."
                )
            
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NVT"
                )
                
                for heat_idx, (temp_prev, temp_next) in enumerate(temp_windows):
                    heat_dir = os.path.join(window_folder, f"heat{heat_idx}")
                    
                    # Select the appropriate restraint_string for this window, falling back to last available if not enough values
                    if len(restraint_strings) > 0:
                        restraint_for_window = (
                            restraint_strings[heat_idx]
                            if heat_idx < len(restraint_strings)
                            else restraint_strings[-1]
                        )
                    else:
                        restraint_for_window = ""
                    
                    # Prepare a special_values dict; add restraint_string for this heat window
                    special_values = {
                        "initial_temperature": temp_prev,
                        "final_temperature": temp_next,
                        "restraint_string": restraint_for_window,
                    }
                    
                    # Create a copy of the template for this specific heat window
                    current_template = template_content

                    # Replace placeholders using registry-based method and special_values per window
                    current_template = self._replace_template_placeholders(
                        current_template, mapping, nvt_config, nvt_group, special_values
                    )
                    
                    # Also replace the temperature and restraint_string placeholders directly (for backward compatibility)
                    current_template = current_template.replace("{initial_temperature}", str(temp_prev))
                    current_template = current_template.replace("{final_temperature}", str(temp_next))
                    current_template = current_template.replace("{restraint_string}", str(restraint_for_window))
                    
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
            system_name = self.system_name  # Use instance variable instead of getattr
            num_windows = self.cfg["global"]["windows"]
            
            # Handle temperature based on ramps value
            if nvt_config.ramps == 1:
                # Single temperature case: use 'temperature' instead of initial/final
                temperature = getattr(nvt_config, "temperature", None)
                if temperature is None:
                    logger.error("temperature is required when ramped_heating=False and ramps=1")
                    return
                
                # Ensure restraint_string is an array with length one
                restraint_strings = getattr(nvt_config, "restraint_string", [])
                if not isinstance(restraint_strings, (list, tuple)):
                    restraint_strings = [restraint_strings]
                if len(restraint_strings) != 1:
                    if len(restraint_strings) > 1:
                        logger.warning(f"restraint_string has {len(restraint_strings)} elements, using first element only")
                        restraint_strings = [restraint_strings[0]]
                    else:
                        logger.warning("restraint_string is empty, using NO restraints")
                        restraint_strings = [""]
                
                restraint_for_window = restraint_strings[0]
                
                # Special values for single-step NVT with single temperature
                special_values = {
                    "initial_temperature": temperature,
                    "final_temperature": temperature,
                    "restraint_string": restraint_for_window
                }
            else:
                # Multi-step case: use initial_temperature and final_temperature
                # Ensure restraint_string is an array with length one
                restraint_strings = getattr(nvt_config, "restraint_string", [])
                if not isinstance(restraint_strings, (list, tuple)):
                    restraint_strings = [restraint_strings]
                if len(restraint_strings) != 1:
                    if len(restraint_strings) > 1:
                        logger.warning(f"restraint_string has {len(restraint_strings)} elements, using first element only")
                        restraint_strings = [restraint_strings[0]]
                    else:
                        logger.warning("restraint_string is empty, using empty string")
                        restraint_strings = [""]
                
                restraint_for_window = restraint_strings[0]
                
                # Special values for single-step NVT with temperature ramp
                special_values = {
                    "initial_temperature": nvt_config.initial_temperature,
                    "final_temperature": nvt_config.final_temperature,
                    "restraint_string": restraint_for_window
                }
            
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NVT"
                )
                
                # Create a copy of the template for this window
                current_template = template_content
                
                # Replace placeholders using registry-based method
                current_template = self._replace_template_placeholders(
                    current_template, mapping, nvt_config, nvt_group, special_values
                )
                
                # This still works in the sense that temperatures will be ramped from 0 -> 300K, but quickly. TODO: Figure out if i can just have tempi AMBER value
                # Also replace the temperature and restraint_string placeholders directly (for backward compatibility)
                current_template = current_template.replace("{initial_temperature}", str(special_values["initial_temperature"]))
                current_template = current_template.replace("{final_temperature}", str(special_values["final_temperature"]))
                current_template = current_template.replace("{restraint_string}", str(special_values["restraint_string"]))
                
                # Write the populated content to the NVT directory
                output_path = os.path.join(window_folder, "heat.in")
                with open(output_path, 'w') as f:
                    f.write(current_template)

        return output_path

    def build_npt_equil(self, registry: Optional[ParameterRegistry] = None):
        """Build NPT equilibration input files using validated parameters from registry."""
        
        # Use registry from instance or parameter
        registry = registry or self.registry
        if not registry:
            logger.error("Registry is required to build NPT input file")
            return None
        
        # Get parameter group from registry
        npt_group = registry.get_group("npt_ensemble")
        if not npt_group:
            logger.error("npt_ensemble group not found in registry")
            return None
        
        # Get parameter mapping from registry
        mapping = self._get_parameter_mapping(npt_group)
        
        npt_config = self.cfg.simulations.NPT_ensemble
        ramps = getattr(npt_config, "ramps", 1)
        
        if ramps > 1:
            # Multiple equilibration ramps
            restraint_strings = getattr(npt_config, "restraint_string", [])
            
            print(f"***NPT equilibration will be done in {ramps} steps***")
            
            # Ensure restraint_string is a list
            if not isinstance(restraint_strings, (list, tuple)):
                restraint_strings = [restraint_strings]
            
            if len(restraint_strings) != ramps:
                print(
                    f"⚠️  restraint_string count ({len(restraint_strings)}) does not match ramps count ({ramps})."
                    " The last restraint will be used for all remaining windows."
                )
            
            # Example variables you'd actually draw from config/environment
            base_sim_dir = "./simulations"
            equil_dirs = []
            num_windows = self.cfg["global"]["windows"]
            system_name = self.system_name  # Use instance variable instead of getattr
            
            # Enumerate through each ramp and generate a subdirectory in each NPT/ folder in simulations/my_protein_window_{i}/NPT/equil0, equil1, etc.
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NPT"
                )
                for idx in range(ramps):
                    equil_dir = os.path.join(window_folder, f"equil{idx}")
                    try:
                        os.makedirs(equil_dir, exist_ok=False)  # FLOW CONTROL FOR OVERWRITING/CREATING NEW FOLDER HIREARCHIES STARTS HERE 
                        print(f"Equilibration directory created!: {equil_dir}")
                    except FileExistsError as e:
                        equil_dirs.append(equil_dir)
                        print(f"Failed to create directory, {equil_dir} already exists!")
            
            ## Build input files for each equilibration ramp
            # Read the template file
            template_path = self.input_files_dir / "equil_BLANK.in"
            
            if not template_path.exists():
                logger.error(f"Template file not found: {template_path}")
                return
            
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Create input files for each ramp in each simulation window
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NPT"
                )
                
                for ramp_idx in range(ramps):
                    equil_dir = os.path.join(window_folder, f"equil{ramp_idx}")
                    
                    # Select the appropriate restraint_string for this ramp, falling back to last available if not enough values
                    if len(restraint_strings) > 0:
                        restraint_for_window = (
                            restraint_strings[ramp_idx]
                            if ramp_idx < len(restraint_strings)
                            else restraint_strings[-1]
                        )
                    else:
                        restraint_for_window = ""
                    
                    # Prepare a special_values dict; add restraint_string for this equilibration ramp
                    special_values = {
                        "restraint_string": restraint_for_window,
                    }
                    
                    # Create a copy of the template for this specific equilibration ramp
                    current_template = template_content
                    
                    # Replace placeholders using registry-based method and special_values per window
                    current_template = self._replace_template_placeholders(
                        current_template, mapping, npt_config, npt_group, special_values
                    )
                    
                    # Also replace the restraint_string placeholder directly (for backward compatibility)
                    current_template = current_template.replace("{restraint_string}", str(restraint_for_window))
                    
                    # Write the populated content to the specific equilibration directory
                    output_path = os.path.join(equil_dir, "equil.in")
                    with open(output_path, 'w') as f:
                        f.write(current_template)
        
        else:
            print("**NPT equilibration will be done in one step**")
            
            # Read the template file for single-step NPT
            template_path = self.input_files_dir / "equil_BLANK.in"
            
            if not template_path.exists():
                logger.error(f"Template file not found: {template_path}")
                return
            
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Create input files for each simulation window (single NPT step)
            base_sim_dir = "./simulations"
            system_name = self.system_name  # Use instance variable instead of getattr
            num_windows = self.cfg["global"]["windows"]
            
            # Ensure restraint_string is an array with length one
            restraint_strings = getattr(npt_config, "restraint_string", [])
            if not isinstance(restraint_strings, (list, tuple)):
                restraint_strings = [restraint_strings]
            if len(restraint_strings) != 1:
                if len(restraint_strings) > 1:
                    logger.warning(f"restraint_string has {len(restraint_strings)} elements, using first element only")
                    restraint_strings = [restraint_strings[0]]
                else:
                    logger.warning("restraint_string is empty, using NO restraints")
                    restraint_strings = [""]
            
            restraint_for_window = restraint_strings[0]
            
            # Special values for single-step NPT
            special_values = {
                "restraint_string": restraint_for_window
            }
            
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{system_name}_window_{window_idx}", "NPT"
                )
                
                # Create a copy of the template for this window
                current_template = template_content
                
                # Replace placeholders using registry-based method
                current_template = self._replace_template_placeholders(
                    current_template, mapping, npt_config, npt_group, special_values
                )
                
                # Also replace the restraint_string placeholder directly (for backward compatibility)
                current_template = current_template.replace("{restraint_string}", str(special_values["restraint_string"]))
                
                # Write the populated content to the NPT directory
                output_path = os.path.join(window_folder, "equil.in")
                with open(output_path, 'w') as f:
                    f.write(current_template)
        
        return output_path
    
    def build_prod(self, registry: Optional[ParameterRegistry] = None):
        """Build production input files using validated parameters from registry."""
        
        # Use registry from instance or parameter
        registry = registry or self.registry
        if not registry:
            logger.error("Registry is required to build production input file")
            return None
        
        # Get parameter group from registry
        prod_group = registry.get_group("production")
        if not prod_group:
            logger.warning("production group not found in registry. Production input file building not yet implemented.")
            return None
        
        # Get parameter mapping from registry
        mapping = self._get_parameter_mapping(prod_group)
        
        # TODO: Implement production input file building logic
        # This will follow the same pattern as build_em and build_nvt_equil
        logger.info("Production input file building not yet fully implemented")
        return None

