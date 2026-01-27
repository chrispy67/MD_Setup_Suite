"""AMBER input file builder."""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
import logging

from src.models.registry import ParameterRegistry
from src.models.group import ParameterGroup
from src.models.parameter import AmberParameter

# Set up logging
logger = logging.getLogger(__name__)


class BuildInputFiles:
    """Build AMBER input files directly from registry and validated configs."""

    def __init__(
        self, 
        registry: ParameterRegistry, 
        validated_configs: Dict[str, Dict[str, Any]],
        system_name: Optional[str] = None
    ):
        """
        Initialize input file builder.
        
        Args:
            registry: ParameterRegistry with all parameter groups (must be complete)
            validated_configs: Dictionary mapping group names to their validated config dicts
                              e.g., {"workflow": {...}, "energy_minimization": {...}, ...}
            system_name: Optional system name (defaults to "my_protein")
        """
        self.registry = registry
        self.validated_configs = validated_configs
        self.system_name = system_name or "my_protein"
    
    def _get_group_config(self, group_name: str) -> Optional[Dict[str, Any]]:
        """Get validated config for a parameter group.
        
        Args:
            group_name: The group name (e.g., "energy_minimization", "nvt_ensemble", "workflow")
            
        Returns:
            The validated configuration dict, or None if not found
        """
        return self.validated_configs.get(group_name)
    
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
        config: Dict[str, Any],
        group: ParameterGroup,
        special_values: Optional[dict] = None
    ) -> str:
        """
        Replace placeholders in template using registry parameters.
        
        Args:
            template_content: The template string
            mapping: List of (yaml_key, amber_flag) tuples
            config: Configuration dict with validated values
            group: ParameterGroup for accessing parameter definitions
            special_values: Optional dict of special values to override (e.g., temperature windows)
        """
        special_values = special_values or {}
        
        for yaml_key, amber_flag in mapping:
            # Check for special override values first
            if yaml_key in special_values:
                value = special_values[yaml_key]
            elif yaml_key in config:
                value = config[yaml_key]
            else:
                logger.warning(f"Key '{yaml_key}' not found in {group.name} section of config!")
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

    def build_em(self):
        """
        Build EM input file (em.in) directly from key/value pairs in the registry.
        Only parameters with amber_flag are included (workflow parameters are excluded).
        Parameters are included if they are set in the validated config or have default values.
        """
        # Get parameter group from registry
        em_group = self.registry.get_group("energy_minimization")
        if not em_group:
            logger.error("energy_minimization group not found in registry")
            return None

        # Get validated EM configuration
        em_config = self._get_group_config("energy_minimization")
        if not em_config:
            logger.error("Could not find validated EM configuration")
            return None

        # Collect parameters to be included (only those with amber_flag)
        lines = []
        for param in em_group.parameters:
            # Skip workflow parameters (those without amber_flag)
            if not param.amber_flag:
                continue
            
            # Get value from validated config or default
            value = em_config.get(param.yaml_key)
            if value is None and param.default_value is not None:
                value = param.default_value
            
            # Skip parameter if no value available (not set and no default)
            if value is None:
                continue

            # Format value using parameter's get_amber_value method
            formatted_value = self._get_parameter_value(param, value)
            
            # Format as "amber_flag = formatted_value"
            line = f"{param.amber_flag} = {formatted_value}"
            lines.append(line)

        # Build file content
        content = "\n".join(lines) + "\n"
        
        # Get output directory from config
        subdirectory = em_config.get("subdirectory")
        if not subdirectory:
            logger.error("No 'subdirectory' specified in EM configuration")
            return None
        
        # Create Path object and ensure directory exists
        output_path = Path(subdirectory)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "em.in"

        # Write file
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"EM input file written to: {output_path}")
        return content

    def build_nvt_equil(self):
        """Build NVT equilibration input files using validated parameters from registry."""
        
        # Get parameter group from registry
        nvt_group = self.registry.get_group("nvt_ensemble")
        if not nvt_group:
            logger.error("nvt_ensemble group not found in registry")
            return None
        
        # Get parameter mapping from registry
        mapping = self._get_parameter_mapping(nvt_group)
        
        # Get validated NVT configuration
        nvt_config = self._get_group_config("nvt_ensemble")
        if not nvt_config:
            logger.error("Could not find validated NVT_ensemble configuration")
            return None
        
        ramped_heat = nvt_config.get("ramped_heating")

        if ramped_heat:

            # Parse config for necessary info
            heat_windows = nvt_config.get("ramps")
            temp_i = nvt_config.get("initial_temperature")
            temp_f = nvt_config.get("final_temperature")
            
            # print(f"***NVT equilibration will be done in {heat_windows} steps***")

            # Set arrays from stored values in JSON config
            temp_gradient = (float(temp_f) - float(temp_i)) / int(heat_windows)  # Deg (K) per window

            temp_windows = []
            temp_prev = float(temp_i)

            ## Create temperature gradient pairs for directories heat0, heat1, heat2...
            for i in range(heat_windows):  # [(0.0, 60.0), (60.0, 120.0), (180.0, 240.0), (240.0, 300.0)]
                temp_next = temp_prev + temp_gradient
                temp_windows.append((temp_prev, temp_next))
                temp_prev = temp_next
                

            # Get workflow config for windows
            workflow_config = self._get_group_config("workflow")
            if not workflow_config:
                logger.error("Could not find validated workflow configuration")
                return None
            
            num_windows = workflow_config.get("windows", 1)
            base_sim_dir = "./simulations"
            window_heat_dirs = []

            # Enumerate through each value pair in temp_windows and generate a subdirectory in each NVT/ folder in simulations/my_protein_window_{i}/NVT/heat1, heat2, etc.
            for window_idx in range(0, num_windows):
                window_folder = os.path.join(
                    base_sim_dir, f"{self.system_name}_window_{window_idx}", "NVT"
                )
                for idx, (t_start, t_end) in enumerate(temp_windows, start=0):

                    heat_dir = os.path.join(window_folder, f"heat{idx}")
                    try:
                        os.makedirs(heat_dir, exist_ok=False)  # FLOW CONTROL FOR OVERWRITING/CREATING NEW FOLDER HIREARCHIES STARTS HERE 
                    except FileExistsError as e:
                        window_heat_dirs.append(heat_dir)
                        print(f"Failed to create directory, {heat_dir} already exists!")


            ## Build input files for each temperature window
            # TODO: Refactor to use registry-based approach instead of templates
            # For now, this method still uses templates but will be updated
            logger.warning("build_nvt_equil still uses template files - consider refactoring to registry-based approach")
            
            # Get restraint_string from config
            restraint_strings = nvt_config.get("restraint_string", [])
            if not isinstance(restraint_strings, (list, tuple)):
                restraint_strings = [restraint_strings] if restraint_strings else []

            # Handles if restraints are wanted but no strings provided 
            restraint_flag = nvt_config.get("restraint")
            if restraint_flag is not None and restraint_flag is False and any(rs for rs in restraint_strings):
                print(
                    "⚠️  Warning: 'restraint' is set to False, but 'restraint_string' is populated. "
                    "Restraints will NOT be applied."
                )

            # Handles if restraint string does not match with ramped_window count
            if len(restraint_strings) != len(temp_windows):
                print(
                    f"⚠️  restraint_string count ({len(restraint_strings)}) does not match temp_windows count ({len(temp_windows)})."
                    " The last restraint will be used for all remaining windows."
                )
            
            # TODO: Template-based code removed - refactor to registry-based approach like build_em()
            logger.error("Template-based NVT building with ramped heating not yet refactored to registry-based approach")
            return None


        else:
            # Single-step NVT (non-ramped)
            # TODO: Refactor to registry-based approach like build_em()
            logger.error("Template-based NVT building (non-ramped) not yet refactored to registry-based approach")
            return None

    def build_npt_equil(self):
        """Build NPT equilibration input files using validated parameters from registry."""
        
        # Get parameter group from registry
        npt_group = self.registry.get_group("npt_ensemble")
        if not npt_group:
            logger.error("npt_ensemble group not found in registry")
            return None
        
        # Get parameter mapping from registry
        mapping = self._get_parameter_mapping(npt_group)
        
        # Get validated NPT configuration
        npt_config = self._get_group_config("npt_ensemble")
        if not npt_config:
            logger.error("Could not find validated NPT_ensemble configuration")
            return None
        
        # TODO: Refactor to registry-based approach like build_em()
        logger.error("Template-based NPT building not yet refactored to registry-based approach")
        return None
    
    def build_prod(self):
        """Build production input files using validated parameters from registry."""
        
        # Get parameter group from registry
        prod_group = self.registry.get_group("production")
        if not prod_group:
            logger.warning("production group not found in registry. Production input file building not yet implemented.")
            return None
        
        # Get validated production configuration
        prod_config = self._get_group_config("production")
        if not prod_config:
            logger.error("Could not find validated production configuration")
            return None
        
        # TODO: Implement production input file building logic
        # This will follow the same pattern as build_em
        logger.info("Production input file building not yet fully implemented")
        return None

