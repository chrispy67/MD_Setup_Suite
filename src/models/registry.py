"""Parameter registry for managing all parameter groups."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple

from src.models.group import ParameterGroup
from src.models.parameter import AmberParameter


class ParameterRegistry(BaseModel):
    """Registry for all AMBER parameter groups."""
    groups: Dict[str, ParameterGroup] = Field(default_factory=dict, description="Parameter groups")
    
    def add_group(self, group: ParameterGroup):
        """Add a parameter group to the registry."""
        self.groups[group.name] = group
    
    def get_group(self, name: str) -> Optional[ParameterGroup]:
        """Get a parameter group by name."""
        return self.groups.get(name)
    
    def get_parameter(self, yaml_key: str, group_name: Optional[str] = None) -> Optional[AmberParameter]:
        """Get a parameter by YAML key."""
        if group_name:
            group = self.get_group(group_name)
            return group.get_parameter(yaml_key) if group else None
        
        # Search all groups
        for group in self.groups.values():
            param = group.get_parameter(yaml_key)
            if param:
                return param
        return None
    
    def search_parameters(self, query: str) -> List[AmberParameter]:
        """Search parameters by description or key."""
        results = []
        query_lower = query.lower()
        
        for group in self.groups.values():
            for param in group.parameters:
                if ((param.yaml_key and query_lower in param.yaml_key.lower()) or 
                    (param.description and query_lower in param.description.lower()) or
                    (param.amber_flag and query_lower in param.amber_flag.lower())):
                    results.append(param)
        
        return results

    def add_cross_group_dependency(
        self,
        condition_group: str,
        condition_param: str,
        condition_value: Any,
        target_group: str,
        required_params: Dict[str, Any],
        error_message: str,
        auto_apply_defaults: bool = False
    ):
        """Declaratively add cross-group dependency rules.
        
        Args:
            condition_group: Name of the group containing the condition parameter
            condition_param: Name of the parameter to check
            condition_value: Value that triggers this dependency
            target_group: Name of the group that must satisfy the dependency
            required_params: Dict of parameter names and their required values
            error_message: Message to display if dependency is not met
            auto_apply_defaults: If True, automatically apply required values and show warnings
                                instead of errors when dependencies aren't met
        """
        # Store in a list for validation
        if not hasattr(self, '_cross_group_dependencies'):
            self._cross_group_dependencies = []
        
        self._cross_group_dependencies.append({
            'condition_group': condition_group,
            'condition_param': condition_param,
            'condition_value': condition_value,
            'target_group': target_group,
            'required_params': required_params,
            'error_message': error_message,
            'auto_apply_defaults': auto_apply_defaults
        })

    def validate_cross_group_dependencies(self, configs: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Validate all registered cross-group dependencies.
        
        Returns:
            Tuple of (errors, warnings). Errors are for dependencies that must be satisfied.
            Warnings are for dependencies with auto_apply_defaults=True that were automatically fixed.
        """
        errors = []
        warnings = []
        
        if not hasattr(self, '_cross_group_dependencies'):
            return errors, warnings
        
        for dep in self._cross_group_dependencies:
            condition_config = configs.get(dep['condition_group'], {})
            condition_value = condition_config.get(dep['condition_param'])
            
            # Get the parameter object to check if it's a string with valid_values
            condition_group_obj = self.get_group(dep['condition_group'])
            condition_param_obj = condition_group_obj.get_parameter(dep['condition_param']) if condition_group_obj else None
            
            # Apply default if condition parameter not set
            if condition_value is None and condition_param_obj and condition_param_obj.default_value is not None:
                condition_value = condition_param_obj.default_value
            
            # Normalize condition_value if it's a string parameter with valid_values
            if condition_param_obj and isinstance(condition_value, str):
                # Use the parameter's conversion method to normalize the value
                try:
                    condition_value = condition_param_obj._convert_value(condition_value)
                except (ValueError, TypeError):
                    pass  # If conversion fails, use original value
            
            # Support both single values and arrays of possible values
            # Use case-insensitive comparison for string values
            condition_matches = False
            if condition_value is None:
                condition_matches = False
            elif isinstance(dep['condition_value'], list):
                # Case-insensitive comparison for string lists
                if all(isinstance(v, str) for v in dep['condition_value']) and isinstance(condition_value, str):
                    normalized_list = [str(v).upper() for v in dep['condition_value']]
                    condition_matches = str(condition_value).upper() in normalized_list
                else:
                    condition_matches = condition_value in dep['condition_value']
            else:
                # Case-insensitive comparison for strings
                if isinstance(condition_value, str) and isinstance(dep['condition_value'], str):
                    condition_matches = str(condition_value).upper() == str(dep['condition_value']).upper()
                else:
                    condition_matches = condition_value == dep['condition_value']
            
            if condition_matches:
                target_config = configs.get(dep['target_group'], {})
                target_group = self.get_group(dep['target_group'])
                auto_apply = dep.get('auto_apply_defaults', False)
                
                for param_key, expected_value in dep['required_params'].items():
                    param_value = target_config.get(param_key)
                    param_obj = target_group.get_parameter(param_key) if target_group else None
                    
                    if param_obj and param_obj.default_value is not None and param_value is None:
                        param_value = param_obj.default_value
                    
                    if param_value != expected_value:
                        if auto_apply:
                            # Apply the required value and add a warning
                            target_config[param_key] = expected_value
                            warnings.append(
                                f"⚠️  {dep['error_message']} "
                                f"Auto-applied {param_key}={expected_value} (was {param_value})"
                            )
                        else:
                            # Add to errors (default behavior)
                            errors.append(
                                f"{dep['error_message']} "
                                f"(Expected {param_key}={expected_value}, got {param_value})"
                            )
        
        return errors, warnings

