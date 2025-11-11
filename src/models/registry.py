"""Parameter registry for managing all parameter groups."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

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
        error_message: str
    ):
        """Declaratively add cross-group dependency rules."""
        # Store in a list for validation
        if not hasattr(self, '_cross_group_dependencies'):
            self._cross_group_dependencies = []
        
        self._cross_group_dependencies.append({
            'condition_group': condition_group,
            'condition_param': condition_param,
            'condition_value': condition_value,
            'target_group': target_group,
            'required_params': required_params,
            'error_message': error_message
        })

    def validate_cross_group_dependencies(self, configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate all registered cross-group dependencies."""
        errors = []
        
        if not hasattr(self, '_cross_group_dependencies'):
            return errors
        
        for dep in self._cross_group_dependencies:
            condition_config = configs.get(dep['condition_group'], {})
            condition_value = condition_config.get(dep['condition_param'])
            
            if condition_value == dep['condition_value']:
                target_config = configs.get(dep['target_group'], {})
                target_group = self.get_group(dep['target_group'])
                
                for param_key, expected_value in dep['required_params'].items():
                    param_value = target_config.get(param_key)
                    param_obj = target_group.get_parameter(param_key) if target_group else None
                    
                    if param_obj and param_obj.default_value is not None and param_value is None:
                        param_value = param_obj.default_value
                    
                    if param_value != expected_value:
                        errors.append(
                            f"{dep['error_message']} "
                            f"(Expected {param_key}={expected_value}, got {param_value})"
                        )
        
        return errors

