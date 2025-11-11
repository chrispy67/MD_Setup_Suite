"""Parameter group model for organizing related parameters."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from src.enums import ParameterCategory
from src.models.parameter import AmberParameter
from src.models.dependency import ParameterDependency


class ParameterGroup(BaseModel):
    """Group of related AMBER parameters."""
    name: str = Field(..., description="Group name")
    description: str = Field(..., description="Group description")
    parameters: List[AmberParameter] = Field(default_factory=list, description="Parameters in this group")
    dependencies: List[ParameterDependency] = Field(default_factory=list, description="Cross-parameter dependency rules")
    
    def add_parameter(self, parameter: AmberParameter):
        """Add a parameter to this group."""
        self.parameters.append(parameter)
    
    # These are going to be consistencies WITHIN 
    def add_dependency(self, dependency: ParameterDependency):
        """Add a dependency rule to this group."""
        self.dependencies.append(dependency)
    
    def get_parameter(self, yaml_key: str) -> Optional[AmberParameter]:
        """Get parameter by YAML key."""
        for param in self.parameters:
            if param.yaml_key == yaml_key:
                return param
        return None
    
    # For error checking by ParameterCategory 
    def get_parameters_by_category(self, category: ParameterCategory) -> List[AmberParameter]:
        """Get all parameters in a specific category."""
        return [param for param in self.parameters if param.category == category]


    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for basic consistencies like typing, ranges, valid values, etc."""
        errors = []
        
        # First, validate individual parameters
        for param in self.parameters:
            if param.yaml_key in config:
                is_valid, error_msg = param.validate_value(config[param.yaml_key])
                if not is_valid:
                    errors.append(f"{param.yaml_key}: {error_msg}")
        
        # Then, validate cross-parameter dependencies
        dependency_errors = self.validate_dependencies(config)
        errors.extend(dependency_errors)
        
        return len(errors) == 0, errors
    
    def validate_dependencies(self, config: Dict[str, Any]) -> List[str]:
        """Validate cross-parameter dependencies based on already chosen parameters."""
        errors = []
        
        for dependency in self.dependencies:
            # Check if condition parameter exists and matches condition value
            condition_value = config.get(dependency.condition_param)
            condition_param_obj = self.get_parameter(dependency.condition_param)
            
            # Apply default if condition parameter not set
            if condition_value is None and condition_param_obj and condition_param_obj.default_value is not None:
                condition_value = condition_param_obj.default_value
            
            if condition_value == dependency.condition_value:
                # Condition is met, check required parameter(s)
                
                # Handle single parameter dependency (backward compatible)
                if dependency.required_param:
                    errors.extend(self._validate_single_param_dependency(
                        dependency, config, dependency.required_param
                    ))
                
                # Handle multiple parameter dependencies
                elif dependency.required_params:
                    errors.extend(self._validate_multi_param_dependency(
                        dependency, config, dependency.required_params
                    ))
        
        return errors
    
    def _validate_single_param_dependency(
        self, 
        dependency: ParameterDependency, 
        config: Dict[str, Any], 
        param_key: str
    ) -> List[str]:
        """Validate a single parameter dependency."""
        errors = []
        required_value = config.get(param_key)
        required_param_obj = self.get_parameter(param_key)
        
        if required_param_obj is None:
            errors.append(f"Dependency error: Required parameter '{param_key}' not found in group")
            return errors
        
        # Apply default value if parameter not in config
        if required_value is None and required_param_obj.default_value is not None:
            required_value = required_param_obj.default_value
        
        # Check dependency condition
        if dependency.required_condition == "required":
            if required_value is None:
                errors.append(dependency.error_message)
        elif dependency.required_condition == "required_gt_zero":
            if required_value is None or (isinstance(required_value, (int, float)) and required_value <= 0):
                errors.append(dependency.error_message)
        elif dependency.required_condition == "must_be_zero":
            if required_value is not None and isinstance(required_value, (int, float)) and required_value != 0:
                errors.append(dependency.error_message)
        elif dependency.required_condition == "must_equal":
            # For must_equal, the expected value should be in the error message or we need to extend the model
            # For now, we'll use a simple approach where required_params dict contains expected values
            pass  # Handled in multi-param validation
        
        return errors
    
    def _validate_multi_param_dependency(
        self,
        dependency: ParameterDependency,
        config: Dict[str, Any],
        required_params: Dict[str, Any]
    ) -> List[str]:
        """Validate multiple parameter dependencies."""
        errors = []
        
        for param_key, expected_value in required_params.items():
            param_value = config.get(param_key)
            param_obj = self.get_parameter(param_key)
            
            if param_obj is None:
                errors.append(f"Dependency error: Required parameter '{param_key}' not found in group")
                continue
            
            # Apply default value if parameter not in config
            if param_value is None and param_obj.default_value is not None:
                param_value = param_obj.default_value
            
            # Check condition based on required_condition type
            if dependency.required_condition == "must_equal":
                if param_value != expected_value:
                    errors.append(f"{dependency.error_message} (Expected {param_key}={expected_value}, got {param_value})")
            elif dependency.required_condition == "required":
                if param_value is None:
                    errors.append(f"{dependency.error_message} (Missing {param_key})")
            elif dependency.required_condition == "required_gt_zero":
                if param_value is None or (isinstance(param_value, (int, float)) and param_value <= 0):
                    errors.append(f"{dependency.error_message} (Parameter {param_key} must be > 0)")
        
        return errors
    
    # This is the main function that checks for a cross depenency or cross dependenCIES BETWEEN GROUPS
    def validate_category_dependencies(self, config: Dict[str, Any], category: ParameterCategory) -> List[str]:
        """
        Validate dependencies for parameters in a specific category.
        
        Filters dependencies where either the condition_param or required_param belongs to the specified category.
        """
        errors = []
        
        # Filter dependencies by category
        for dependency in self.dependencies:
            condition_param_obj = self.get_parameter(dependency.condition_param)
            required_param_obj = None
            
            # Get required param object - handle both single and multiple params
            if dependency.required_param: #singular
                required_param_obj = self.get_parameter(dependency.required_param)
            elif dependency.required_params: #plural
                # For multiple params, check if any belong to the category
                for param_key in dependency.required_params.keys():
                    param_obj = self.get_parameter(param_key)
                    if param_obj and param_obj.category == category:
                        required_param_obj = param_obj
                        break
            
            # Check if either parameter belongs to the specified category
            condition_in_category = condition_param_obj and condition_param_obj.category == category
            required_in_category = required_param_obj and required_param_obj.category == category
            
            if condition_in_category or required_in_category:
                # This dependency is relevant to this category, validate it
                condition_value = config.get(dependency.condition_param)
                
                # Apply default if condition parameter not set
                if condition_value is None and condition_param_obj and condition_param_obj.default_value is not None:
                    condition_value = condition_param_obj.default_value
                
                if condition_value == dependency.condition_value:
                    # Condition is met, check required parameter(s)
                    
                    # Handle single parameter dependency
                    if dependency.required_param:
                        required_value = config.get(dependency.required_param)
                        
                        if required_param_obj is None:
                            errors.append(f"Dependency error: Required parameter '{dependency.required_param}' not found in group")
                            continue
                        
                        # Apply default value if parameter not in config
                        if required_value is None and required_param_obj.default_value is not None:
                            required_value = required_param_obj.default_value
                        
                        # Check dependency condition
                        if dependency.required_condition == "required":
                            if required_value is None:
                                errors.append(dependency.error_message)
                        elif dependency.required_condition == "required_gt_zero":
                            if required_value is None or (isinstance(required_value, (int, float)) and required_value <= 0):
                                errors.append(dependency.error_message)
                        elif dependency.required_condition == "must_be_zero":
                            if required_value is not None and isinstance(required_value, (int, float)) and required_value != 0:
                                errors.append(dependency.error_message)
                    
                    # Handle multiple parameter dependencies
                    elif dependency.required_params:
                        for param_key, expected_value in dependency.required_params.items():
                            param_obj = self.get_parameter(param_key)
                            if param_obj and param_obj.category == category:
                                param_value = config.get(param_key)
                                
                                if param_obj.default_value is not None and param_value is None:
                                    param_value = param_obj.default_value
                                
                                # Check condition based on required_condition type
                                if dependency.required_condition == "must_equal":
                                    if param_value != expected_value:
                                        errors.append(f"{dependency.error_message} (Expected {param_key}={expected_value}, got {param_value})")
                                elif dependency.required_condition == "required":
                                    if param_value is None:
                                        errors.append(f"{dependency.error_message} (Missing {param_key})")
                                elif dependency.required_condition == "required_gt_zero":
                                    if param_value is None or (isinstance(param_value, (int, float)) and param_value <= 0):
                                        errors.append(f"{dependency.error_message} (Parameter {param_key} must be > 0)")
        
        return errors
