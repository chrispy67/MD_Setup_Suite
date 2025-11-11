"""Parameter dependency models."""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, Any


class ParameterDependency(BaseModel):
    """Defines a dependency rule between parameters."""
    condition_param: str = Field(..., description="YAML key of the parameter that triggers the dependency")
    condition_value: Any = Field(..., description="Value that must match for dependency to be active")
    required_param: Optional[str] = Field(default=None, description="YAML key of the parameter that must be set/valid (single param)")
    required_params: Optional[Dict[str, Any]] = Field(default=None, description="Dict of {param_key: expected_value} for multiple params")
    required_condition: str = Field(..., description="Condition: 'required', 'required_gt_zero', 'must_be_zero', 'must_equal', etc.")
    error_message: str = Field(..., description="Error message if dependency is violated")
    
    @model_validator(mode='after')
    def validate_required_fields(self):
        """Ensure either required_param or required_params is set, but not both."""
        if self.required_param is None and self.required_params is None:
            raise ValueError("Either 'required_param' or 'required_params' must be provided")
        if self.required_param is not None and self.required_params is not None:
            raise ValueError("Cannot specify both 'required_param' and 'required_params'")
        return self

