from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EstimationResult:
    model: Optional[str] = None
    treatment: Optional[str] = None
    estimand: Optional[Any] = None
    estimate: Optional[Any] = None
    control_value: Optional[Any] = None
    treatment_value: Optional[Any] = None
    p_value: Optional[Any] = None
