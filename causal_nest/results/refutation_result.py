from dataclasses import dataclass
from typing import Optional


@dataclass
class RefutationResult:
    treatment: str
    estimated_effect: float
    p_value: float
    new_effect: float
    model: Optional[str] = None
    runtime: Optional[float] = None
    passed: bool = False

    def __init__(
        self,
        treatment: str,
        model: Optional[str],
        p_value: float,
        estimated_effect: float,
        new_effect: float,
        passed: bool = None,
    ):
        self.treatment = treatment
        self.model = model
        self.p_value = p_value
        self.estimated_effect = estimated_effect
        self.new_effect = new_effect

        if passed is None:
            self.passed = p_value >= 0.05
