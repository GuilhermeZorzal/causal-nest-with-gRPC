from dataclasses import dataclass
from typing import Optional


@dataclass
class RefutationResult:
    """
    Class to store and display the results of a causal refutation process.

    This class is designed to encapsulate the results of a causal refutation process, providing
    a structured way to store and access various attributes related to the refutation. It is
    intended to be used in causal inference workflows where the results of a refutation need
    to be passed around, stored, or analyzed further.

    Attributes:
        treatment (str): The treatment variable in the causal model.
        estimated_effect (float): The estimated effect of the treatment.
        p_value (float): The p-value associated with the refutation.
        new_effect (float): The new effect estimated after refutation.
        model (Optional[str]): The name of the model used for refutation.
        runtime (Optional[float]): The runtime of the refutation process in seconds.
        passed (bool): Indicates whether the refutation passed based on the p-value.
    """

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
