from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EstimationResult:
    """
    Class to store and display the results of a causal estimation process.

    This class is designed to encapsulate the results of a causal estimation process, providing
    a structured way to store and access various attributes related to the estimation. It is
    intended to be used in causal inference workflows where the results of an estimation need
    to be passed around, stored, or analyzed further.

    Attributes:
        model (Optional[str]): The name of the model used for estimation.
        treatment (Optional[str]): The treatment variable in the causal model.
        estimand (Optional[Any]): The estimand, which is the quantity being estimated.
        estimate (Optional[Any]): The estimated value of the estimand.
        control_value (Optional[Any]): The value of the control group.
        treatment_value (Optional[Any]): The value of the treatment group.
        p_value (Optional[Any]): The p-value associated with the estimate.
    """
    model: Optional[str] = None
    treatment: Optional[str] = None
    estimand: Optional[Any] = None
    estimate: Optional[Any] = None
    control_value: Optional[Any] = None
    treatment_value: Optional[Any] = None
    p_value: Optional[Any] = None
