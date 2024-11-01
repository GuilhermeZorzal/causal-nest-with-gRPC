from dowhy import CausalModel

from causal_nest.problem import Dataset
from causal_nest.refutation_models.refutation_method_model import RefutationMethodModel
from causal_nest.results import EstimationResult, RefutationResult


class PlaceboPermute(RefutationMethodModel):
    """
    Class for placebo permutation refutation method.

    This class implements the placebo permutation method for refuting causal estimates.
    It inherits from the RefutationMethodModel base class and overrides the refute_estimate method
    to provide the specific implementation for placebo permutation.

    Attributes:
        None
    """

    def refute_estimate(self, dataset: Dataset, estimation_result: EstimationResult, **kwargs):
        """
        Refutes the estimate using placebo permutation.

        Args:
            dataset (Dataset): The dataset to use for refutation.
            estimation_result (EstimationResult): The estimation result to refute.
            **kwargs: Additional arguments for refutation.

        Returns:
            RefutationResult: The result of the refutation.
        """
        model = CausalModel(data=dataset.data, outcome=dataset.target, treatment=estimation_result.treatment)

        r = model.refute_estimate(
            estimation_result.estimand,
            estimation_result.estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
        )

        return RefutationResult(
            treatment=estimation_result.treatment,
            model="PlaceboPermute",
            p_value=r.refutation_result["p_value"],
            estimated_effect=r.estimated_effect,
            new_effect=r.new_effect,
        )
