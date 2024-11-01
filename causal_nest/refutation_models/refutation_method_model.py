class RefutationMethodModel:
    """
    Base class for all causal refutation models.

    This class serves as an abstract base class for implementing various causal refutation methods.
    It provides a common interface and structure for refutation models, ensuring consistency and
    ease of extension. The primary method to be implemented by subclasses is `refute_estimate`.

    Attributes:
        None
    """

    def __init__(self):
        pass

    def refute_estimate(self, dataset, estimation_result, **kwargs):
        """
        Refutes the causal estimate using a specific method.

        This method is intended to be overridden by subclasses to provide specific implementations
        for different refutation methods. It raises a NotImplementedError if called directly from
        the base class.

        Args:
            dataset (Dataset): The dataset to use for refutation.
            estimation_result (EstimationResult): The estimation result to refute.
            **kwargs: Additional arguments for refutation.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
