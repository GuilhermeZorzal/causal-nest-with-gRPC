class RefutationMethodModel:
    """Base class for all causal refutations models"""

    def __init__(self):
        pass

    def refute_estimate(self, dataset, estimation_result, **kwargs):
        """Infer a directed graph out of data.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        raise NotImplementedError
