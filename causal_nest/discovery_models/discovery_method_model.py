from typing import List

from causal_nest.dataset import Dataset, FeatureType
from causal_nest.distribution import is_linear, is_normal


class DiscoveryMethodModel:
    """Base class for all causal discovery models.

    Usage for undirected/directed graphs and raw data. All causal discovery
    models out of observational data base themselves on this class. Its main
    feature is the predict function that executes a function according to the
    given arguments.
    """

    allowed_feature_types: List[FeatureType] = list(FeatureType)
    gaussian_assumption: bool = False
    linearity_assumption: bool = False

    def __init__(
        self,
        allowed_feature_types: List[FeatureType] = None,
        gaussian_assumption: bool = None,
        linearity_assumption: bool = None,
    ):
        """Init."""

        if allowed_feature_types is not None:
            # TODO: Check values
            self.allowed_feature_types = allowed_feature_types

        if gaussian_assumption is not None:
            self.gaussian_assumption = gaussian_assumption
        if linearity_assumption is not None:
            self.linearity_assumption = linearity_assumption

    def _check_dataset_valid(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError("Field 'dataset' must be a CausalNest `Dataset` instance")

    def is_method_allowed(self, dataset: Dataset):
        """Checks if a given method can be applied to a a given dataset.

        It validates as it follows, stopping when a falsy condition is met:
            1. Validates the dataset field type mapping with the subclass `allowed_feature_types` field;
            2. Validates if the data is normal if the subclass `gaussian_assumption` field is `True`;
            3. Validates if all feature pairs in the data is linear if the subclass `linearity_assumption` field is `True`;
        """
        self._check_dataset_valid(dataset)

        type_condition = all(fm.type in self.allowed_feature_types for fm in dataset.feature_mapping)
        if not type_condition:
            return False

        if self.gaussian_assumption and not is_normal(dataset):
            return False

        if self.linearity_assumption and not is_linear(dataset):
            return False

        return True

    def create_graph_from_data(self, data: Dataset, **kwargs):
        """Infer a directed graph out of data.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        raise NotImplementedError
