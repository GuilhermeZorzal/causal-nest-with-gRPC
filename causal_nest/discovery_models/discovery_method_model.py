from typing import List

from causal_nest.dataset import Dataset, FeatureType
from causal_nest.distribution import is_linear, is_normal


class DiscoveryMethodModel:
    """
    Base class for all causal discovery models.

    This class serves as a base for all causal discovery models that operate on observational data.
    It provides a common interface and structure for discovery models, ensuring consistency and
    ease of extension. The primary feature of this class is the `predict` function that executes
    a function according to the given arguments.

    Attributes:
        allowed_feature_types (List[FeatureType]): List of allowed feature types for the discovery method.
        gaussian_assumption (bool): Indicates if the method assumes the data follows a Gaussian distribution.
        linearity_assumption (bool): Indicates if the method assumes linear relationships between features.
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
        """
        Checks if the provided dataset is valid.

        Args:
            dataset (Dataset): The dataset to validate.

        Raises:
            ValueError: If 'dataset' is not an instance of `Dataset`.
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("Field 'dataset' must be a CausalNest `Dataset` instance")

    def is_method_allowed(self, dataset: Dataset):
        """
        Checks if a given method can be applied to a given dataset.

        It validates as follows, stopping when a falsy condition is met:
            1. Validates the dataset field type mapping with the subclass `allowed_feature_types` field;
            2. Validates if the data is normal if the subclass `gaussian_assumption` field is `True`;
            3. Validates if all feature pairs in the data are linear if the subclass `linearity_assumption` field is `True`;

        Args:
            dataset (Dataset): The dataset to validate.

        Raises:
            ValueError: If the dataset does not meet the required conditions.
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
        """
        Infers a directed graph from the data.

        This method is intended to be overridden by subclasses to provide specific implementations
        for different discovery methods. It raises a NotImplementedError if called directly from
        the base class.

        Args:
            data (Dataset): The dataset to use for graph inference.
            **kwargs: Additional arguments for graph inference.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
