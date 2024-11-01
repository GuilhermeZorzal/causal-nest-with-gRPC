from itertools import combinations

import numpy as np
from scipy.stats import normaltest
from sklearn import linear_model

from causal_nest.dataset import Dataset, FeatureType

normality_checkable_types = [
    FeatureType.CONTINUOUS,
    FeatureType.DISCRETE,
]


def is_normal(dataset: Dataset, threshold: float = 0.05):
    """
    Checks if the features in the dataset follow a normal distribution.

    Args:
        dataset (Dataset): The dataset to check for normality.
        threshold (float, optional): The p-value threshold for the normality test. Defaults to 0.05.

    Returns:
        bool: True if all features pass the normality test, False otherwise.

    Raises:
        ValueError: If the provided dataset is not an instance of `Dataset`.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("Argument 'dataset' must be a CausalNest `Dataset` instance")

    features_to_test = [f.feature for f in dataset.feature_mapping if f.type in normality_checkable_types]

    p_values = [normaltest(dataset.data[f].values).pvalue for f in features_to_test]

    return all(pv >= threshold for pv in p_values)


def is_linear(dataset: Dataset):
    """
    Checks if the relationships between features in the dataset are linear.

    Args:
        dataset (Dataset): The dataset to check for linearity.

    Returns:
        bool: True if all feature pairs have a linear relationship, False otherwise.

    Raises:
        ValueError: If the provided dataset is not an instance of `Dataset`.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("Argument 'dataset' must be a CausalNest `Dataset` instance")

    pairs = list(combinations([f.feature for f in dataset.feature_mapping] + [dataset.target], 2))
    return all([check_linearity(dataset.data[x].values, dataset.data[y].values) for x, y in pairs])


def check_linearity(x, y, threshold=0.05):
    """
    Checks if the relationship between two variables is linear.

    Args:
        x (np.ndarray): The first variable.
        y (np.ndarray): The second variable.
        threshold (float, optional): The threshold for the mean of residuals to consider the relationship linear. Defaults to 0.05.

    Returns:
        bool: True if the relationship is linear, False otherwise.
    """
    reshaped_x = x.reshape(-1, 1)
    reshaped_y = y.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(reshaped_x, reshaped_y)
    y_pred = regr.predict(reshaped_x)
    residuals = reshaped_y - y_pred
    return np.mean(residuals) <= threshold
