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
    if not isinstance(dataset, Dataset):
        raise ValueError("Argument 'dataset' must be a CausalNest `Dataset` instance")

    features_to_test = [f.feature for f in dataset.feature_mapping if f.type in normality_checkable_types]

    p_values = [normaltest(dataset.data[f].values).pvalue for f in features_to_test]

    return all(pv >= threshold for pv in p_values)


def is_linear(dataset: Dataset):
    if not isinstance(dataset, Dataset):
        raise ValueError("Argument 'dataset' must be a CausalNest `Dataset` instance")

    pairs = list(combinations([f.feature for f in dataset.feature_mapping] + [dataset.target], 2))
    return all([check_linearity(dataset.data[x].values, dataset.data[y].values) for x, y in pairs])


def check_linearity(x, y, threshold=0.05):
    reshaped_x = x.reshape(-1, 1)
    reshaped_y = y.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(reshaped_x, reshaped_y)
    y_pred = regr.predict(reshaped_x)
    residuals = reshaped_y - y_pred
    return np.mean(residuals) <= threshold
