import random
import re

import numpy as np
import pandas as pd
import pytest

from causal_nest.distribution import is_linear, is_normal
from causal_nest.problem import Dataset, FeatureType, FeatureTypeMap


# Normality
def test_is_normal_validates_dataset_as_cn_instance():
    with pytest.raises(ValueError, match=r"Argument 'dataset' must be a causal nest `Dataset` instance"):
        _ = is_normal([1, 2, 3])


def test_is_normal_returns_true_for_valid_normal_data():
    df = pd.DataFrame(data=np.random.normal(0, 5, size=(100, 3)), columns=["foo", "bar", "test"])

    dataset = Dataset(
        data=df,
        target="test",
        feature_mapping=[
            FeatureTypeMap(feature="foo", type=FeatureType.CONTINUOUS),
            FeatureTypeMap(feature="bar", type=FeatureType.CONTINUOUS),
        ],
    )

    result = is_normal(dataset)
    assert result


def test_is_normal_returns_false_for_valid_non_normal_data():
    df = pd.DataFrame(data=np.random.normal(0, 5, size=(100, 3)), columns=["foo", "bar", "test"])
    df["non_normal"] = [(random.randint(0, 200) if random.random() < 0.1 else 0) for _i in range(0, 100)]

    dataset = Dataset(
        data=df,
        target="test",
        feature_mapping=[
            FeatureTypeMap(feature="foo", type=FeatureType.CONTINUOUS),
            FeatureTypeMap(feature="bar", type=FeatureType.CONTINUOUS),
            FeatureTypeMap(feature="non_normal", type=FeatureType.DISCRET),
        ],
    )

    result = is_normal(dataset)
    assert result == False


def test_is_normal_ignores_non_numeric_features():
    df = pd.DataFrame(data=np.random.normal(0, 5, size=(100, 3)), columns=["foo", "bar", "test"])
    df["random_column"] = [random.choice(["a", "b", "c", "d"]) for _i in range(0, 100)]

    dataset = Dataset(
        data=df,
        target="test",
        feature_mapping=[
            FeatureTypeMap(feature="foo", type=FeatureType.CONTINUOUS),
            FeatureTypeMap(feature="bar", type=FeatureType.CONTINUOUS),
            FeatureTypeMap(feature="random_column", type=FeatureType.CATEGORICAL),
        ],
    )

    result = is_normal(dataset)
    assert result
