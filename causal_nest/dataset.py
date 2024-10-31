from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class FeatureType(Enum):
    """The macro type for a feature in the dataset. Although a columnhas its primitive type (int, float, etc) it does not
    provide enough info if it's discrete or continuous, for example."""

    CATEGORICAL = 1
    """Categorical features are values in a predetermined set of allowed values explicitly treated as strings or integers,
    which can be later on encoded."""

    DISCRETE = 2
    """Integer values."""

    CONTINUOUS = 3
    """Real values."""

    IGNORABLE = 4
    """Denotes a feature which will be ignored in causality evaluations. It does not matter the primitive type, the feature
    won't count for any algorithm."""


@dataclass
class FeatureTypeMap:
    """
    Data structure to map a feature to a given type. A dataset may have columns that can be interpreted ambigously.
    To eliminate this problem, this feature mapping exists so it can be used to choose which causal discovery methods are
    allowed and particular stats to run.
    """

    feature: str
    """The column name in dataset referring this map"""

    type: FeatureType
    """The corresponding type for the given feature"""

    importance: float = 0.0
    """The importance score based on feature importance on regresors or classifiers"""

    def __post_init__(self):
        if not isinstance(self.type, FeatureType):
            raise ValueError("Field type must be a FeatureType enum value")


@dataclass
class Dataset:
    """
    Data structure to represent a dataset, its feature mappings and parameters.
    This structure is meant to be used in multiple `Problems`, preventing mutation altogheter in all its operations.
    """

    data: pd.DataFrame
    """The dataframe containing the data.
    Although there are some data handling functions, prefer to input a clean, formatted dataframe."""

    target: str
    """Column name with the target value to be evaluated in causal inference.
    It must belong in the dataframe at the definition."""

    feature_mapping: List[FeatureTypeMap] = field(default_factory=list)
    """A map to detemine the feature types which will be used to evaluate metrics and allowed causal discovery algorithms."""

    def __post_init__(self):
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Field 'data' must be a pandas dataframe")
        if self.target not in self.data.columns:
            raise ValueError("Field 'target' must exist in the dataframe")

        mf = [i.feature for i in self.feature_mapping]
        duplicated = [i for i in set(mf) if mf.count(i) > 1]
        if len(duplicated):
            raise ValueError(f"Field 'feature_mapping' must not have duplicated keys. Found {duplicated}")

        invalid_mappings = [i for i in mf if i not in self.data.columns]
        if len(invalid_mappings):
            raise ValueError(
                f"Field 'feature_mapping' must not have keys that does not belong in the dataset. Found {invalid_mappings}"
            )

        categorical_features = list(filter(lambda x: x.type == FeatureType.CATEGORICAL, self.feature_mapping))
        if len(categorical_features) > 1:
            label_encoder = LabelEncoder()

            for f in categorical_features:
                self.data[f.feature] = label_encoder.fit_transform(self.data[f.feature])


class MissingDataHandlingMethod(Enum):
    """An enumeration of allowed methods for treating missing data."""

    DROP = "dropna"
    """Remove missing values. Uses the `dropna` function in the dataframe."""

    FORWARD_FILL = "ffill"
    """Fill missing values propagating the last valid observation.
    Currently uses the `ffill` function in the pandas dataframe."""

    FORWARD_INTERPOLATION = "interpolation"
    """Fill missing values using a linear forward interpolation.
    Currently uses the `interpolate` function in the pandas dataframe."""


def handle_missing_data(dataset: Dataset, method: MissingDataHandlingMethod = MissingDataHandlingMethod.DROP):
    """Generates a copy of the dataset handling missing values in the dataset.

    Args:
        dataset (Dataset): The dataset definition.
        method (MissingDataHandlingMethod, optional): The method for handling
        missing values. Defaults to MissingDataHandlingMethod.DROP.

    Returns:
        Dataset: A copy of the original dataset with the dataframe missing
        values handled by the provided method.
    """

    def treat_dataframe(df, method):
        if method == MissingDataHandlingMethod.FORWARD_FILL:
            return df.fillna(method="ffill")
        if method == MissingDataHandlingMethod.DROP:
            return df.dropna()
        if method == MissingDataHandlingMethod.FORWARD_INTERPOLATION:
            return df.interpolate(method="linear", limit_direction="forward")

        # TODO: Warn nothing was done
        return df

    updated_df = treat_dataframe(dataset.data, method)
    return replace(dataset, data=updated_df)


def problem_type_for_dataset(dataset: Dataset, unique_threshold=0.05):
    """
    Determine whether the given dataset is suitable for regression or classification.

    Args:
        dataset (Dataset): The dataset definition.

    Returns:
        str: 'regression' if suitable for regression, 'classification' if suitable for classification.
    """
    target_values = dataset.data[dataset.target]

    target_dtype = target_values.dtype

    # Check the unique values in the target column
    unique_values = target_values.unique()

    # Check if all unique target values are integers
    is_integer = all(isinstance(val, int) for val in unique_values)

    # Check if the data type of the target column is object, indicating categorical data
    is_categorical = target_dtype == "object"

    # Check if there are only a few unique values compared to the total number of samples
    unique_ratio = len(unique_values) / len(target_values)
    is_few_unique_values = unique_ratio < unique_threshold

    # Determine the dataset type
    return "classification" if (is_integer and (is_categorical or is_few_unique_values)) else "regression"


def estimate_feature_importances(dataset: Dataset):
    features = list(map(lambda x: x.feature, dataset.feature_mapping))
    x = dataset.data[features]
    y = dataset.data[dataset.target]

    model = (
        RandomForestClassifier() if problem_type_for_dataset(dataset) == "classification" else RandomForestRegressor()
    )
    model.fit(x, y)

    feature_importances = model.feature_importances_

    importances = {f: i for f, i in zip(x.columns, feature_importances)}

    for f in dataset.feature_mapping:
        f.importance = importances[f.feature]

    updated_feature_mapping = [replace(f, importance=importances[f.feature]) for f in dataset.feature_mapping]
    sorted_updated_feature_mapping = sorted(updated_feature_mapping, key=lambda f: f.importance, reverse=True)

    return replace(dataset, feature_mapping=sorted_updated_feature_mapping)


def featured_only_data(dataset: Dataset):
    whitelist = list(map(lambda x: x.feature, dataset.feature_mapping)) + [dataset.target]
    return dataset.data[whitelist]
