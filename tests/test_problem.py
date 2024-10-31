import re

import pandas as pd
import pytest

from causal_nest.problem import (
    Dataset,
    FeatureType,
    FeatureTypeMap,
    MissingDataHandlingMethod,
    Problem,
    handle_missing_data,
)


# Feature types
def test_feature_type_map_validates_type_as_enum_member():
    with pytest.raises(ValueError, match=r"Field type must be a FeatureType enum value"):
        _ = FeatureTypeMap(feature="foo", type="invalid")


def test_feature_type_map_with_valid_arguments_wont_raise_errors():
    tm = FeatureTypeMap(feature="foo", type=FeatureType.CATEGORICAL)
    assert tm


# Dataset
def test_dataset_validates_data_field_as_pandas_dataframe():
    with pytest.raises(ValueError, match=r"Field 'data' must be a pandas dataframe"):
        _ = Dataset(data=[{"foo": "bar"}], target="foo", feature_mapping=[])


def test_dataset_validates_target_is_column_in_dataframe():
    with pytest.raises(ValueError, match=r"Field 'target' must exist in the dataframe"):
        _ = Dataset(data=pd.DataFrame([{"foo": "invalid"}]), target="bar", feature_mapping=[])


def test_dataset_with_valid_arguments_wont_raise_errors():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo", feature_mapping=[])
    assert dataset


def test_dataset_defines_a_default_feature_mapping():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    assert dataset.feature_mapping == []


def test_dataset_validates_feature_mapping_has_no_duplicate_keys():
    with pytest.raises(
        ValueError, match=re.escape("Field 'feature_mapping' must not have duplicated keys. Found ['foo']")
    ):
        _ = Dataset(
            data=pd.DataFrame([{"foo": "bar", "a": 1}]),
            target="foo",
            feature_mapping=[
                FeatureTypeMap(feature="foo", type=FeatureType.CATEGORICAL),
                FeatureTypeMap(feature="foo", type=FeatureType.CONTINUOUS),
            ],
        )


def test_dataset_validates_feature_mapping_fields_belongs_to_dataset():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Field 'feature_mapping' must not have keys that does not belong in the dataset. Found ['test']"
        ),
    ):
        _ = Dataset(
            data=pd.DataFrame([{"foo": "bar", "a": 1}]),
            target="foo",
            feature_mapping=[
                FeatureTypeMap(feature="foo", type=FeatureType.CATEGORICAL),
                FeatureTypeMap(feature="test", type=FeatureType.CONTINUOUS),
            ],
        )


def test_handle_missing_data_does_not_mutate_the_dataset():
    df = pd.DataFrame(
        [
            {"foo": 1, "bar": 2},
            {"foo": None, "bar": None},
            {"foo": 9, "bar": 8},
        ]
    )

    ds = Dataset(data=df, target="bar")
    assert df.shape[0] == 3

    updated_ds = handle_missing_data(ds, method=MissingDataHandlingMethod.DROP)
    assert df.shape[0] == 3
    assert updated_ds.data.shape[0] == 2


# Problem
def test_problem_validates_dataset_field_as_dataset_instance():
    with pytest.raises(ValueError, match=r"Field 'dataset' must be a causal nest `Dataset` instance"):
        _ = Problem(dataset=[{"foo": "bar"}], description="Foo")


def test_problem_with_valid_arguments_wont_raise_errors():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    problem = Problem(dataset=dataset, description="Test")
    assert problem


def test_problem_defines_a_default_description():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    problem = Problem(dataset=dataset)
    assert problem.description == ""
