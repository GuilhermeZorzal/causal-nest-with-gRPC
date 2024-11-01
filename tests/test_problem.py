import re

import pandas as pd
import pytest

from causal_nest.dataset import Dataset
from causal_nest.knowledge import Knowledge
from causal_nest.problem import Problem
from causal_nest.results import DiscoveryResult, EstimationResult, RefutationResult
import networkx as nx


# Problem
def test_problem_validates_dataset_field_as_dataset_instance():
    with pytest.raises(ValueError, match=r"Field 'dataset' must be a CausalNest `Dataset` instance"):
        _ = Problem(dataset=[{"foo": "bar"}], description="Foo")


def test_problem_with_valid_arguments_wont_raise_errors():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    problem = Problem(dataset=dataset, description="Test")
    assert problem


def test_problem_defines_a_default_description():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    problem = Problem(dataset=dataset)
    assert problem.description == ""


def test_problem_validates_knowledge_field_as_knowledge_instance():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    with pytest.raises(ValueError, match=r"Field 'knowledge' must be a CausalNest `Knowledge` instance"):
        _ = Problem(dataset=dataset, knowledge={"foo": "bar"})


def test_problem_with_valid_knowledge_wont_raise_errors():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    knowledge = Knowledge(required_edges=[("foo", "bar")], forbidden_edges=[("bar", "foo")])
    problem = Problem(dataset=dataset, knowledge=knowledge)
    assert problem


def test_problem_initializes_with_default_values():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    problem = Problem(dataset=dataset)
    assert problem.ground_truth is None
    assert problem.discovery_results is None
    assert problem.estimation_results is None
    assert problem.refutation_results is None


def test_problem_allows_setting_ground_truth():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    ground_truth = nx.DiGraph()
    ground_truth.add_edges_from([("foo", "bar")])
    problem = Problem(dataset=dataset, ground_truth=ground_truth)
    assert problem.ground_truth is ground_truth


def test_problem_allows_setting_discovery_results():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    discovery_result = DiscoveryResult(output_graph=nx.DiGraph(), model="test_model")
    problem = Problem(dataset=dataset, discovery_results={"test_model": discovery_result})
    assert problem.discovery_results["test_model"] is discovery_result


def test_problem_allows_setting_estimation_results():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    estimation_result = EstimationResult(model="test_model", treatment="foo", estimand=None, estimate=None)
    problem = Problem(dataset=dataset, estimation_results={"test_model": [estimation_result]})
    assert problem.estimation_results["test_model"][0] is estimation_result


def test_problem_allows_setting_refutation_results():
    dataset = Dataset(data=pd.DataFrame([{"foo": "bar"}]), target="foo")
    refutation_result = RefutationResult(treatment="foo", model="test_model", p_value=0.05, estimated_effect=1.0, new_effect=0.5)
    problem = Problem(dataset=dataset, refutation_results={"test_model": [refutation_result]})
    assert problem.refutation_results["test_model"][0] is refutation_result