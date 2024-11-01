import pytest
from causal_nest.results import DiscoveryResult, EstimationResult, RefutationResult
import networkx as nx


def test_discovery_result_initialization():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C")])
    discovery_result = DiscoveryResult(
        output_graph=graph,
        model="test_model",
        auc_pr=0.85,
        shd=2,
        sid=1,
        runtime=10.5,
        priority_score=0.9,
        knowledge_integrity_score=0.95,
        forbidden_edges_violation_rate=0.1,
        required_edges_compliance_rate=0.9
    )
    assert discovery_result.output_graph == graph
    assert discovery_result.model == "test_model"
    assert discovery_result.auc_pr == 0.85
    assert discovery_result.shd == 2
    assert discovery_result.sid == 1
    assert discovery_result.runtime == 10.5
    assert discovery_result.priority_score == 0.9
    assert discovery_result.knowledge_integrity_score == 0.95
    assert discovery_result.forbidden_edges_violation_rate == 0.1
    assert discovery_result.required_edges_compliance_rate == 0.9


def test_estimation_result_initialization():
    estimation_result = EstimationResult(
        model="test_model",
        treatment="A",
        estimand="ATE",
        estimate=0.5,
        control_value=0.2,
        treatment_value=0.7,
        p_value=0.05
    )
    assert estimation_result.model == "test_model"
    assert estimation_result.treatment == "A"
    assert estimation_result.estimand == "ATE"
    assert estimation_result.estimate == 0.5
    assert estimation_result.control_value == 0.2
    assert estimation_result.treatment_value == 0.7
    assert estimation_result.p_value == 0.05


def test_refutation_result_initialization():
    refutation_result = RefutationResult(
        treatment="A",
        model="test_model",
        p_value=0.05,
        estimated_effect=0.5,
        new_effect=0.3,
        passed=True
    )
    assert refutation_result.treatment == "A"
    assert refutation_result.model == "test_model"
    assert refutation_result.p_value == 0.05
    assert refutation_result.estimated_effect == 0.5
    assert refutation_result.new_effect == 0.3
    assert refutation_result.passed is False


def test_refutation_result_initialization_default_passed():
    refutation_result = RefutationResult(
        treatment="A",
        model="test_model",
        p_value=0.05,
        estimated_effect=0.5,
        new_effect=0.3
    )
    assert refutation_result.passed is True  # Default behavior when p_value is 0.05