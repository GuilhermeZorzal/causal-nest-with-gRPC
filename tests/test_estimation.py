import pytest
import pandas as pd
import networkx as nx

from causal_nest.estimation import estimate_model_effects, estimate_effect
from causal_nest.problem import Problem
from causal_nest.dataset import Dataset, FeatureType, FeatureTypeMap
from causal_nest.results import EstimationResult, DiscoveryResult

# def test_estimate_model_effects():
#     df = pd.DataFrame({
#         "treatment": [0, 1, 0, 1, 0, 1],
#         "outcome": [1, 2, 1, 3, 1, 4],
#         "covariate": [5, 6, 5, 6, 5, 6]
#     })

#     dataset = Dataset(
#         data=df,
#         target="outcome",
#         feature_mapping=[
#             FeatureTypeMap(feature="treatment", type=FeatureType.DISCRETE),
#             FeatureTypeMap(feature="outcome", type=FeatureType.CONTINUOUS),
#             FeatureTypeMap(feature="covariate", type=FeatureType.CONTINUOUS),
#         ],
#     )

#     problem = Problem(dataset=dataset)
#     graph = nx.DiGraph()
#     graph.add_edges_from([("treatment", "outcome"), ("covariate", "outcome")])
#     discovery_result = DiscoveryResult(output_graph=graph, model="test_model")

#     response = estimate_model_effects(problem, discovery_result, timeout=180)
#     assert isinstance(response, dict)
#     assert response["model"] == "test_model"
#     assert "results" in response
#     assert isinstance(response["results"], list)


def test_estimate_model_effects_timeout():
    df = pd.DataFrame({
        "treatment": [0, 1, 0, 1, 0, 1],
        "outcome": [1, 2, 1, 3, 1, 4],
        "covariate": [5, 6, 5, 6, 5, 6]
    })

    dataset = Dataset(
        data=df,
        target="outcome",
        feature_mapping=[
            FeatureTypeMap(feature="treatment", type=FeatureType.DISCRETE),
            FeatureTypeMap(feature="outcome", type=FeatureType.CONTINUOUS),
            FeatureTypeMap(feature="covariate", type=FeatureType.CONTINUOUS),
        ],
    )

    problem = Problem(dataset=dataset)
    graph = nx.DiGraph()
    graph.add_edges_from([("treatment", "outcome"), ("covariate", "outcome")])
    discovery_result = DiscoveryResult(output_graph=graph, model="test_model")

    response = estimate_model_effects(problem, discovery_result, timeout=0)  # Set timeout to 0 to trigger timeout
    assert isinstance(response, dict)
    assert response["model"] == "test_model"
    assert "results" in response
    assert isinstance(response["results"], list)
    assert len(response["results"]) == 0  # No results should be returned due to timeout


def test_estimate_effect():
    df = pd.DataFrame({
        "treatment": [0, 1, 0, 1, 0, 1],
        "outcome": [1, 2, 1, 3, 1, 4],
        "covariate": [5, 6, 5, 6, 5, 6]
    })

    dataset = Dataset(
        data=df,
        target="outcome",
        feature_mapping=[
            FeatureTypeMap(feature="treatment", type=FeatureType.DISCRETE),
            FeatureTypeMap(feature="outcome", type=FeatureType.CONTINUOUS),
            FeatureTypeMap(feature="covariate", type=FeatureType.CONTINUOUS),
        ],
    )

    problem = Problem(dataset=dataset)
    graph = nx.DiGraph()
    graph.add_edges_from([("treatment", "outcome"), ("covariate", "outcome")])
    discovery_result = DiscoveryResult(output_graph=graph, model="test_model")

    result = estimate_effect(problem, discovery_result, treatment="treatment")
    assert isinstance(result, EstimationResult)
    assert result.treatment == "treatment"
    assert result.estimand is not None
    assert result.p_value is not None