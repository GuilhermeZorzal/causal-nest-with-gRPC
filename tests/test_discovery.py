import pytest
from unittest.mock import MagicMock, patch
from causal_nest.problem import Problem
from causal_nest.discovery_models import DiscoveryMethodModel
from causal_nest.results import DiscoveryResult
from causal_nest.dataset import Dataset
from causal_nest.knowledge import Knowledge

from causal_nest.discovery import (
    applyable_models,
    discover_with_model,
    discover_with_all_models,
    _run_discover_with_model_task,
)

@pytest.fixture
def mock_problem():
    dataset = MagicMock(spec=Dataset)
    ground_truth = MagicMock()
    knowledge = MagicMock(spec=Knowledge)
    return Problem(dataset=dataset, ground_truth=ground_truth, knowledge=knowledge)

@pytest.fixture
def mock_model():
    model = MagicMock(spec=DiscoveryMethodModel)
    model.__name__ = "MockModel"
    return model

def test_applyable_models(mock_problem):
    mock_problem.dataset = MagicMock(spec=Dataset)
    mock_problem.dataset.target = "target"
    mock_problem.dataset.some_property = True

    with patch("causal_nest.discovery.known_methods", [MagicMock()]) as mock_methods:
        mock_methods[0]().is_method_allowed.return_value = True
        result = applyable_models(mock_problem)
        assert len(result) == 1
        assert result[0] == mock_methods[0]

def test_discover_with_model(mock_problem, mock_model):
    mock_problem.dataset = MagicMock(spec=Dataset)
    mock_problem.dataset.target = "target"
    mock_problem.ground_truth = MagicMock()
    mock_problem.knowledge = MagicMock(spec=Knowledge)
    mock_problem.knowledge.forbidden_edges = []

    mock_model().create_graph_from_data.return_value = MagicMock()

    with patch("causal_nest.discovery.timer", side_effect=[0, 1]):
        with patch("causal_nest.discovery.dagify_graph_v2", return_value=MagicMock()):
            with patch("causal_nest.discovery.calculate_graph_ranking_score", return_value=0.5):
                with patch("causal_nest.discovery.calculate_auc_pr", return_value=0.8):
                    with patch("causal_nest.discovery.calculate_shd", return_value=2):
                        with patch("causal_nest.discovery.calculate_sid", return_value=1):
                            result = discover_with_model(mock_problem, mock_model, verbose=False, orient_toward_target=True)
                            assert isinstance(result, DiscoveryResult)
                            assert result.model == "MockModel"
                            assert result.runtime == 1
                            assert result.priority_score == 0.5
                            assert result.auc_pr == 0.8
                            assert result.shd == 2
                            assert result.sid == 1

def test_run_discover_with_model_task(mock_problem, mock_model):
    args = (mock_problem, mock_model, False, True)
    with patch("causal_nest.discovery.discover_with_model", return_value="result"):
        result = _run_discover_with_model_task(args)
        assert result == "result"


# def test_discover_with_all_models(mock_problem):
#     mock_problem.dataset = MagicMock(spec=Dataset)
#     mock_problem.dataset.target = "target"
#     mock_problem.ground_truth = MagicMock()
#     mock_problem.knowledge = MagicMock()
#     mock_problem.knowledge.forbidden_edges = []

#     with patch("causal_nest.discovery.applyable_models", return_value=[MagicMock()]) as mock_models:
#         with patch("causal_nest.discovery.ProcessPool") as mock_pool:
#             mock_future = MagicMock()
#             mock_future.result.return_value = iter([MagicMock(model="MockModel")])
#             mock_pool.return_value.__enter__.return_value.map.return_value = mock_future

#             result = discover_with_all_models(mock_problem, max_seconds_model=90, verbose=False, max_workers=1, orient_toward_target=True)
#             assert isinstance(result, Problem)
#             assert "MockModel" in result.discovery_results
#             assert result.discovery_results["MockModel"] is not None
#         result = _run_discover_with_model_task(args)
#         assert result == "result"