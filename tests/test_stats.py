import networkx as nx
import pytest
from networkx import DiGraph
from causal_nest.stats import calculate_auc_pr, calculate_shd, calculate_sid


def test_calculate_auc_pr():
    graph_1 = nx.DiGraph()
    graph_1.add_edges_from([("A", "B"), ("B", "C")])
    
    graph_2 = nx.DiGraph()
    graph_2.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    
    auc_pr = calculate_auc_pr(graph_1, graph_2)
    assert isinstance(auc_pr, float)
    assert 0 <= auc_pr <= 1


def test_calculate_shd():
    graph_1 = nx.DiGraph()
    graph_1.add_edges_from([("A", "B"), ("B", "C")])
    
    graph_2 = nx.DiGraph()
    graph_2.add_edges_from([("A", "B"), ("C", "B")])
    
    shd = calculate_shd(graph_1, graph_2)
    assert isinstance(shd, float)
    assert shd >= 0


def test_calculate_sid():
    graph_1 = nx.DiGraph()
    graph_1.add_edges_from([("A", "B"), ("B", "C")])
    
    graph_2 = nx.DiGraph()
    graph_2.add_edges_from([("A", "B"), ("C", "B")])
    
    sid = calculate_sid(graph_1, graph_2)
    assert isinstance(sid, float)
    assert sid >= 0


def test_calculate_auc_pr_identical_graphs():
    graph_1 = nx.DiGraph()
    graph_1.add_edges_from([("A", "B"), ("B", "C")])
    
    graph_2 = nx.DiGraph()
    graph_2.add_edges_from([("A", "B"), ("B", "C")])
    
    auc_pr = calculate_auc_pr(graph_1, graph_2)
    assert auc_pr == 1.0


def test_calculate_shd_identical_graphs():
    graph_1 = nx.DiGraph()
    graph_1.add_edges_from([("A", "B"), ("B", "C")])
    
    graph_2 = nx.DiGraph()
    graph_2.add_edges_from([("A", "B"), ("B", "C")])
    
    shd = calculate_shd(graph_1, graph_2)
    assert shd == 0


def test_calculate_sid_identical_graphs():
    graph_1 = nx.DiGraph()
    graph_1.add_edges_from([("A", "B"), ("B", "C")])
    
    graph_2 = nx.DiGraph()
    graph_2.add_edges_from([("A", "B"), ("B", "C")])
    
    sid = calculate_sid(graph_1, graph_2)
    assert sid == 0