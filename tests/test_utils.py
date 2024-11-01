import pytest
import networkx as nx
from causal_nest.utils import graph_to_pydot_string, dagify_graph


def test_graph_to_pydot_string():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C")])
    pydot_string = graph_to_pydot_string(graph)
    assert isinstance(pydot_string, str)
    assert 'digraph' in pydot_string
    assert 'A -> B' in pydot_string
    assert 'B -> C' in pydot_string


def test_dagify_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])  # This creates a cycle
    dagified_graph = dagify_graph(graph)
    assert isinstance(dagified_graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(dagified_graph)
    assert dagified_graph.has_edge("A", "B")
    assert dagified_graph.has_edge("B", "C")
    # One of the edges in the cycle should be removed
    assert not dagified_graph.has_edge("C", "A") or not dagified_graph.has_edge("A", "B") or not dagified_graph.has_edge("B", "C")


def test_dagify_graph_no_cycles():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C")])  # No cycles
    dagified_graph = dagify_graph(graph)
    assert isinstance(dagified_graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(dagified_graph)
    assert dagified_graph.has_edge("A", "B")
    assert dagified_graph.has_edge("B", "C")
    assert dagified_graph.number_of_edges() == 2  # No edges should be removed