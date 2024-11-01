import pytest
from causal_nest.knowledge import Knowledge

def test_knowledge_initialization():
    knowledge = Knowledge(
        required_edges=[("A", "B"), ("B", "C")],
        forbidden_edges=[("C", "A"), ("D", "E")]
    )
    assert knowledge.required_edges == [("A", "B"), ("B", "C")]
    assert knowledge.forbidden_edges == [("C", "A"), ("D", "E")]


def test_knowledge_add_required_edge_manually():
    knowledge = Knowledge()
    knowledge.required_edges.append(("A", "B"))
    assert knowledge.required_edges == [("A", "B")]


def test_knowledge_add_forbidden_edge_manually():
    knowledge = Knowledge()
    knowledge.forbidden_edges.append(("C", "D"))
    assert knowledge.forbidden_edges == [("C", "D")]


def test_knowledge_remove_required_edge_manually():
    knowledge = Knowledge(required_edges=[("A", "B")])
    knowledge.required_edges.remove(("A", "B"))
    assert knowledge.required_edges == []


def test_knowledge_remove_forbidden_edge_manually():
    knowledge = Knowledge(forbidden_edges=[("C", "D")])
    knowledge.forbidden_edges.remove(("C", "D"))
    assert knowledge.forbidden_edges == []


def test_knowledge_clear_required_edges():
    knowledge = Knowledge(required_edges=[("A", "B"), ("B", "C")])
    knowledge.required_edges.clear()
    assert knowledge.required_edges == []


def test_knowledge_clear_forbidden_edges():
    knowledge = Knowledge(forbidden_edges=[("C", "A"), ("D", "E")])
    knowledge.forbidden_edges.clear()
    assert knowledge.forbidden_edges == []


def test_knowledge_contains_required_edge_manually():
    knowledge = Knowledge(required_edges=[("A", "B")])
    assert ("A", "B") in knowledge.required_edges
    assert ("B", "C") not in knowledge.required_edges


def test_knowledge_contains_forbidden_edge_manually():
    knowledge = Knowledge(forbidden_edges=[("C", "D")])
    assert ("C", "D") in knowledge.forbidden_edges
    assert ("A", "B") not in knowledge.forbidden_edges