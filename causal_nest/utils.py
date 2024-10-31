from copy import deepcopy
from typing import List, Optional, Tuple

import networkx as nx
from networkx.drawing.nx_pydot import to_pydot


def graph_to_pydot_string(graph: nx.DiGraph):
    return to_pydot(graph).to_string()


def dagify_graph(g: nx.DiGraph) -> nx.DiGraph:
    """
    Input a graph and output a DAG.

    The heuristic is to reverse the edge with the lowest score of the cycle
    if possible, else remove it.

    Args:
        g (networkx.DiGraph): Graph to modify to output a DAG

    Returns:
        networkx.DiGraph: DAG made out of the input graph.
    """
    ncycles = len(list(nx.simple_cycles(g)))
    while not nx.is_directed_acyclic_graph(g):
        cycle = next(nx.simple_cycles(g))
        if len(cycle) == 2:
            source_node = cycle[0]
            target_node = cycle[1]
            if g.has_edge(source_node, target_node):
                g.remove_edge(source_node, target_node)
        else:
            edges = [(cycle[-1], cycle[0])]
            if g.has_edge(cycle[-1], cycle[0]):
                scores = [g[cycle[-1]][cycle[0]].get("weight", 1)]
            else:
                scores = [1]  # Default weight for unweighted edges
            for i, j in zip(cycle[:-1], cycle[1:]):
                edges.append((i, j))
                if g.has_edge(i, j):
                    scores.append(g[i][j].get("weight", 1))
                else:
                    scores.append(1)  # Default weight for unweighted edges

            i, j = edges[scores.index(min(scores))]
            gc = deepcopy(g)
            gc.remove_edge(i, j)
            gc.add_edge(j, i)
            ngc = len(list(nx.simple_cycles(gc)))
            if ngc < ncycles:
                if g.has_edge(j, i):
                    g.add_edge(j, i, weight=min(scores))
                else:
                    g.add_edge(j, i)
            g.remove_edge(i, j)
            ncycles = ngc
    return g


def dagify_graph_v2(g: nx.DiGraph, target_node) -> nx.DiGraph:
    """
    Input a graph and output a DAG while handling cycles involving a target node.

    The function modifies the graph to output a DAG. If a cycle contains an edge
    X -> Y where Y is the target_node, it preserves this edge. If edges are not
    oriented and one of the nodes in it is the target_node, it orients it as X -> target_node.

    Args:
        g (networkx.DiGraph): Graph to modify to output a DAG.
        target_node: The node that should be treated specially in cycle handling.

    Returns:
        networkx.DiGraph: DAG made out of the input graph.
    """
    ncycles = len(list(nx.simple_cycles(g)))
    while not nx.is_directed_acyclic_graph(g):
        cycle = next(nx.simple_cycles(g))
        if len(cycle) == 2:
            source_node = cycle[0]
            target_node_in_cycle = cycle[1]
            if target_node_in_cycle == target_node:
                continue  # Preserve the edge X -> target_node
            if g.has_edge(source_node, target_node_in_cycle):
                g.remove_edge(source_node, target_node_in_cycle)
        else:
            edges = [(cycle[-1], cycle[0])]
            if g.has_edge(cycle[-1], cycle[0]):
                scores = [g[cycle[-1]][cycle[0]].get("weight", 1)]
            else:
                scores = [1]  # Default weight for unweighted edges
            for i, j in zip(cycle[:-1], cycle[1:]):
                edges.append((i, j))
                if g.has_edge(i, j):
                    scores.append(g[i][j].get("weight", 1))
                else:
                    scores.append(1)  # Default weight for unweighted edges

            # Check if any edge in the cycle points towards target_node
            edge_to_preserve = None
            for edge in edges:
                if edge[1] == target_node:
                    edge_to_preserve = edge
                    break

            if edge_to_preserve:
                # Preserve the edge X -> target_node
                i, j = edge_to_preserve
                if not g.has_edge(i, j):
                    g.add_edge(i, j)
                continue

            # Otherwise, find the edge with the minimum score to reverse
            i, j = edges[scores.index(min(scores))]
            gc = deepcopy(g)
            gc.remove_edge(i, j)
            gc.add_edge(j, i)
            ngc = len(list(nx.simple_cycles(gc)))
            if ngc < ncycles:
                if g.has_edge(j, i):
                    g.add_edge(j, i, weight=min(scores))
                else:
                    g.add_edge(j, i)
            g.remove_edge(i, j)
            ncycles = ngc

    return g
