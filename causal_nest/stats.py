import networkx as nx
from cdt.metrics import SHD, SID, precision_recall
from networkx import DiGraph

from causal_nest.knowledge import Knowledge


def calculate_auc_pr(graph_1: DiGraph, graph_2: DiGraph) -> float:
    auc_pr, _curve = precision_recall(graph_1, graph_2)
    return auc_pr


def calculate_shd(graph_1: DiGraph, graph_2: DiGraph) -> float:
    return SHD(graph_1, graph_2, double_for_anticausal=False)


def calculate_sid(graph_1: DiGraph, graph_2: DiGraph) -> float:
    return SID(graph_1, graph_2)


def calculate_graph_ranking_score(graph: DiGraph, target: str) -> float:
    # Score 0 para grafos sem o alvo
    if target not in graph.nodes():
        return 0

    # Verifica se há arestas chegando ao vértice alvo
    num_in_edges = graph.in_degree(target)

    # Calcula a distância média de cada nó para o vértice alvo
    num_nodes = len(graph.nodes())
    distances = []
    for node in graph.nodes():
        try:
            distance = nx.shortest_path_length(graph, source=node, target=target)
            distances.append(distance)
        except nx.NetworkXNoPath:
            pass
    avg_distance = sum(distances) / len(distances) if distances else 0

    # Calcula a densidade do grafo
    density = nx.density(graph)
    # Verifica se o grafo é conexo
    is_connected = nx.is_weakly_connected(graph)
    # Calcula a centralidade de intermediação (betweenness centrality)
    betweenness_centrality = nx.betweenness_centrality(graph)

    # Aumenta o score proporcionalmente ao número de arestas chegando ao vértice alvo
    edge_score = num_in_edges / num_nodes
    # Aumenta o score inversamente proporcional à distância média dos nós até o vértice alvo, para penalizar grafos com nós distantes do alvo
    distance_score = 1 / (avg_distance + 1)
    # Aumenta o score proporcionalmente à densidade do grafo, indicando uma maior interconectividade
    density_score = density
    # Aumenta o score se o grafo for conexo, indicando que todas as variáveis estão conectadas de alguma forma
    connectivity_score = 1 if is_connected else 0
    # Aumenta o score proporcionalmente à soma da centralidade de intermediação de todos os nós, indicando a importância dos nós na rede
    betweenness_score = sum(betweenness_centrality.values())

    # Score final baseado nas métricas
    score = 100 * (edge_score * distance_score * density_score * connectivity_score * betweenness_score)

    return score


def forbidden_edges_violation_rate(graph: nx.DiGraph, knowledge: Knowledge) -> float:
    if len(knowledge.forbidden_edges) == 0:
        return 0.0

    violated_edges = sum(1 for forbidden_edge in knowledge.forbidden_edges if graph.has_edge(*forbidden_edge))
    return violated_edges / len(knowledge.forbidden_edges)


def required_edges_compliance_rate(graph: nx.DiGraph, knowledge: Knowledge) -> float:
    if len(knowledge.required_edges) == 0:
        return 1.0

    required_edges = sum(1 for required_edge in knowledge.required_edges if graph.has_edge(*required_edge))
    return required_edges / len(knowledge.required_edges)


def graph_integrity_score(violation_rate: float, compliance_rate: float) -> float:
    return (1 - violation_rate) * compliance_rate
