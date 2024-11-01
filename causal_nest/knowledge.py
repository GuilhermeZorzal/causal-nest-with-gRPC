from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Knowledge:
    """
    A class to represent the knowledge of required and forbidden edges in a causal graph.

    Attributes:
        required_edges (List[Tuple[str, str]]): A list of tuples representing the required edges.
        forbidden_edges (List[Tuple[str, str]]): A list of tuples representing the forbidden edges.
    """
    required_edges: List[Tuple[str, str]] = field(default_factory=list)
    forbidden_edges: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self._validate_edges(self.required_edges, "required_edges")
        self._validate_edges(self.forbidden_edges, "forbidden_edges")

    def _validate_edges(self, edges: List[Tuple[str, str]], attribute_name: str):
        """
        Validates that the edges are tuples of two strings.

        Args:
            edges (List[Tuple[str, str]]): The list of edges to validate.
            attribute_name (str): The name of the attribute being validated.

        Raises:
            ValueError: If any edge is not a tuple of two strings.
        """
        if not all(
            isinstance(edge, tuple) and len(edge) == 2 and all(isinstance(node, str) for node in edge) for edge in edges
        ):
            raise ValueError(f"All elements of {attribute_name} must be tuples of two strings")


def parse_knowledge_file(file_path: str) -> Knowledge:
    """
    Parses a knowledge file to extract required and forbidden edges.

    Args:
        file_path (str): The path to the knowledge file.

    Returns:
        Knowledge: An instance of the Knowledge class containing the parsed edges.

    Raises:
        ValueError: If the file format is incorrect.
    """
    required_edges = set()
    forbidden_edges = set()
    temporal_tiers = {}

    with open(file_path, "r") as file:
        lines = file.readlines()

    section = None

    for line in lines:
        line = line.strip()

        if line.startswith("addtemporal"):
            section = "temporal"
            continue

        if line.startswith("forbiddirect"):
            section = "forbidden"
            continue

        if line.startswith("requiredirect"):
            section = "required"
            continue

        if section == "temporal":
            if line:
                parts = line.split()
                tier = parts[0]
                nodes = parts[1:]
                if tier.endswith("*"):
                    tier = tier.rstrip("*")
                    for i in range(len(nodes)):
                        for j in range(i + 1, len(nodes)):
                            forbidden_edges.add((nodes[i], nodes[j]))
                            forbidden_edges.add((nodes[j], nodes[i]))
                temporal_tiers[int(tier)] = nodes

        if section == "forbidden":
            if line:
                nodes = line.split()
                if len(nodes) == 2:
                    forbidden_edges.add((nodes[0], nodes[1]))

        if section == "required":
            if line:
                nodes = line.split()
                if len(nodes) == 2:
                    required_edges.add((nodes[0], nodes[1]))

    # Process temporal tiers to create forbidden edges
    tiers_sorted = sorted(temporal_tiers.keys(), reverse=True)
    for i in range(len(tiers_sorted)):
        for j in range(i + 1, len(tiers_sorted)):
            higher_tier_nodes = temporal_tiers[tiers_sorted[i]]
            lower_tier_nodes = temporal_tiers[tiers_sorted[j]]
            for higher_node in higher_tier_nodes:
                for lower_node in lower_tier_nodes:
                    forbidden_edges.add((higher_node, lower_node))

    return Knowledge(required_edges=list(required_edges), forbidden_edges=list(forbidden_edges))
