from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from causal_nest.dataset import Dataset
from causal_nest.knowledge import Knowledge
from causal_nest.results import DiscoveryResult, EstimationResult, RefutationResult


@dataclass
class Problem:
    """
    Data structure to represent a problem.

    A problem is the starting point to causality. It contains all the required data and configurations to a causal pipeline.
    """

    dataset: Dataset
    """The dataset setup for this problem. Multiple problems may be used with the same dataset."""

    description: str = ""
    """A quick description to this problem as it may be overwhelming to determine which exact problem is which in benchmarks"""

    ground_truth: Optional[nx.DiGraph] = None
    """A graph containing the ground truth to a problem"""

    knowledge: Knowledge = field(default_factory=Knowledge)
    """Set of required and forbidden edges, which will be evluated later on to prioritize edges while dagifying and shown"""

    discovery_results: Optional[Dict[str, DiscoveryResult]] = None
    """Map of discovery results. The key is the discovery method name and the value is the result"""

    estimation_results: Optional[Dict[str, List[EstimationResult]]] = None
    """Map of estimation results. The key is the discovery method name and the value is the list of features estimations"""

    refutation_results: Optional[Dict[str, List[RefutationResult]]] = None
    """Map of estimation results. The key is the discovery method name and the value is the list of features refutations"""

    def __post_init__(self):
        if not isinstance(self.dataset, Dataset):
            raise ValueError("Field 'dataset' must be a CausalNest `Dataset` instance")
        if not isinstance(self.knowledge, Knowledge):
            raise ValueError("Field 'knowledge' must be a CausalNest `Knowledge` instance")
        # TODO: Validate `discovery_results`, `estimation_results` and `refutation_results` instances
