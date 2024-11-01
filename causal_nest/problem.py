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

    A problem is the starting point to causality. It contains all the required data and configurations
    for a causal pipeline.

    Attributes:
        dataset (Dataset): The dataset setup for this problem. Multiple problems may be used with the same dataset.
        description (str): A quick description of this problem, useful for distinguishing between problems in benchmarks.
        ground_truth (Optional[nx.DiGraph]): A graph containing the ground truth for the problem.
        knowledge (Knowledge): Set of required and forbidden edges, which will be evaluated later to prioritize edges while dagifying and shown.
        discovery_results (Optional[Dict[str, DiscoveryResult]]): Map of discovery results. The key is the discovery method name and the value is the result.
        estimation_results (Optional[Dict[str, List[EstimationResult]]]): Map of estimation results. The key is the discovery method name and the value is the list of feature estimations.
        refutation_results (Optional[Dict[str, List[RefutationResult]]]): Map of refutation results. The key is the discovery method name and the value is the list of feature refutations.
    """

    dataset: Dataset
    description: str = ""
    ground_truth: Optional[nx.DiGraph] = None
    knowledge: Knowledge = field(default_factory=Knowledge)
    discovery_results: Optional[Dict[str, DiscoveryResult]] = None
    estimation_results: Optional[Dict[str, List[EstimationResult]]] = None
    refutation_results: Optional[Dict[str, List[RefutationResult]]] = None

    def __post_init__(self):
        """
        Post-initialization processing to validate the fields.

        Raises:
            ValueError: If 'dataset' is not an instance of `Dataset` or 'knowledge' is not an instance of `Knowledge`.
        """
        if not isinstance(self.dataset, Dataset):
            raise ValueError("Field 'dataset' must be a CausalNest `Dataset` instance")
        if not isinstance(self.knowledge, Knowledge):
            raise ValueError("Field 'knowledge' must be a CausalNest `Knowledge` instance")
        # TODO: Validate `discovery_results`, `estimation_results` and `refutation_results` instances
