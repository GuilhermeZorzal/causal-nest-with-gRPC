from dataclasses import dataclass
from typing import Optional

import networkx as nx


@dataclass
class DiscoveryResult:
    """
    Class to store and display the results of a causal discovery process.

    Attributes:
        output_graph (nx.DiGraph): The directed graph representing the discovered causal structure.
        model (str): The name of the model used for discovery.
        auc_pr (Optional[float]): The Area Under the Precision-Recall Curve (AUC-PR) score.
        shd (Optional[int]): The Structural Hamming Distance (SHD) score.
        sid (Optional[int]): The Structural Intervention Distance (SID) score.
        runtime (Optional[float]): The runtime of the discovery process in seconds.
        priority_score (float): The priority score of the discovery result.
        knowledge_integrity_score (Optional[float]): The knowledge integrity score.
        forbidden_edges_violation_rate (Optional[float]): The rate of forbidden edges violations.
        required_edges_compliance_rate (Optional[float]): The rate of required edges compliance.
    """

    output_graph: nx.DiGraph = None
    model: str = None
    auc_pr: Optional[float] = None
    shd: Optional[int] = None
    sid: Optional[int] = None
    runtime: Optional[float] = None
    priority_score: float = 0
    knowledge_integrity_score: Optional[float] = None
    forbidden_edges_violation_rate: Optional[float] = None
    required_edges_compliance_rate: Optional[float] = None

    def print(self):
        """
        Prints the discovery result statistics in a formatted manner.

        This method prints various statistics related to the discovery result, including runtime,
        AUC-PR, SHD, SID, priority score, knowledge integrity score, forbidden edges violation rate,
        and required edges compliance rate.

        Returns:
            str: An empty string.
        """
        print("\n~Stats for {}~\n".format(self.model))
        print("\t\tRuntime: {:3f} seconds".format(self.runtime))
        print("\t\tAUC_PR: {}".format(self.auc_pr))
        print("\t\tSHD: {}".format(self.shd))
        print("\t\tSID: {}".format(self.sid))
        print("\t\tPriority Score: {}".format(self.priority_score))
        print("\t\tIntegrity Score: {}".format(self.knowledge_integrity_score))
        print("\t\tForbidden Edges Violation Rate: {}".format(self.forbidden_edges_violation_rate))
        print("\t\tRequired Edges Compliance Rate: {}".format(self.required_edges_compliance_rate))
        print("\n")

        return ""
