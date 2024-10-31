from dataclasses import dataclass
from typing import Optional

import networkx as nx


@dataclass
class DiscoveryResult:
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
