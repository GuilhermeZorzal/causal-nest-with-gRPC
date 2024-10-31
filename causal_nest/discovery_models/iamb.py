from cdt.causality.graph.bnlearn import IAMB as CDT_IAMB

from causal_nest.dataset import Dataset, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Incremental Association Markov Blanket algorithm
class IAMB(DiscoveryMethodModel):
    def __init__(self):
        super().__init__(gaussian_assumption=False, linearity_assumption=False)

    def create_graph_from_data(self, dataset: Dataset):
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        m = CDT_IAMB()
        graph = m.predict(featured_only_data(dataset))

        return graph
