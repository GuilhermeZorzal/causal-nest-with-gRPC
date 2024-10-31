from cdt.causality.graph import GES as CDT_GES

from causal_nest.dataset import Dataset, FeatureType, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Greedy Equivalance Search algorithm
class GES(DiscoveryMethodModel):
    def __init__(self):
        super().__init__(
            allowed_feature_types=[FeatureType.CONTINUOUS, FeatureType.CATEGORICAL],
            gaussian_assumption=True,
            linearity_assumption=False,
        )

    def create_graph_from_data(self, dataset: Dataset):
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        m = CDT_GES()
        graph = m.predict(featured_only_data(dataset))

        return graph
