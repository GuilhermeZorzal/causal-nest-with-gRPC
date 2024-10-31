from cdt.causality.graph import PC as CDT_PC

from causal_nest.dataset import Dataset, FeatureType, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Peter-Clark algorithm
class PC(DiscoveryMethodModel):
    def __init__(self):
        super().__init__(
            allowed_feature_types=[FeatureType.CONTINUOUS, FeatureType.DISCRETE],
            gaussian_assumption=False,
            linearity_assumption=False,
        )

    def create_graph_from_data(self, dataset: Dataset):
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        m = CDT_PC()
        graph = m.predict(featured_only_data(dataset))

        return graph
