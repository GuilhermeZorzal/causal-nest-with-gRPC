from cdt.causality.graph import LiNGAM as CDT_LINGAM

from causal_nest.dataset import Dataset, FeatureType, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Linear Non-Gaussian Acyclic Model algorithm
class LINGAM(DiscoveryMethodModel):
    def __init__(self):
        super().__init__(
            allowed_feature_types=[FeatureType.CONTINUOUS], gaussian_assumption=True, linearity_assumption=True
        )

    def create_graph_from_data(self, dataset: Dataset):
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        m = CDT_LINGAM()
        graph = m.predict(featured_only_data(dataset))

        return graph
