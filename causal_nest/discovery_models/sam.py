from cdt.causality.graph import SAM as CDT_SAM

from causal_nest.dataset import Dataset, FeatureType, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Structural Agnostic Model
class SAM(DiscoveryMethodModel):
    def __init__(self):
        super().__init__(
            allowed_feature_types=[FeatureType.CONTINUOUS], gaussian_assumption=False, linearity_assumption=False
        )

    def create_graph_from_data(self, dataset: Dataset):
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        m = CDT_SAM(mixed_data=True, train_epochs=750, test_epochs=250, nruns=8)
        graph = m.predict(featured_only_data(dataset))

        return graph
