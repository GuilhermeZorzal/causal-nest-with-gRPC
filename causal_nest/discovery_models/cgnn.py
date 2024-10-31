from cdt.causality.graph import CGNN as CDT_CGNN

from causal_nest.dataset import Dataset, FeatureType, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Causal Generative Neural Networks algorithm
class CGNN(DiscoveryMethodModel):
    def __init__(self):
        super().__init__(
            allowed_feature_types=[FeatureType.CONTINUOUS], gaussian_assumption=False, linearity_assumption=False
        )

    def create_graph_from_data(self, dataset: Dataset):
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        m = CDT_CGNN(nruns=4, nh=5, train_epochs=150, test_epochs=50, verbose=False)
        graph = m.predict(featured_only_data(dataset))

        return graph
