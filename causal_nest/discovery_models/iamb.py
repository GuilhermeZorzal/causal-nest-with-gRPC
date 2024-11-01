from cdt.causality.graph.bnlearn import IAMB as CDT_IAMB

from causal_nest.dataset import Dataset, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Incremental Association Markov Blanket algorithm
class IAMB(DiscoveryMethodModel):
    """
    Incremental Association Markov Blanket (IAMB) algorithm for causal discovery.

    This class implements the IAMB algorithm, which is used to discover causal graphs from data.
    It does not assume Gaussian distribution or linearity of the data.

    Attributes:
        allowed_feature_types (list): List of allowed feature types for this method.
        gaussian_assumption (bool): Indicates if the method assumes Gaussian distribution.
        linearity_assumption (bool): Indicates if the method assumes linearity.
    """

    def __init__(self):
        super().__init__(gaussian_assumption=False, linearity_assumption=False)

    def create_graph_from_data(self, dataset: Dataset):
        """
        Creates a causal graph from the given dataset using the IAMB algorithm.

        Args:
            dataset (Dataset): The dataset from which to create the causal graph.

        Returns:
            nx.DiGraph: The discovered causal graph.

        Raises:
            ValueError: If the method is not allowed to be used with the given dataset.
        """
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        m = CDT_IAMB()
        graph = m.predict(featured_only_data(dataset))

        return graph
