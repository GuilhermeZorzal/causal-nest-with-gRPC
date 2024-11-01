import networkx as nx
from causallearn.search.PermutationBased.GRaSP import grasp

from causal_nest.dataset import Dataset, FeatureType, featured_only_data
from causal_nest.discovery_models.discovery_method_model import DiscoveryMethodModel


# Greedy Relaxation of the Sparsest Permutation algorithm
class GRASP(DiscoveryMethodModel):
    """
    Greedy Sparsest Permutation (GRaSP) algorithm for causal discovery.

    This class implements the GRaSP algorithm, which is used to discover causal graphs from data.
    It assumes both Gaussian distribution and linearity of the data.

    Attributes:
        allowed_feature_types (list): List of allowed feature types for this method.
        gaussian_assumption (bool): Indicates if the method assumes Gaussian distribution.
        linearity_assumption (bool): Indicates if the method assumes linearity.
    """

    def __init__(self):
        super().__init__(
            allowed_feature_types=[FeatureType.CONTINUOUS], gaussian_assumption=False, linearity_assumption=False
        )

    def create_graph_from_data(self, dataset: Dataset):
        """
        Creates a causal graph from the given dataset using the GRaSP algorithm.

        Args:
            dataset (Dataset): The dataset from which to create the causal graph.

        Returns:
            nx.DiGraph: The discovered causal graph.

        Raises:
            ValueError: If the method is not allowed to be used with the given dataset.
        """
        if not self.is_method_allowed(dataset):
            raise ValueError("This method can not be used with this dataset")

        fod = featured_only_data(dataset)
        mapping = mapping = {i: fod.columns[i] for i in range(len(fod.columns))}

        g = grasp(fod.to_numpy())
        graph = nx.from_numpy_array(g.graph, create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, mapping)

        return graph
