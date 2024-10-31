from dataclasses import dataclass

from networkx import DiGraph

from causal_nest.stats import BenchmarkStats


@dataclass
class ModelBenchmark:
    stats: BenchmarkStats = BenchmarkStats()
    model_name: str
    out_graph: DiGraph
