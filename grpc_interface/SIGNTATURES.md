

```python
# Datasets
def handle_missing_data(dataset: Dataset, method: MissingDataHandlingMethod = MissingDataHandlingMethod.DROP):
return dataset: Dataset

def applyable_models(problem: Problem):
return List[str]

# Discovery
def discover_with_model(
    problem: Problem, model: DiscoveryMethodModel, verbose: bool = False, orient_toward_target: bool = True
):
return DiscoveryResult

def discover_with_all_models(
    problem: Problem,
    max_seconds_model: int = 90,
    verbose: bool = False,
    max_workers: int = None,
    orient_toward_target: bool = True,
):
return Problem

# Estimation
def estimate_model_effects(problem: Problem, dr: DiscoveryResult, timeout: int = 180):
  return EstimationResult

def estimate_all_effects(
    problem: Problem,
    max_seconds_model: int = 360,
    verbose: bool = False,
    max_workers=None,
):
return Problem

# Refutation
def refute_with_model(problem: Problem, estimation_result: EstimationResult, model: RefutationMethodModel):
return RefutationResult

def refute_all_results(
    problem: Problem,
    max_seconds_global: int = 180,
    max_seconds_model: int = 25,
    verbose: bool = False,
    max_workers=None,
):
return Problem


# Knowledge
# Maybe its better to handle the knowledge file in the types library: i would need to send files through the grpc connection, and it would increase the complexity of the grpc service.
def parse_knowledge_file(file_path: str) -> Knowledge:

# Result
def generate_result_graph(
    dr: DiscoveryResult,
    problem: Problem,
    layout_option: str = None,
):

return str
def generate_all_results(problem: Problem, layout_option=None):
return str
```
