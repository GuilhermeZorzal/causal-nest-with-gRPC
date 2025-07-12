

```python
# Datasets
def handle_missing_data(dataset: Dataset, method: MissingDataHandlingMethod = MissingDataHandlingMethod.DROP):

def applyable_models(problem: Problem):

# Discovery
def discover_with_model(
    problem: Problem, model: DiscoveryMethodModel, verbose: bool = False, orient_toward_target: bool = True
):

def discover_with_all_models(
    problem: Problem,
    max_seconds_model: int = 90,
    verbose: bool = False,
    max_workers: int = None,
    orient_toward_target: bool = True,
):

# Estimation
def estimate_model_effects(problem: Problem, dr: DiscoveryResult, timeout: int = 180):

def estimate_all_effects(
    problem: Problem,
    max_seconds_model: int = 360,
    verbose: bool = False,
    max_workers=None,
):

# Refutation
def refute_with_model(problem: Problem, estimation_result: EstimationResult, model: RefutationMethodModel):

def refute_all_results(
    problem: Problem,
    max_seconds_global: int = 180,
    max_seconds_model: int = 25,
    verbose: bool = False,
    max_workers=None,
):


# Knowledge
# Maybe its better to handle the knowledge file in the types library: i would need to send files through the grpc connection, and it would increase the complexity of the grpc service.
def parse_knowledge_file(file_path: str) -> Knowledge:

# Result
def generate_result_graph(
    dr: DiscoveryResult,
    problem: Problem,
    layout_option: str = None,
):
def generate_all_results(problem: Problem, layout_option=None):
```
