# CausalNest
A nest of causal tools for building a full causal pipeline: from data treating, exploring to causal discovery and inference!

## Overview
CausalNest provides a comprehensive suite of tools for building a full causal pipeline, from data treatment and exploration to causal discovery and inference. The framework is organized into several modules:

- `dataset`: Tools for handling datasets, including missing data handling and feature type definitions.
- `discovery_models`: Implementations of various causal discovery algorithms.
- `distribution`: Functions for statistical distribution analysis.
- `estimation`: Methods for estimating causal effects.
- `knowledge`: Structures for encoding prior knowledge about causal relationships.
- `problem`: Defines the causal problem and integrates various components.
- `refutation_models`: Methods for refuting causal claims.
- `result`: Structures for storing and displaying results from causal discovery, estimation, and refutation.
- `stats`: Statistical functions for evaluating causal models.
- `utils`: Utility functions used across the framework.

## Commands
- Install: `poetry install` 
- Test: `poetry run poe test` 
- Format: `poetry run poe format` 

## Quick Example
Here is a quick example of how to use the CausalNest framework with Sachs dataset

```python
import networkx as nx 


# Import the dataset
from cdt.data import load_dataset
sachs, sachs_ground_truth = load_dataset('sachs')


# Define the dataset feature mapping, target, and the problem
from causal_nest.problem import MissingDataHandlingMethod, Problem, Dataset, handle_missing_data, FeatureTypeMap, FeatureType, estimate_feature_importances

feature_mapping = [FeatureTypeMap(feature=c, type=FeatureType.CONTINUOUS) for c in sachs.drop('plcg', axis=1).columns]

dataset = Dataset(data=sachs, target='plcg', feature_mapping=feature_mapping)
dataset = handle_missing_data(dataset, MissingDataHandlingMethod.FORWARD_FILL)
dataset = estimate_feature_importances(dataset)
problem = Problem(dataset=dataset, description='Sachs')

# Check out the feature sorting (a cool feature from CausalNest)
print(dataset.feature_mapping)


# Causal Discovery
## Fetch applyable models
from causal_nest.discovery import applyable_models

models = applyable_models(dataset)
print(models)

## Run discovery from these models
from causal_nest.discovery import discover_with_all_models

real_results = discover_with_all_models(dataset)
print(real_results)

## Sorting results by potential and feature importance
sorted_results = list(sorted(filter(lambda x: x, real_results.values()), key=lambda x: x.priority_score, reverse=True))
print(sorted_results)

# Estimating
from causal_nest.estimation import estimate_all_effects

est_results = estimate_all_effects(dataset, real_results, verbose=True)
print(est_results)

# Refuting
from causal_nest.refutation import refute_all_results

ref_results = refute_all_results(dataset, est_results, max_seconds_global=300, max_seconds_model=120)
print(ref_results)

# Visualizing
from causal_nest.result import generate_result_graph

graphs = {}
for r in real_results.values():
    if (r is None):
        continue

    graph = generate_result_graph(r, est_results, ref_results, dataset.target)
    graphs[r.model] = graph


## Using Graphviz to render a custom DOT
import pydot
from graphviz import Source

print('FAST_IAMB')
graphviz_source = Source(graphs['FAST_IAMB'])
graphviz_source.render(format='png')

print('PC')
graphviz_source = Source(graphs['PC'])
graphviz_source.render(format='png')
```

For more detailed examples and usage, please refer to the notebooks directory.