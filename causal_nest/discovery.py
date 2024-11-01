import os
from concurrent.futures import TimeoutError
from dataclasses import replace
from multiprocessing import cpu_count
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import networkx as nx
from pebble import ProcessExpired, ProcessPool

from causal_nest.discovery_models import (
    BES,
    CAM,
    CCDR,
    CGNN,
    FAST_IAMB,
    GES,
    GIES,
    GRASP,
    GS,
    IAMB,
    INTER_IAMB,
    LINGAM,
    PC,
    SAM,
    DiscoveryMethodModel,
)
from causal_nest.problem import Problem
from causal_nest.results import DiscoveryResult
from causal_nest.stats import (
    calculate_auc_pr,
    calculate_graph_ranking_score,
    calculate_shd,
    calculate_sid,
    forbidden_edges_violation_rate,
    graph_integrity_score,
    required_edges_compliance_rate,
)
from causal_nest.utils import dagify_graph, dagify_graph_v2

known_methods = [
    PC,
    GS,
    GES,
    GIES,
    CAM,
    CCDR,
    IAMB,
    # FAST_IAMB,
    # INTER_IAMB,
    LINGAM,
    SAM,
    BES,
    GRASP,
    CGNN,
]

def applyable_models(problem: Problem):
    """
    Filters and returns a list of models that are applicable to the given problem.

    Args:
        problem (Problem): The problem instance containing the dataset.

    Returns:
        list: A list of models that are applicable to the given problem.
    """
    return list(filter(lambda m: m().is_method_allowed(problem.dataset), known_methods))


def discover_with_model(
    problem: Problem, model: DiscoveryMethodModel, verbose: bool = False, orient_toward_target: bool = True
):
    """
    Discovers a causal graph using the specified model.

    Args:
        problem (Problem): The problem instance containing the dataset.
        model (DiscoveryMethodModel): The discovery model to use.
        verbose (bool, optional): If True, prints and plots the discovered graph. Defaults to False.
        orient_toward_target (bool, optional): If True, orients the graph toward the target. Defaults to True.

    Returns:
        DiscoveryResult: The result of the discovery process, including the discovered graph and various statistics.
    """
    model_name = model.__name__

    start = timer()
    m = model()
    output_graph = m.create_graph_from_data(problem.dataset)
    end = timer()

    runtime = end - start

    output_graph = (
        dagify_graph_v2(output_graph, problem.dataset.target) if orient_toward_target else dagify_graph(output_graph)
    )

    priority_score = calculate_graph_ranking_score(output_graph, problem.dataset.target)
    stats = {"auc_pr": None, "shd": None, "sid": None, "kis": None, "fevr": None, "recr": None}

    if problem.ground_truth is not None:
        stats["auc_pr"] = calculate_auc_pr(problem.ground_truth, output_graph)
        stats["shd"] = calculate_shd(problem.ground_truth, output_graph)
        stats["sid"] = calculate_sid(problem.ground_truth, output_graph)

    if problem.knowledge is not None and len(problem.knowledge.forbidden_edges) > 0:
        stats["fevr"] = forbidden_edges_violation_rate(output_graph, problem.knowledge)
        stats["recr"] = required_edges_compliance_rate(output_graph, problem.knowledge)
        stats["kis"] = graph_integrity_score(stats["fevr"], stats["recr"])

    dr = DiscoveryResult(
        model=model_name,
        output_graph=output_graph,
        runtime=runtime,
        priority_score=priority_score,
        auc_pr=stats["auc_pr"],
        shd=stats["shd"],
        sid=stats["sid"],
        knowledge_integrity_score=stats["kis"],
        forbidden_edges_violation_rate=stats["fevr"],
        required_edges_compliance_rate=stats["recr"],
    )

    if verbose:
        dr.print()
        nx.draw(output_graph, with_labels=True, node_size=500, font_size=8, node_color="yellow")
        plt.title(f"Discovered graph for {model_name}")
        plt.show()

    return dr


def _run_discover_with_model_task(args):
    """
    Helper function to run the discovery process with a model.

    Args:
        args (tuple): A tuple containing the arguments for the discover_with_model function.

    Returns:
        DiscoveryResult: The result of the discovery process.
    """
    return discover_with_model(*args)


def discover_with_all_models(
    problem: Problem,
    max_seconds_model: int = 90,
    verbose: bool = False,
    max_workers: int = None,
    orient_toward_target: bool = True,
):
    """
    Discovers causal graphs using all applicable models.

    Args:
        problem (Problem): The problem instance containing the dataset.
        max_seconds_model (int, optional): The maximum time allowed for each model. Defaults to 90.
        verbose (bool, optional): If True, prints warnings and errors. Defaults to False.
        max_workers (int, optional): The maximum number of workers to use. Defaults to the number of CPU cores.
        orient_toward_target (bool, optional): If True, orients the graph toward the target. Defaults to True.

    Returns:
        Problem: The problem instance with the discovery results added.
    """
    if max_workers is None:
        max_workers = cpu_count()
    
    models = applyable_models(problem)
    
    discovery_results = {models[i].__name__: None for i in range(len(models))}
    pool_args = [(problem, model, verbose, orient_toward_target) for model in models]
    
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(_run_discover_with_model_task, pool_args, timeout=max_seconds_model)
        
        iterator = future.result()
        
        while True:
            try:
                result = next(iterator)
                discovery_results[result.model] = result
            except StopIteration:
                break
            except TimeoutError as _error:
                if verbose:
                    print(f"Warning: discovery method took longer than {max_seconds_model} seconds")
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
                # raise error
            except Exception as error:
                print("Function raised %s" % error)
                print(error.traceback)
                # raise error
                
    return replace(problem, discovery_results=discovery_results)
