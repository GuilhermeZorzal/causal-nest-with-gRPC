import time
from concurrent.futures import TimeoutError
from dataclasses import replace
from multiprocessing import cpu_count

from dowhy import CausalModel
from pebble import ProcessPool

from causal_nest.problem import Problem
from causal_nest.results import DiscoveryResult, EstimationResult
from causal_nest.utils import graph_to_pydot_string


def estimate_model_effects(problem: Problem, dr: DiscoveryResult, timeout: int = 180):
    """
    Estimates the causal effects for all features in the dataset using the discovered model.

    Args:
        problem (Problem): The problem instance containing the dataset.
        dr (DiscoveryResult): The discovery result containing the causal graph.
        timeout (int, optional): The maximum time allowed for the estimation process. Defaults to 180 seconds.

    Returns:
        dict: A dictionary containing the model name and the estimation results for each feature.
    """

    response = {"model": dr.model, "results": []}
    start_time = time.time()

    for f in problem.dataset.feature_mapping:
        if time.time() - start_time > timeout:
            # If timeout thershold is reached, then return the results up to that point
            return response

        r = estimate_effect(problem, dr, f.feature)
        response["results"].append(r)

    return response


def estimate_effect(problem: Problem, dr: DiscoveryResult, treatment: str) -> EstimationResult:
    """
    Estimates the causal effect of a treatment on the outcome using the discovered model.

    Args:
        problem (Problem): The problem instance containing the dataset.
        dr (DiscoveryResult): The discovery result containing the causal graph.
        treatment (str): The treatment variable for which to estimate the causal effect.

    Returns:
        EstimationResult: The result of the estimation process, including the estimand and p-value.

    Raises:
        ValueError: If the treatment variable does not exist in the dataset.
    """

    if treatment not in problem.dataset.data.columns:
        raise ValueError("Argument 'treatment' must exist in the dataframe")

    model = CausalModel(
        data=problem.dataset.data,
        treatment=treatment,
        outcome=problem.dataset.target,
        graph=graph_to_pydot_string(dr.output_graph),
    )

    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

    p_value = None
    try:
        p_value = estimate.test_stat_significance()["p_value"]
    except Exception:
        pass

    return EstimationResult(
        estimand=estimand,
        model=dr.model,
        treatment=treatment,
        estimate=estimate,
        control_value=estimate.control_value,
        treatment_value=estimate.treatment_value,
        p_value=p_value,
    )


def estimate_all_effects(
    problem: Problem,
    max_seconds_model: int = 360,
    verbose: bool = False,
    max_workers=None,
):
    """
    Estimates the causal effects for all features in the dataset using all discovered models.

    Args:
        problem (Problem): The problem instance containing the dataset and discovery results.
        max_seconds_model (int, optional): The maximum time allowed for each model's estimation process. Defaults to 360 seconds.
        verbose (bool, optional): If True, prints warnings and errors. Defaults to False.
        max_workers (int, optional): The maximum number of workers to use. Defaults to the number of CPU cores.

    Returns:
        Problem: The problem instance with the estimation results added.
    """

    sorted_results = list(
        sorted(filter(lambda x: x, problem.discovery_results.values()), key=lambda x: x.priority_score, reverse=True)
    )
    estimation_results = {sorted_results[i].model: None for i in range(len(sorted_results))}

    if max_workers is None:
        max_workers = cpu_count()

    with ProcessPool(max_workers=max_workers) as pool:
        futures = []

        for dr in sorted_results:
            futures.append(pool.schedule(estimate_model_effects, args=(problem, dr, max_seconds_model)))

        for future in futures:
            try:
                future_result = future.result()
                if future_result is not None:
                    estimation_results[future_result["model"]] = future_result["results"]
            except TimeoutError:
                pass
            except Exception as e:
                pass

    return replace(problem, estimation_results=estimation_results)
