import time
from concurrent.futures import TimeoutError
from dataclasses import replace
from multiprocessing import cpu_count
from typing import Dict

from dowhy import CausalModel
from pebble import ProcessPool

from causal_nest.problem import Problem
from causal_nest.results import DiscoveryResult, EstimationResult
from causal_nest.utils import graph_to_pydot_string


def estimate_model_effects(problem: Problem, dr: DiscoveryResult, timeout: int = 180):
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
