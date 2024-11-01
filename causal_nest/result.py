from dataclasses import dataclass
from typing import Any, Dict, List

from causal_nest.problem import Problem
from causal_nest.results import DiscoveryResult, EstimationResult, RefutationResult

colors = {
    "gray": "#A9A9A9",
    "green": "#4CAF50",
    "amber": "#FFC107",
    "red": "#F44336",
    "blue": "#2196F3",
    "cyan": "#7EC1D7",
    "light_blue": "#ADD8E6",
    "orange": "#FF9800",
    "purple": "#9C27B0",
}


def get_node_style(estimate_value: int, refutations: List[EstimationResult]):
    """
    Determines the style of a node based on the estimate value and refutation results.

    Args:
        estimate_value (int): The estimated value of the node.
        refutations (List[EstimationResult]): A list of refutation results.

    Returns:
        Dict[str, str]: A dictionary containing the style attributes for the node.
    """
    passed_amount = len(list(filter(lambda x: x.passed == True, refutations)))
    passed_rate = 0
    if len(refutations) > 0:
        passed_rate = passed_amount / len(refutations)

    if estimate_value < 0:
        return {"fillcolor": "lightpink"}
    elif estimate_value > 0:
        if len(refutations) == 0:
            return {"fillcolor": f'{colors["light_blue"]};0.5:{colors["cyan"]}', "fontcolor": "black"}

        if passed_rate < 0.33:
            return {"fillcolor": colors["red"], "fontcolor": "white"}

        if passed_amount == 3:
            return {"fillcolor": colors["green"], "fontcolor": "white"}

        return {"fillcolor": colors["amber"], "fontcolor": "black"}

    else:
        return {
            "fillcolor": "gray;0.25:lightgray;0.25:gray;0.25:lightgray;0.25",
            "color": "lightgray",
            "style": "wedged",
            "shape": "doublecircle",
        }


def generate_result_graph(
    dr: DiscoveryResult,
    problem: Problem,
    layout_option: str = None,
):
    """
    Generates a result graph based on the discovery result and problem instance.

    Args:
        dr (DiscoveryResult): The discovery result containing the causal graph.
        problem (Problem): The problem instance containing the dataset.
        layout_option (str, optional): The layout option for the graph visualization. Defaults to None.

    Returns:
        Any: The generated graph object.
    """
    features = [n for n in dr.output_graph.nodes()]
    estimates = {f: None for f in features}
    refutations = {f: [] for f in features}

    for f in features:
        estimates[f] = next((r for r in problem.estimation_results[dr.model] if r.treatment == f), None)

    for key in refutations.keys():
        refutations[key] = list(filter(lambda x: x.treatment == key, problem.refutation_results[dr.model]))

    dot = [
        "digraph G {",
        'fontname="Helvetica,Arial,sans-serif";',
        'node[style="filled", fontsize=20, penwidth=2.5, fixedsize=true, fontcolor="black", fillcolor="gray", color="black", shape="circle"];',
        "edge[penwidth=2, minlen=2];",
    ]

    if layout_option == "fdp":
        dot.append('layout="fdp";')
        dot.append('splines="compound";')
    elif layout_option == "circo":
        dot.append('layout="circo"')
        dot.append('splines="polyline";')
    else:
        dot.append('splines="polyline";')

    for n in dr.output_graph.nodes():
        size = 1
        style = {"tooltip": n}
        label = n

        if estimates[n] is not None:
            e = estimates[n]
            r = refutations[n]

            # size = min(5, max(1, 1 + abs(e.estimate.value)))
            style = get_node_style(e.estimate.value, r)
            label = f"{n}\n{e.estimate.value:.2f}"

        if n == problem.dataset.target:
            style = {"fillcolor": "magenta", "color": "purple", "shape": "hexagon", "fontcolor": "white"}

        style_str = ", ".join(f'{k}="{v}"' for k, v in style.items())

        dot.append(f'   "{n}"[width={size}, height={size}, label="{label}", {style_str}];')

    for u, v in dr.output_graph.edges():
        color = "black"
        if problem.knowledge is not None and (u, v) in problem.knowledge.forbidden_edges:
            color = "red"

        dot.append(f'   "{u}" -> "{v}"[color={color}];')

    if problem.knowledge is not None:
        for u, v in problem.knowledge.required_edges:
            if not dr.output_graph.has_edge(u, v):
                dot.append(f'   "{u}" -> "{v}"[color="lightgray", style="dashed"];')

    dot.append("}")

    return "\n".join(dot)


def generate_all_results(problem: Problem, layout_option=None):
    """
    Generates a result graph based on the discovery result and problem instance.

    Args:
        dr (DiscoveryResult): The discovery result containing the causal graph.
        problem (Problem): The problem instance containing the dataset.
        layout_option (str, optional): The layout option for the graph visualization. Defaults to None.

    Returns:
        Any: The generated graph object.
    """
    return {
        model: (generate_result_graph(dr, problem, layout_option) if dr is not None else None)
        for model, dr in problem.discovery_results.items()
    }
