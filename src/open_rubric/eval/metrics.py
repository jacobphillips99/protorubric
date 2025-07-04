"""
Map scoring configs and answers to metrics; e.g. accuracy, f1, etc.
"""

import typing as t
from collections import defaultdict

import Levenshtein
import numpy as np


def map_answer_type_to_metrics(
    answer_type: t.Type[t.Any],
) -> dict[str, t.Callable[[t.Any, t.Any], float]]:
    """
    Returns a dictionary of metric names to metric functions.
    Metric functions are callable with (answer, reference) as arguments and return a float.
    """
    results = {"exact_match": lambda x, y: x == y}
    if answer_type in [str]:
        # probably bad choice for options / categorical
        results.update(
            {
                "levenshtein": lambda x, y: Levenshtein.distance(x, y),
            }
        )
    if answer_type in [int, float]:
        results.update(
            {
                "absolute_error": lambda x, y: abs(x - y),
                "relative_error": lambda x, y: abs(x - y) / x,
                "squared_error": lambda x, y: (x - y) ** 2,
            }
        )
    return results


def results_to_metrics(
    req_name_to_score: dict[str, t.Any], answers: dict[str, t.Any]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    Maps from {Requirement name: score} to helpful metric shapes
    """
    # {Requirement name: {metric name: metric function}}
    metrics_functions_per_req: dict[str, dict[str, t.Callable[[t.Any, t.Any], float]]] = {
        k: map_answer_type_to_metrics(type(v)) for k, v in req_name_to_score.items()
    }

    # {Requirement name: {metric name: score}}
    req_to_metric_to_score: dict[str, dict[str, float]] = defaultdict(dict)
    for req_name, metric_functions in metrics_functions_per_req.items():
        for metric_name, metric_func in metric_functions.items():
            req_to_metric_to_score[req_name][metric_name] = metric_func(
                answers[req_name], req_name_to_score[req_name]
            )

    # {metric name: [score1, score2, ...]}
    metric_to_results: dict[str, list[float]] = defaultdict(list)
    for _, metric_to_score in req_to_metric_to_score.items():
        for metric_name, score in metric_to_score.items():
            metric_to_results[metric_name].append(score)

    # {metric name: {mean: mean, std: std, n: n}}
    metric_to_mean_score: dict[str, dict[str, float | int]] = {
        k: dict(mean=np.mean(v), std=np.std(v), n=len(v)) for k, v in metric_to_results.items()
    }
    return req_to_metric_to_score, metric_to_mean_score
