"""
Map scoring configs and answers to metrics; e.g. accuracy, f1, etc.
"""
import Levenshtein
import typing as t

def map_answer_type_to_metrics(answer_type: t.Type[t.Any]) -> str:
    results = {"exact_match": lambda x, y: x == y}
    if answer_type in [str]:
        results.update({
            "levenshtein": lambda x, y: Levenshtein.distance(x, y),
        })
    if answer_type in [int, float]:
        results.update({
            "absolute_error": lambda x, y: abs(x - y),
            "relative_error": lambda x, y: abs(x - y) / x,
            "squared_error": lambda x, y: (x - y) ** 2,
        })
    return results


