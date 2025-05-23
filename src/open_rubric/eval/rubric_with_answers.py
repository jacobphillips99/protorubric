from collections import defaultdict
import numpy as np
from open_rubric.configs.answers import AnswerConfig, BoolAnswerConfig, FloatAnswerConfig, IntAnswerConfig, StringAnswerConfig
from open_rubric.eval.metrics import map_answer_type_to_metrics
from open_rubric.rubric import Rubric
from open_rubric.configs.aggregating import AggregatedQueryConfig
import typing as t


def generate_test_answers(rubric: Rubric) -> dict[str, t.Any]:
    answers = {}
    for req in rubric.requirements.get_all_requirements():
        answer_type = type(req._result.score)
        if answer_type == int:
            answers[req.name] = np.random.randint(-100, 100)
        elif answer_type == float:
            answers[req.name] = np.random.random()
        elif answer_type == bool:
            answers[req.name] = np.random.random() > 0.5
        elif answer_type == str:
            answers[req.name] = np.random.choice(["a", "b", "c"])
        else:
            raise ValueError(f"Unsupported answer type: {answer_type}")
    return answers


class RubricWithAnswers(Rubric):
    answers: dict[str, t.Any]

    @classmethod
    def from_rubric_and_answers(cls, rubric: Rubric, answers: dict[str, t.Any]) -> "RubricWithAnswers":
        return cls(requirements=rubric.requirements, answers=answers)


    def run_metrics(self) -> dict[str, float]:
        assert self.solved, "Rubric must be solved to run metrics"
        assert all(k in self.answers for k in [req.name for req in self.requirements.get_all_requirements()]), "All requirements must have answers"

        req_results = {k:v._result.score for k,v in self.requirements.requirements.items()}
        metrics_functions_per_req = {k: map_answer_type_to_metrics(type(v)) for k,v in req_results.items()}

        req_to_metric_to_score = defaultdict(dict)
        for req_name, metric_functions in metrics_functions_per_req.items():
            for metric_name, metric_func in metric_functions.items():
                req_to_metric_to_score[req_name][metric_name] = metric_func(self.answers[req_name], req_results[req_name])

        metric_to_results = defaultdict(list)
        for req_name, metric_to_score in req_to_metric_to_score.items():
            for metric_name, score in metric_to_score.items():
                metric_to_results[metric_name].append(score)

        metric_to_mean_score = {k: dict(mean=np.mean(v), std=np.std(v), n=len(v)) for k,v in metric_to_results.items()}
        breakpoint()

        # metrics = {}
        # for req in self.requirements.get_all_requirements():
        #     metrics[req.name] = map_scoring_config_to_metric(self.answers[req.name])
        # return metrics

