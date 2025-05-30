import copy
import typing as t

import numpy as np

from open_rubric.configs.aggregating import AggregatedQueryConfig
from open_rubric.configs.answers import ANSWER_TYPE_TO_SCORE_TYPE_MAPPING
from open_rubric.configs.query import NULL_QUERY_CONFIG
from open_rubric.eval.metrics import results_to_metrics
from open_rubric.rubric import Rubric


def generate_test_answers(rubric: Rubric) -> dict[str, t.Any]:
    # random answer generator
    answers = {}
    for req in rubric.requirements.get_all_requirements():
        answer_type = ANSWER_TYPE_TO_SCORE_TYPE_MAPPING.get(
            req.query.scoring_config.answer_type, str
        )
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
    teacher_force: bool = False  # whether or not to use the answers DURING rubric evaluation
    _results_metrics: t.Optional[dict[str, dict[str, dict[str, float]]]] = (
        None  # cached results metrics
    )

    @classmethod
    def from_rubric_and_answers(
        cls, rubric: Rubric, answers: dict[str, t.Any], **kwargs: t.Any
    ) -> "RubricWithAnswers":
        return cls(requirements=rubric.requirements, answers=answers, **kwargs)

    def run_metrics(self) -> dict[str, dict[str, dict[str, float]]]:
        assert self.solved, "Rubric must be solved to run metrics"
        assert all(
            k in self.answers
            for k in [req.name for req in self.requirements.get_all_requirements()]
        ), "All requirements must have answers"

        # {Requirement name: score}
        req_to_results = {
            k: v._result.score
            for k, v in self.requirements.requirements.items()
            if v._result is not None
        }
        req_to_metric_to_score, metric_to_mean_score = results_to_metrics(
            req_to_results, self.answers
        )
        self._results_metrics = {
            "requirements": req_to_metric_to_score,
            "metrics": metric_to_mean_score,
        }
        return self._results_metrics

    def update_state(
        self,
        state: dict[str, AggregatedQueryConfig],
        level_results: dict[str, AggregatedQueryConfig],
    ) -> dict[str, AggregatedQueryConfig]:
        """
        Adds teacher forcing to the state dictionary. If teacher_force is True, the state dictionary is updated with the answers.
        # TODO: currently adding a null query config because we don't have a reasoning / scoring config for the answer. need to handle prompting
        """
        for req_name, agg_query_config in level_results.items():
            # TODO: should probably use the original AQC here somehow?
            if self.teacher_force and req_name in self.answers:
                scoring_config = agg_query_config.queries[0].scoring_config
                this_null_query_config = copy.deepcopy(NULL_QUERY_CONFIG)
                this_null_query_config.scoring_config = scoring_config
                state[req_name] = AggregatedQueryConfig(
                    queries=[this_null_query_config],
                    score=self.answers[req_name],
                    confidence=None,
                )
                print(f"Teacher forced {req_name} to {self.answers[req_name]}")
            else:
                state[req_name] = agg_query_config
        return state

    def save_pkl(self, path: str) -> None:
        if not self._results_metrics:
            self.run_metrics()
        super().save_pkl(path)
