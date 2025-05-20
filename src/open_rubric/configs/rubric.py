import typing as t

import yaml

from open_rubric.configs.aggregating import AggregatedQueryConfig, aggregator_configs
from open_rubric.configs.base import BaseConfig
from open_rubric.configs.evaluator import EvaluatorConfigs
from open_rubric.configs.requirement import Requirements
from open_rubric.configs.scoring import ScoringConfigs


class Rubric(BaseConfig):
    scoring_configs: ScoringConfigs
    requirements: Requirements

    @classmethod
    def from_data(cls, data: t.Any, **kwargs: t.Any) -> "Rubric":
        assert "scoring_configs" in data, f"Rubric must contain scoring_configs; got {data.keys()}"
        assert "requirements" in data, f"Rubric must contain requirements; got {data.keys()}"
        scoring_configs = ScoringConfigs.from_data_or_yaml(data["scoring_configs"])
        evaluator_configs = EvaluatorConfigs.from_data_or_yaml(data["evaluators"])
        requirements = Requirements.from_data(
            data["requirements"],
            scoring_configs=scoring_configs,
            evaluator_configs=evaluator_configs,
            aggregator_configs=aggregator_configs,
        )
        return cls(scoring_configs=scoring_configs, requirements=requirements)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "Rubric":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    def evaluate(self) -> dict[str, AggregatedQueryConfig]:
        # solve the DAG of requirements, skip for now. just loop through
        results: dict[str, AggregatedQueryConfig] = dict()
        for req in self.requirements.requirements.values():
            if req.query.dependency_names is None:
                result = req.evaluate(dict())
            else:
                dependent_results = {
                    dep_name: results[dep_name] for dep_name in req.query.dependency_names
                }
                result = req.evaluate(dependent_results)
            results[req.name] = result
        return results
