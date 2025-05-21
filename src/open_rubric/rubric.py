import typing as t

import yaml

from open_rubric.aggregators import AggregatedQueryConfig, AggregatorConfigs, aggregator_configs
from open_rubric.base import BaseConfig
from open_rubric.evaluators import EvaluatorConfigs
from open_rubric.requirement import Requirements
from open_rubric.scoring import ScoringConfigs


class Rubric(BaseConfig):
    requirements: Requirements
    scoring_configs: ScoringConfigs
    evaluators: EvaluatorConfigs
    aggregator_configs: AggregatorConfigs

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
        return cls(
            requirements=requirements,
            scoring_configs=scoring_configs,
            evaluators=evaluator_configs,
            aggregator_configs=aggregator_configs,
        )

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "Rubric":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    def solve(self) -> dict[str, AggregatedQueryConfig]:
        # solve the DAG of requirements, skip for now. just loop through as the reqs are provided in a topological sort
        # TODO: solve the DAG of requirements
        results: dict[str, AggregatedQueryConfig] = dict()
        for req in self.requirements.requirements.values():
            dependent_results = (
                {dep_name: results[dep_name] for dep_name in req.dependency_names}
                if req.dependency_names is not None
                else None
            )
            result = req.evaluate(dependent_results)
            results[req.name] = result
        return results
