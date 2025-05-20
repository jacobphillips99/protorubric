import typing as t

import yaml

from open_rubric.configs.base import BaseConfig
from open_rubric.configs.requirement import Requirements
from open_rubric.configs.scoring import ScoringConfigs
from open_rubric.configs.evaluator import EnsembledModelEvaluator


class Rubric(BaseConfig):
    scoring_configs: ScoringConfigs
    requirements: Requirements

    @classmethod
    def from_data(cls, data: t.Any, **kwargs: t.Any) -> "Rubric":
        assert "scoring_configs" in data, f"Rubric must contain scoring_configs; got {data.keys()}"
        assert "requirements" in data, f"Rubric must contain requirements; got {data.keys()}"
        # scoring_configs = ScoringConfigs.from_data_or_yaml(data["scoring_configs"])
        # requirements = Requirements.from_data(data["requirements"], scoring_configs=scoring_configs)

        breakpoint()

        return cls(scoring_configs=scoring_configs, requirements=requirements)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "Rubric":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)
    
    def evaluate(self):
        # solve the DAG of requirements, skip for now. just loop through
        results = dict()
        for req in self.requirements.requirements.values():
            dependent_results = {dep_name: results[dep_name] for dep_name in req.dependency_names}
            results[req.name] = req.evaluate(dependent_results)

