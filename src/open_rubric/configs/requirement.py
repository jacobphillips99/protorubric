import typing as t

import yaml
from pydantic import model_validator

from open_rubric.configs.aggregating import (
    AggregatedQueryConfig,
    AggregatorConfigs,
    BaseAggregatingConfig,
    NullAggregatingConfig,
)
from open_rubric.configs.base import BaseConfig
from open_rubric.configs.evaluating import BaseEvaluatorConfig, EvaluatorConfigs
from open_rubric.configs.query import QueryConfig
from open_rubric.configs.scoring import ScoringConfig, ScoringConfigs


class RequirementConfig(BaseConfig):
    name: str
    query: QueryConfig
    evaluator: BaseEvaluatorConfig
    dependency_names: t.Optional[list[str]] = None
    aggregator: BaseAggregatingConfig = NullAggregatingConfig()
    _result: t.Optional[AggregatedQueryConfig] = None

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "RequirementConfig":
        # replace string scoring_config in Query with ScoringConfig object
        assert all(
            k in kwargs for k in ["evaluator_configs", "aggregator_configs"]
        ), f"Missing required kwargs [evaluator_configs, aggregator_configs]. Found kwargs: {kwargs.keys()}"

        scoring_configs: ScoringConfigs = kwargs["scoring_configs"]
        query = data["query"]
        if "scoring_config" in query and not isinstance(query["scoring_config"], ScoringConfig):
            query["scoring_config"] = scoring_configs.get_config_by_name(query["scoring_config"])

        data["query"] = QueryConfig.from_data(query, **kwargs)
        if "query" in data and isinstance(data["query"], str):
            data["query"] = QueryConfig.from_yaml(data["query"], **kwargs)

        # replace string evaluator in data with EvaluatorConfig object
        evaluator_configs: EvaluatorConfigs = kwargs["evaluator_configs"]
        evaluator: BaseEvaluatorConfig = evaluator_configs.get_config_by_name(data["evaluator"])
        data["evaluator"] = evaluator

        # replace string aggregator in data with AggregatorConfig object
        aggregator_configs: AggregatorConfigs = kwargs["aggregator_configs"]
        agg_name = data.get("aggregator", "null")
        aggregator: BaseAggregatingConfig = aggregator_configs.get_config_by_name(agg_name)
        data["aggregator"] = aggregator
        return cls(**data)

    async def async_evaluate(
        self, dependent_results: t.Optional[dict[str, AggregatedQueryConfig]] = None
    ) -> AggregatedQueryConfig:
        evaluated_queries = await self.evaluator.async_call(self.query, dependent_results)
        aggregated_query = self.aggregator(evaluated_queries)
        self._result = aggregated_query
        print(
            f"{self.name}: {self._result.score} over {self._result.n_votes} votes aggregated by {self.aggregator.name}"
        )
        return aggregated_query

    def set_inputs(self, inputs: t.Any) -> None:
        self.query.inputs = inputs


class Requirements(BaseConfig):
    requirements: dict[str, RequirementConfig]
    dependencies: dict[str, t.Optional[list[str]]]

    @classmethod
    def from_data(cls, data: list[dict] | dict, **kwargs: t.Any) -> "Requirements":
        assert all(k in kwargs for k in ["scoring_configs", "evaluator_configs", "aggregator_configs"]), f"Missing required kwargs [scoring_configs, evaluator_configs, aggregator_configs]. Found kwargs: {kwargs.keys()}"
        reqs = []
        for req in data:
            reqs.append(RequirementConfig.from_data(req, **kwargs))
        all_names = [req.name for req in reqs]
        assert len(all_names) == len(set(all_names)), f"Duplicate requirement names! {all_names}"
        requirement_dict = {req.name: req for req in reqs}
        dependency_dict = {req.name: req.dependency_names for req in reqs}
        return cls(requirements=requirement_dict, dependencies=dependency_dict)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "Requirements":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            data = data["requirements"]
        return cls.from_data(data, **kwargs)

    def get_requirement_by_name(self, name: str) -> RequirementConfig:
        if name not in self.requirements:
            raise ValueError(
                f"Requirement {name} not found in requirements; got requirements: {self.requirements.keys()}"
            )
        return self.requirements[name]

    def get_dependencies_by_name(self, name: str) -> t.Optional[list[str]]:
        return self.dependencies.get(name, None)

    def get_all_requirements(self) -> list[RequirementConfig]:
        return list(self.requirements.values())

    @model_validator(mode="before")
    def check_dependencies(cls, data: dict) -> dict:
        if "dependencies" not in data:
            deps = dict()
            for req in data["requirements"].values():
                deps[req.name] = req.dependency_names
            data["dependencies"] = deps
        return data
