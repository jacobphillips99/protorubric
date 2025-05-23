import asyncio
import time
import typing as t

import yaml

from open_rubric.aggregating import AggregatedQueryConfig, AggregatorConfigs 
from open_rubric.base import BaseConfig
from open_rubric.dag import topological_levels
from open_rubric.evaluating import EvaluatorConfigs
from open_rubric.requirement import RequirementConfig, Requirements
from open_rubric.scoring import ScoringConfigs


class Rubric(BaseConfig):
    requirements: Requirements

    @classmethod
    def from_data(cls, data: t.Any, **kwargs: t.Any) -> "Rubric":
        assert "scoring_configs" in data, f"Rubric must contain scoring_configs; got {data.keys()}"
        assert "requirements" in data, f"Rubric must contain requirements; got {data.keys()}"
        scoring_configs = ScoringConfigs.from_data_or_yaml(data["scoring_configs"])
        evaluator_configs = EvaluatorConfigs.from_data_or_yaml(data["evaluator_configs"])
        aggregator_configs = AggregatorConfigs.from_data_or_yaml(data["aggregator_configs"]) if "aggregator_configs" in data else AggregatorConfigs.from_data([])
        requirements = Requirements.from_data(
            data["requirements"],
            scoring_configs=scoring_configs,
            evaluator_configs=evaluator_configs,
            aggregator_configs=aggregator_configs,
        )
        return cls(requirements=requirements)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "Rubric":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    async def asolve(
        self,
        inputs: t.Any,  # TODO: fix any
    ) -> dict[str, AggregatedQueryConfig]:
        # check if inputs need to be added to requirements
        all_requirements = self.requirements.get_all_requirements()
        for req in all_requirements:
            if not req.query.inputs:
                req.query.inputs = inputs

        results: dict[str, AggregatedQueryConfig] = dict()
        level_sorted_reqs = [
            [self.requirements.get_requirement_by_name(req) for req in level]
            for level in topological_levels(self.requirements.dependencies)
        ]

        print(f"\n\nFound {len(level_sorted_reqs)} levels")
        for i, level in enumerate(level_sorted_reqs):
            print("-" * 100)
            print(
                f"Solving level {i + 1} of {len(level_sorted_reqs)} over requirements: {[req.name for req in level]}"
            )
            tic = time.time()
            level_results = await self.asolve_level(level, results)
            toc = time.time()
            print(
                f"Solved level {i + 1} of {len(level_sorted_reqs)} in {round(toc - tic, 2)} seconds"
            )
            results.update(level_results)
            print(f"len results: {len(results)}")
        print("-" * 100)
        return results

    async def asolve_level(
        self,
        level: list[RequirementConfig],
        results: dict[str, AggregatedQueryConfig],
    ) -> dict[str, AggregatedQueryConfig]:
        payloads: list[tuple[RequirementConfig, dict[str, t.Any]]] = []
        for req in level:
            dependent_results = (
                {dep_name: results[dep_name] for dep_name in req.dependency_names}
                if req.dependency_names is not None
                else None
            )
            payloads.append((req, {"dependent_results": dependent_results}))
        agg_query_results = await asyncio.gather(
            *[req.async_evaluate(**payload) for req, payload in payloads]
        )
        return {req.name: aqr for req, aqr in zip(level, agg_query_results)}
