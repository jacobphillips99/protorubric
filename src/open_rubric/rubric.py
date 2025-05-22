import asyncio
import time
import typing as t

import yaml

from open_rubric.aggregators import AggregatedQueryConfig, AggregatorConfigs, aggregator_configs
from open_rubric.base import BaseConfig
from open_rubric.dag import topological_levels
from open_rubric.evaluators import EvaluatorConfigs
from open_rubric.requirement import RequirementConfig, Requirements
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

    def solve(
        self, inputs: t.Optional[dict[str, t.Any]] = None
    ) -> dict[str, AggregatedQueryConfig]:
        # solve the DAG of requirements; prepend any requirements without dependencies
        results: dict[str, AggregatedQueryConfig] = dict()
        level_sorted_reqs = [
            [self.requirements.get_requirement_by_name(req) for req in level]
            for level in topological_levels(self.requirements.dependencies)
        ]
        print(f"\n\nFound {len(level_sorted_reqs)} levels")
        for i, level in enumerate(level_sorted_reqs):
            print("--------------------------------")
            print(
                f"Solving level {i + 1} of {len(level_sorted_reqs)} over requirements: {[req.name for req in level]}"
            )
            tic = time.time()
            level_results = asyncio.run(self.solve_level(level, results))
            toc = time.time()
            print(
                f"Solved level {i + 1} of {len(level_sorted_reqs)} in {round(toc - tic, 2)} seconds"
            )
            results.update(level_results)
        return results

    async def solve_level(
        self, level: list[RequirementConfig], results: dict[str, AggregatedQueryConfig]
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
