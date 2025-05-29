import asyncio
import os
import pickle
import time
import typing as t

import yaml

from open_rubric.configs.aggregating import AggregatedQueryConfig, AggregatorConfigs
from open_rubric.configs.base import BaseConfig
from open_rubric.configs.evaluating import EvaluatorConfigs
from open_rubric.configs.requirement import RequirementConfig, Requirements
from open_rubric.configs.scoring import ScoringConfigs
from open_rubric.utils.dag import topological_levels


class Rubric(BaseConfig):
    requirements: Requirements

    @classmethod
    def from_data(cls, data: t.Any, **kwargs: t.Any) -> "Rubric":
        assert "scoring_configs" in data, f"Rubric must contain scoring_configs; got {data.keys()}"
        assert "requirements" in data, f"Rubric must contain requirements; got {data.keys()}"
        scoring_configs = ScoringConfigs.from_data_or_yaml(data["scoring_configs"])
        evaluator_configs = EvaluatorConfigs.from_data_or_yaml(data["evaluator_configs"])
        aggregator_configs = (
            AggregatorConfigs.from_data_or_yaml(data["aggregator_configs"])
            if "aggregator_configs" in data
            else AggregatorConfigs.from_data([])
        )
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

    def setup_graph(self, inputs: t.Any) -> list[list[RequirementConfig]]:
        # check if inputs need to be added to requirements
        self.requirements.update_with_inputs(inputs)

        # get topological levels of requirements via dependencies
        level_sorted_requirement_names = topological_levels(self.requirements.dependencies)
        level_sorted_reqs = [
            [self.requirements.get_requirement_by_name(req) for req in level]
            for level in level_sorted_requirement_names
        ]
        print(f"\n\nFound {len(level_sorted_reqs)} levels")
        return level_sorted_reqs

    def update_state(
        self,
        state: dict[str, AggregatedQueryConfig],
        level_results: dict[str, AggregatedQueryConfig],
    ) -> dict[str, AggregatedQueryConfig]:
        """
        Updates the state dictionary with the results of the current level.
        """
        state.update(level_results)
        return state

    async def asolve(
        self,
        inputs: t.Any,  # TODO: fix any
    ) -> dict[str, AggregatedQueryConfig]:
        # setup graph of requirements and their dependencies
        level_sorted_reqs = self.setup_graph(inputs)

        # initialize results / solved requirements dictionary
        results: dict[str, AggregatedQueryConfig] = dict()
        state: dict[str, AggregatedQueryConfig] = dict()

        for i, level in enumerate(level_sorted_reqs):
            print("-" * 100)
            print(
                f"Solving level {i + 1} of {len(level_sorted_reqs)} over requirements: {[req.name for req in level]}"
            )
            tic = time.time()
            level_results = await self.asolve_level(level, state)
            toc = time.time()
            print(
                f"Solved level {i + 1} of {len(level_sorted_reqs)} in {round(toc - tic, 2)} seconds"
            )
            results.update(level_results)
            state = self.update_state(state, level_results)
            print(f"len results: {len(results)}")
        print("-" * 100)
        return results

    def solve(self, inputs: t.Any) -> dict[str, AggregatedQueryConfig]:
        return asyncio.run(self.asolve(inputs))

    async def asolve_level(
        self,
        level: list[RequirementConfig],
        state: dict[str, AggregatedQueryConfig],
    ) -> dict[str, AggregatedQueryConfig]:
        """
        Solves one level of the rubric's DAG of requirements with the current state of the rubric.
        Returns a dictionary of results for each requirement in the level.
        """
        payloads: list[tuple[RequirementConfig, dict[str, t.Any]]] = []
        for req in level:
            dependent_results = (
                {dep_name: state[dep_name] for dep_name in req.dependency_names}
                if req.dependency_names is not None
                else None
            )
            payloads.append((req, {"dependent_results": dependent_results}))
        agg_query_results = await asyncio.gather(
            *[req.async_evaluate(**payload) for req, payload in payloads]
        )
        return {req.name: aqr for req, aqr in zip(level, agg_query_results)}

    @property
    def solved(self) -> bool:
        return all(req.solved for req in self.requirements.get_all_requirements())

    def save_pkl(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pkl(cls, path: str) -> "Rubric":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rubric pkl file not found at {path}")
        with open(path, "rb") as f:
            rubric: "Rubric" = pickle.load(f)
        return rubric
