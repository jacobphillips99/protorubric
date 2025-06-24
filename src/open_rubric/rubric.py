import asyncio
import os
import pickle
import time
import typing as t

import yaml

from protorubric.configs.aggregating import AggregatedQueryConfig, AggregatorConfigCollector
from protorubric.configs.base import BaseConfig
from protorubric.configs.evaluating import EvaluatorConfigCollector
from protorubric.configs.requirement import RequirementConfig, Requirements
from protorubric.configs.scoring import ScoringConfigCollector
from protorubric.utils.dag import topological_levels


class Rubric(BaseConfig):
    requirements: Requirements
    _levels: t.Optional[list[list[RequirementConfig]]] = None

    @classmethod
    def from_data(cls, data: t.Any, **kwargs: t.Any) -> "Rubric":
        assert "requirements" in data, f"Rubric must contain requirements; got {data.keys()}"
        scoring_configs = ScoringConfigCollector.from_data_or_yaml(data.get("scoring_configs"))
        evaluator_configs = EvaluatorConfigCollector.from_data_or_yaml(
            data.get("evaluator_configs")
        )
        aggregator_configs = AggregatorConfigCollector.from_data_or_yaml(
            data.get("aggregator_configs")
        )
        requirements = Requirements.from_data(
            data["requirements"],
            scoring_configs=scoring_configs,
            evaluator_configs=evaluator_configs,
            aggregator_configs=aggregator_configs,
        )
        return cls(requirements=requirements)

    @property
    def levels(self) -> list[list[RequirementConfig]]:
        if self._levels is None:
            self._levels = self.setup_graph()
        return self._levels

    def setup_graph(self) -> list[list[RequirementConfig]]:
        """
        Sets up the graph of requirements and their dependencies.
        Returns a list of levels, where each level is a list of requirements that can be solved in parallel.
        """
        # get topological levels of requirements via dependencies
        level_sorted_requirement_names = topological_levels(self.requirements.dependencies)
        level_sorted_reqs = [
            [self.requirements.get_requirement_by_name(req) for req in level]
            for level in level_sorted_requirement_names
        ]
        return level_sorted_reqs

    def print_levels(self) -> None:
        print(f"\n\nFound {len(self.levels)} levels:")
        for i, level in enumerate(self.levels):
            level_strs = [f"{req.name} ({req.query.scoring_config.name})" for req in level]
            print(f"Level {i + 1}: [{', '.join(level_strs)}]")

    def update_state(
        self,
        state: dict[str, AggregatedQueryConfig],
        level_results: dict[str, AggregatedQueryConfig],
    ) -> dict[str, AggregatedQueryConfig]:
        """
        Updates the state dictionary with the results of the current level.
        This is here to optionally allow a custom update function to allow e.g. teacher forcing with correct answers
        """
        state.update(level_results)
        return state

    async def asolve(
        self,
        inputs: t.Any,  # TODO: fix any
    ) -> dict[str, AggregatedQueryConfig]:
        """
        Solves the rubric by iteratively solving each level of the DAG of requirements.
        Returns a dictionary of results for each requirement in the rubric.
        """
        # setup graph of requirements and their dependencies
        self.requirements.update_with_inputs(inputs)
        level_sorted_reqs = self.levels
        self.print_levels()

        # initialize results / state dictionary
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

    def describe(self) -> str:
        title = f"Rubric with {len(self.requirements.requirements)} requirements"
        level_descriptions = []
        for i, level in enumerate(self.levels):
            this_level = []
            for req in level:
                this_level.append(f"- {req.name.title()}: {req.query.instruction}{' + (' + ', '.join(req.dependency_names) if req.dependency_names else '' + ')' if req.dependency_names else ''} => ({req.query.scoring_config.name})")
            level_descriptions.append(f"Level {i + 1}:" +  "\n" + '\n'.join(this_level))
        return "\n".join([title, *level_descriptions])
