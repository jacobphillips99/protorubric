"""
Decide on output of aggregation?
"""
import typing as t 
import numpy as np
from open_rubric.configs.base import BaseConfig
from open_rubric.configs.scoring import ScoringConfig, DiscreteScoringConfig, ContinuousScoringConfig, discrete_scoring_configs, continuous_scoring_configs
from open_rubric.configs.requirement import RequirementConfig


class BaseAggregatingConfig(BaseConfig):
    name: str
    valid_scoring_configs: t.Optional[list[ScoringConfig]] = None

    def check_requirements(self, requirements: list[RequirementConfig]) -> bool:
        if self.valid_scoring_configs is not None:
            assert all(type(req.score) in self.valid_scoring_configs for req in requirements)


    def __call__(self, requirements: list[RequirementConfig], **kwargs: t.Any) -> t.Any:
        self.check_requirements(requirements)
        pass

class NullAggregatingConfig(BaseAggregatingConfig):
    name: str = "null"
    valid_scoring_configs = None

    def __call__(self, requirements: list[ScoringConfig], **kwargs: t.Any) -> t.Any:
        return [req.score for req in requirements]


class MeanAggregatingConfig(BaseAggregatingConfig):
    name: str = "mean"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(self, requirements: list[ContinuousScoringConfig], **kwargs: t.Any) -> t.Any:
        self.check_requirements(requirements)
        return np.mean([req.score for req in requirements])


class MedianAggregatingConfig(BaseAggregatingConfig):
    name: str = "median"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(self, requirements: list[ContinuousScoringConfig], **kwargs: t.Any) -> t.Any:
        self.check_requirements(requirements)
        return np.median([req.score for req in requirements])


class ModeAggregatingConfig(BaseAggregatingConfig):
    name: str = "mode"
    valid_scoring_configs = discrete_scoring_configs

    def __call__(self, requirements: list[DiscreteScoringConfig], **kwargs: t.Any) -> t.Any:
        self.check_requirements(requirements)
        return np.mode([req.score for req in requirements])
