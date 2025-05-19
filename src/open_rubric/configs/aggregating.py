"""
Decide on output of aggregation?
"""

import typing as t

import numpy as np
from scipy.stats import mode

from open_rubric.configs.base import BaseConfig
from open_rubric.configs.requirement import RequirementConfig
from open_rubric.configs.scoring import (
    BinaryScoringConfig,
    ScoringConfig,
    continuous_scoring_configs,
    discrete_scoring_configs,
)


class AggregatedRequirementConfig(RequirementConfig):
    requirements: list[RequirementConfig]
    score: t.Any
    confidence: t.Optional[t.Any] = None

    @property
    def n_votes(self) -> int:
        return len(self.requirements)


class BaseAggregatingConfig(BaseConfig):
    name: str
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None

    def check_requirements(self, requirements: list[RequirementConfig]) -> bool:
        if self.valid_scoring_configs is not None:
            for req in requirements:
                assert (
                    req.scoring_config in self.valid_scoring_configs
                ), f"Invalid scoring config: {req.scoring_config} for requirement {req.name} in agg config {self.name} with valid configs {self.valid_scoring_configs}"
        return True

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class NullAggregatingConfig(BaseAggregatingConfig):
    """
    Placeholder for no aggregation, just returns the first requirement as an "aggregated" requirement
    """

    name: str = "null"
    valid_scoring_configs = None

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=requirements[0].score,
            confidence=1,
        )


class MeanAggregatingConfig(BaseAggregatingConfig):
    name: str = "mean"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        score = np.mean([req.score for req in requirements])
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=None,
        )


class MedianAggregatingConfig(BaseAggregatingConfig):
    name: str = "median"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        score = np.median([req.score for req in requirements])
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=None,
        )


class ModeAggregatingConfig(BaseAggregatingConfig):
    name: str = "mode"
    valid_scoring_configs = discrete_scoring_configs

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        scores = [req.score for req in requirements]
        score, count = mode(scores)
        conf = count / len(scores)
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=conf,
        )


class AllAggregatingConfig(BaseAggregatingConfig):
    name: str = "all"
    valid_scoring_configs = [BinaryScoringConfig]

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        score = all([req.score for req in requirements])
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=1 if score else 0,
        )


class AnyAggregatingConfig(BaseAggregatingConfig):
    name: str = "any"
    valid_scoring_configs = [BinaryScoringConfig]

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        score = any([req.score for req in requirements])
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=1 if score else 0,
        )


class MaxAggregatingConfig(BaseAggregatingConfig):
    name: str = "max"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        score = max([req.score for req in requirements])
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=None,
        )


class MinAggregatingConfig(BaseAggregatingConfig):
    name: str = "min"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        self.check_requirements(requirements)
        score = min([req.score for req in requirements])
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=None,
        )


class WeightedAggregatingConfig(BaseAggregatingConfig):
    name: str = "weighted"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        assert (
            "weights" in kwargs
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs}"
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class WeightedSumAggregatingConfig(BaseAggregatingConfig):
    name: str = "weighted_sum"
    valid_scoring_configs = continuous_scoring_configs

    def __call__(
        self, requirements: list[RequirementConfig], **kwargs: t.Any
    ) -> AggregatedRequirementConfig:
        assert (
            "weights" in kwargs
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs}"
        weights = kwargs["weights"]
        assert len(weights) == len(
            requirements
        ), f"Weights must be the same length as requirements; got weights: {weights} and requirements: {requirements}"
        self.check_requirements(requirements)
        score = sum([req.score * weight for req, weight in zip(requirements, weights)])
        return AggregatedRequirementConfig(
            requirements=requirements,
            score=score,
            confidence=None,
        )


name_to_aggregating_config = {
    "null": NullAggregatingConfig,
    "mean": MeanAggregatingConfig,
    "median": MedianAggregatingConfig,
    "mode": ModeAggregatingConfig,
    "all": AllAggregatingConfig,
    "any": AnyAggregatingConfig,
    "max": MaxAggregatingConfig,
    "min": MinAggregatingConfig,
}

all_aggregating_configs = list(name_to_aggregating_config.values())
