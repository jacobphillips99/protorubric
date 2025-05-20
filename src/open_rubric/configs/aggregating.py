"""
Decide on output of aggregation?
"""

import typing as t

import numpy as np
from scipy.stats import mode

from open_rubric.configs.base import BaseConfig
from open_rubric.configs.query import QueryConfig
from open_rubric.configs.scoring import (
    BinaryScoringConfig,
    ScoringConfig,
    continuous_scoring_configs,
    discrete_scoring_configs,
)


class AggregatedQueryConfig(QueryConfig):
    queries: list[QueryConfig]
    score: t.Any
    confidence: t.Optional[t.Any] = None

    @property
    def n_votes(self) -> int:
        return len(self.queries)


class BaseAggregatingConfig(BaseConfig):
    name: str
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None

    def check_queries(self, queries: list[QueryConfig]) -> bool:
        if self.valid_scoring_configs is not None:
            for req in queries:
                assert (
                    req.scoring_config in self.valid_scoring_configs
                ), f"Invalid scoring config: {req.scoring_config} for query {req.name} in agg config {self.name} with valid configs {self.valid_scoring_configs}"
        return True

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class NullAggregatingConfig(BaseAggregatingConfig):
    """
    Placeholder for no aggregation, just returns the first query as an "aggregated" query
    """

    name: str = "null"
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        return AggregatedQueryConfig(
            queries=queries,
            score=queries[0].score,
            confidence=1,
        )


class MeanAggregatingConfig(BaseAggregatingConfig):
    name: str = "mean"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = np.mean([req.score for req in queries])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class MedianAggregatingConfig(BaseAggregatingConfig):
    name: str = "median"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = np.median([req.score for req in queries])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class ModeAggregatingConfig(BaseAggregatingConfig):
    name: str = "mode"
    valid_scoring_configs: list[type[ScoringConfig]] = discrete_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = [req.score for req in queries]
        score, count = mode(scores)
        conf = count / len(scores)
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=conf,
        )


class AllAggregatingConfig(BaseAggregatingConfig):
    name: str = "all"
    valid_scoring_configs: list[type[ScoringConfig]] = [BinaryScoringConfig]

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = all([req.score for req in queries])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=1 if score else 0,
        )


class AnyAggregatingConfig(BaseAggregatingConfig):
    name: str = "any"
    valid_scoring_configs: list[type[ScoringConfig]] = [BinaryScoringConfig]

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = any([req.score for req in queries])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=1 if score else 0,
        )


class MaxAggregatingConfig(BaseAggregatingConfig):
    name: str = "max"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = max([req.score for req in queries])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class MinAggregatingConfig(BaseAggregatingConfig):
    name: str = "min"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = min([req.score for req in queries])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class WeightedAggregatingConfig(BaseAggregatingConfig):
    name: str = "weighted"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs}"
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class WeightedSumAggregatingConfig(BaseAggregatingConfig):
    name: str = "weighted_sum"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs}"
        weights = kwargs["weights"]
        assert len(weights) == len(
            queries
        ), f"Weights must be the same length as queries; got weights: {weights} and queries: {queries}"
        self.check_queries(queries)
        score = sum([req.score * weight for req, weight in zip(queries, weights)])
        return AggregatedQueryConfig(
            queries=queries,
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
    "weighted": WeightedAggregatingConfig,
    "weighted_sum": WeightedSumAggregatingConfig,
}

all_aggregating_configs = list(name_to_aggregating_config.values())


class AggregatedQueryConfigs(BaseConfig):
    aggregators: dict[str, type[BaseAggregatingConfig]] = name_to_aggregating_config

    def get_config_by_name(self, name: str) -> type[BaseAggregatingConfig]:
        if name not in self.aggregators:
            return self.aggregators["null"]
        return self.aggregators[name]


aggregator_configs = AggregatedQueryConfigs()
