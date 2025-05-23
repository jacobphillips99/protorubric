"""
Methods for aggregating the results of several evaluated queries.
Returns an AggregatedQueryConfig object that contains the queries, the score, and may contain a confidence score.
"""

import typing as t

import numpy as np
from scipy.stats import mode
import yaml

from open_rubric.base import BaseConfig
from open_rubric.query import NULL_QUERY_CONFIG, QueryConfig
from open_rubric.scoring import (
    BinaryScoringConfig,
    ScoringConfig,
    continuous_scoring_configs,
    discrete_scoring_configs,
)


class AggregatedQueryConfig(BaseConfig):
    queries: list[QueryConfig]
    score: t.Any
    confidence: t.Optional[t.Any] = None

    @property
    def n_votes(self) -> int:
        return len(self.queries)

    def get_scores(self) -> list[t.Any]:
        return [query.score for query in self.queries]

    def get_reasonings(self) -> list[t.Any]:
        return [query.reasoning for query in self.queries]


class BaseAggregatingConfig(BaseConfig):
    name: str
    subtype: str
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "BaseAggregatingConfig":
        if isinstance(data, str):
            if data in name_to_aggregating_config:
                return name_to_aggregating_config[data](**kwargs)
            else:
                raise ValueError(f"Cannot find aggregating config with name {data}")
        elif isinstance(data, dict):
            agg_type = data.pop("subtype")
            name = data.pop("name", agg_type)
            if name in name_to_aggregating_config:
                return name_to_aggregating_config[name](**{**data, "name": name})
            else:
                raise ValueError(f"Cannot find aggregating config with name {name}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
    
    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "BaseAggregatingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    def get_scores(self, queries: list[QueryConfig]) -> list[t.Any]:
        return [query.score for query in queries]

    def check_queries(self, queries: list[QueryConfig]) -> bool:
        if self.valid_scoring_configs is not None:
            for query in queries:
                assert (
                    (query == NULL_QUERY_CONFIG)
                    or type(query.scoring_config) in self.valid_scoring_configs
                ), f"Invalid scoring config: {type(query.scoring_config)} for query {str(query)} in agg config {self.name} with valid configs {self.valid_scoring_configs}"
        if not all(
            [(query == NULL_QUERY_CONFIG) or query.been_answered for query in queries]
        ):
            raise ValueError(
                f"All queries must be answered before aggregation; got queries: {queries}"
            )
        return True

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class NullAggregatingConfig(BaseAggregatingConfig):
    """
    Placeholder for no aggregation, just returns the first query as an "aggregated" query
    """

    name: str = "null"
    subtype: str = "null"
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
    subtype: str = "mean"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = np.mean(self.get_scores(queries))
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class MedianAggregatingConfig(BaseAggregatingConfig):
    name: str = "median"
    subtype: str = "median"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)

        score = np.median(self.get_scores(queries))
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class ModeAggregatingConfig(BaseAggregatingConfig):
    name: str = "mode"
    subtype: str = "mode"
    valid_scoring_configs: list[type[ScoringConfig]] = discrete_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        unique_scores, counts = np.unique(scores, return_counts=True)
        max_idx = np.argmax(counts)
        score = unique_scores[max_idx]
        count = counts[max_idx]
        conf = count / len(scores)
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=conf,
        )


class AllAggregatingConfig(BaseAggregatingConfig):
    name: str = "all"
    subtype: str = "all"
    valid_scoring_configs: list[type[ScoringConfig]] = [BinaryScoringConfig]

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = all(self.get_scores(queries))
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=1 if score else 0,
        )


class AnyAggregatingConfig(BaseAggregatingConfig):
    name: str = "any"
    subtype: str = "any"
    valid_scoring_configs: list[type[ScoringConfig]] = [BinaryScoringConfig]

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = any(self.get_scores(queries))
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=1 if score else 0,
        )


class MaxAggregatingConfig(BaseAggregatingConfig):
    name: str = "max"
    subtype: str = "max"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = max(self.get_scores(queries))
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class MinAggregatingConfig(BaseAggregatingConfig):
    name: str = "min"
    subtype: str = "min"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        self.check_queries(queries)
        score = min(self.get_scores(queries))
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class WeightedAggregatingConfig(BaseAggregatingConfig):
    name: str = "weighted"
    subtype: str = "weighted"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs
    weights: t.Optional[list[float]] = None

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs or self.weights is not None
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs}"
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class WeightedSumAggregatingConfig(BaseAggregatingConfig):
    name: str = "weighted_sum"
    subtype: str = "weighted_sum"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs or self.weights is not None
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs} and weights: {self.weights}"
        breakpoint()
        weights = kwargs["weights"] if "weights" in kwargs else self.weights
        assert len(weights) == len(
            queries
        ), f"Weights must be the same length as queries; got weights: {weights} and queries: {queries}"
        self.check_queries(queries)
        breakpoint()
        score = sum([query.score * weight for query, weight in zip(queries, weights)])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )
    
class WeightedAverageAggregatingConfig(WeightedAggregatingConfig):
    name: str = "weighted_average"
    subtype: str = "weighted_average"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs
    
    def __call__(self, queries: list[QueryConfig], **kwargs: t.Any) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs or self.weights is not None), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs} and weights: {self.weights}"
        weights = kwargs["weights"] if "weights" in kwargs else self.weights
        breakpoint()
        assert len(weights) == len(
            queries
        ), f"Weights must be the same length as queries; got weights: {weights} and queries: {queries}"
        self.check_queries(queries)
        score = sum([query.score * weight for query, weight in zip(queries, weights)]) / sum(weights)
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

preset_aggregator_configs = [NullAggregatingConfig(), MeanAggregatingConfig(), MedianAggregatingConfig(), ModeAggregatingConfig(), AllAggregatingConfig(), AnyAggregatingConfig(), MaxAggregatingConfig(), MinAggregatingConfig()]

class AggregatorConfigs(BaseConfig):
    aggregator_configs: dict[str, BaseAggregatingConfig]
    
    @classmethod
    def from_data(cls, data: list[dict] | dict, **kwargs: t.Any) -> "AggregatorConfigs":
        if isinstance(data, dict):
            if "aggregator_configs" in data:
                list_data: list[dict] = data["aggregator_configs"]
        else:
            list_data = data
        configs: list[BaseAggregatingConfig] = []
        for item in list_data:
            if isinstance(item, str) and item.endswith(
                ".yaml"
            ):  # recursive loading of aggregator configs
                configs.extend(AggregatorConfigs.from_yaml(item, **kwargs).aggregator_configs.values())
            else:
                configs.append(BaseAggregatingConfig.from_data(item, **kwargs))
        all_names = [config.name for config in configs]
        for present_config in preset_aggregator_configs:
            if present_config.name not in all_names:
                configs.append(present_config)
        # todo: check for duplicate names
        return cls(aggregator_configs={config.name: config for config in configs})

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "AggregatorConfigs":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            data = data["aggregator_configs"]
        return cls.from_data(data, **kwargs)
    

    def get_config_by_name(self, name: str) -> type[BaseAggregatingConfig]:
        if name not in self.aggregator_configs:
            raise ValueError(f"Cannot find aggregating config with name {name}; got aggregator configs: {self.aggregator_configs.keys()}")
        return self.aggregator_configs[name]