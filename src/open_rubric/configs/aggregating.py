"""
Methods for aggregating the results of several evaluated queries.
Returns an AggregatedQueryConfig object that contains the queries, the score, and may contain a confidence score.
"""

import typing as t

import numpy as np
import yaml

from open_rubric.configs.base import BaseConfig
from open_rubric.configs.query import NULL_QUERY_CONFIG, QueryConfig
from open_rubric.configs.scoring import (
    BinaryScoringConfig,
    FreeTextScoringConfig,
    ScoringConfig,
    continuous_scoring_configs,
    discrete_scoring_configs,
)
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest


class AggregatedQueryConfig(BaseConfig):
    queries: list[
        t.Union[QueryConfig, "AggregatedQueryConfig"]
    ]  # self-reference enables recursive aggregation
    score: t.Any
    confidence: t.Optional[t.Any] = None

    @property
    def n_votes(self) -> int:
        return len(self.queries)

    def get_scores(self) -> list[t.Any]:
        return [query.score for query in self.queries]

    def get_reasonings(self) -> list[str | None]:
        return [query.reasoning for query in self.queries]


class BaseAggregatingConfig(BaseConfig):
    name: str
    subtype: str
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "BaseAggregatingConfig":
        if isinstance(data, str):
            if data in subtype_to_aggregating_config:
                return subtype_to_aggregating_config[data](**kwargs)
            else:
                raise ValueError(f"Cannot find aggregating config with name {data}")
        elif isinstance(data, dict):
            subtype = data.pop("subtype")
            name = data.pop("name", subtype)
            if subtype in subtype_to_aggregating_config:
                return subtype_to_aggregating_config[subtype](**{**data, "name": name})
            else:
                raise ValueError(f"Cannot find aggregating config with subtype {subtype}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "BaseAggregatingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    def get_scores(self, queries: list[QueryConfig | AggregatedQueryConfig]) -> list[t.Any]:
        return [query.score for query in queries]

    def coerce_score_types(self, scores: list[t.Any], new_type: type[t.Any]) -> list[t.Any]:
        return [new_type(score) for score in scores]

    def check_queries(self, queries: list[QueryConfig | AggregatedQueryConfig]) -> bool:
        if self.valid_scoring_configs is not None:
            for query in queries:
                these_queries = (
                    query.queries if isinstance(query, AggregatedQueryConfig) else [query]
                )
                for this_query in these_queries:
                    assert (this_query == NULL_QUERY_CONFIG) or type(
                        this_query.scoring_config
                    ) in self.valid_scoring_configs, f"Invalid scoring config: {type(this_query.scoring_config)} for query in agg config {self.name} with valid configs {self.valid_scoring_configs}"
                    assert (
                        this_query == NULL_QUERY_CONFIG or this_query.been_answered
                    ), f"Query {this_query.instruction} in agg config {self.name} must be answered before aggregation"
        return True

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class NullAggregatingConfig(BaseAggregatingConfig):
    """
    Placeholder for no aggregation, just returns the first query as an "aggregated" query
    """

    name: str = "null"
    subtype: str = "null"
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        return AggregatedQueryConfig(
            queries=queries,
            score=queries[0].score,
            confidence=1,
        )


class MeanAggregatingConfig(BaseAggregatingConfig):
    name: str = "mean"
    subtype: str = "mean"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score_type = type(scores[0])
        if score_type in [str]:
            raise ValueError(f"Cannot take mean of scores of type {score_type}")
        elif score_type in [bool]:
            scores = self.coerce_score_types(scores, int)
        score = np.mean(scores)
        score = score_type(score)
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class MedianAggregatingConfig(BaseAggregatingConfig):
    name: str = "median"
    subtype: str = "median"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score_type = type(scores[0])
        if score_type in [str]:
            raise ValueError(f"Cannot take median of scores of type {score_type}")
        elif score_type in [bool]:
            scores = self.coerce_score_types(scores, int)
        score = np.median(scores)
        score = score_type(score)
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class ModeAggregatingConfig(BaseAggregatingConfig):
    name: str = "mode"
    subtype: str = "mode"
    valid_scoring_configs: list[type[ScoringConfig]] = discrete_scoring_configs

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score_type = type(scores[0])
        unique_scores, counts = np.unique(scores, return_counts=True)
        max_idx = np.argmax(counts)
        score = unique_scores[max_idx]
        score = score_type(score)
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

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
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

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
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

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
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

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
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

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs or self.weights is not None
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs}"
        raise NotImplementedError(f"Aggregating config {self.name} must implement __call__")


class WeightedSumAggregatingConfig(WeightedAggregatingConfig):
    name: str = "weighted_sum"
    subtype: str = "weighted_sum"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs or self.weights is not None
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs} and weights: {self.weights}"
        weights = kwargs["weights"] if "weights" in kwargs else self.weights
        assert len(weights) == len(
            queries
        ), f"Weights must be the same length as queries; got weights: {weights} and queries: {queries}"
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score_type = type(scores[0])
        if score_type in [str]:
            raise ValueError(f"Cannot take weighted sum of scores of type {score_type}")
        elif score_type in [bool]:
            scores = self.coerce_score_types(scores, int)
        score = sum([score * weight for score, weight in zip(scores, weights)])
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class WeightedAverageAggregatingConfig(WeightedAggregatingConfig):
    name: str = "weighted_average"
    subtype: str = "weighted_average"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs or self.weights is not None
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs} and weights: {self.weights}"
        weights = kwargs["weights"] if "weights" in kwargs else self.weights
        assert len(weights) == len(
            queries
        ), f"Weights must be the same length as queries; got weights: {weights} and queries: {queries}"
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score_type = type(scores[0])
        if score_type in [str]:
            raise ValueError(f"Cannot take weighted average of scores of type {score_type}")
        elif score_type in [bool]:
            scores = self.coerce_score_types(scores, int)
        score = sum([score * weight for score, weight in zip(scores, weights)]) / sum(weights)
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=None,
        )


class LLMAggregatingConfig(BaseAggregatingConfig):
    model: str
    name: str = "llm-aggregator"
    subtype: str = "llm-aggregator"
    aggregation_prompt: str = "Aggregate the provided results into a single result."
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = (
        None  # indicates all scoring configs are valid
    )

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scoring_config = FreeTextScoringConfig()
        results = []
        for query in queries:
            if isinstance(query, AggregatedQueryConfig):
                aqc = query
                results.extend(
                    [
                        f"Score: {score}, Reasoning: {reasoning}"
                        for score, reasoning in zip(aqc.get_scores(), aqc.get_reasonings())
                    ]
                )
            else:
                results.append(f"Score: {query.score}, Reasoning: {query.reasoning}")
        results_str = "\n".join(results)
        prompt = f"""
{self.aggregation_prompt}
{results_str}
{scoring_config.to_prompt()}
""".strip()
        # TODO: must get recursive with the queries
        breakpoint()
        req = ModelRequest(
            model=self.model,
            model_input=ModelInput(
                prompt=prompt,
                response_format=type(scoring_config),
            ),
        )
        response = await MODEL.agenerate(req)
        answer = scoring_config.parse_response(response.texts[0])
        return AggregatedQueryConfig(
            queries=queries,
            score=answer.score,
            confidence=None,
        )


subtype_to_aggregating_config = {
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
    "llm": LLMAggregatingConfig,
}

preset_aggregator_configs = [
    NullAggregatingConfig(),
    MeanAggregatingConfig(),
    MedianAggregatingConfig(),
    ModeAggregatingConfig(),
    AllAggregatingConfig(),
    AnyAggregatingConfig(),
    MaxAggregatingConfig(),
    MinAggregatingConfig(),
]


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
                configs.extend(
                    AggregatorConfigs.from_yaml(item, **kwargs).aggregator_configs.values()
                )
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

    def get_config_by_name(self, name: str) -> BaseAggregatingConfig:
        if name not in self.aggregator_configs:
            raise ValueError(
                f"Cannot find aggregating config with name {name}; got aggregator configs: {self.aggregator_configs.keys()}"
            )
        return self.aggregator_configs[name]
