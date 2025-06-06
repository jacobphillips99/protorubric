"""
Methods for aggregating the results of several evaluated queries.
Returns an AggregatedQueryConfig object that contains the queries, the score, and may contain a confidence score.
"""

import typing as t
from typing import ClassVar

import numpy as np

from open_rubric.configs.base import BaseConfig, BaseConfigCollector
from open_rubric.configs.query import NullQueryConfig, QueryConfig
from open_rubric.configs.scoring import (
    BinaryScoringConfig,
    FreeTextScoringConfig,
    ScoringConfig,
    continuous_scoring_configs,
    discrete_scoring_configs,
)
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest

ReasoningsType = str | None | list["ReasoningsType"]


class AggregatedQueryConfig(BaseConfig):
    """
    A loose wrapper that acts as a container for answered QueryConfigs. This object can hold nested AggregatedQueryConfigs, enabling recursive aggregation.
    """
    queries: list[
        t.Union[QueryConfig, "AggregatedQueryConfig"]
    ]  # self-reference enables recursive aggregation
    score: t.Any
    reasoning: t.Optional[str] = None
    confidence: t.Optional[t.Any] = None

    @property
    def n_votes(self) -> int:
        return len(self.queries)

    def get_scores(self) -> list[t.Any]:
        return [query.score for query in self.queries]

    def get_reasonings(self) -> list[ReasoningsType]:
        return [query.reasoning for query in self.queries]

    def get_example_query_config(self) -> t.Optional[QueryConfig]:
        # recursively find a query config from the AQC
        for query in self.queries:
            if isinstance(query, QueryConfig):
                return query
            else:
                maybe_query_config = query.get_example_query_config()
                if maybe_query_config is not None:
                    return maybe_query_config
        return None

    def to_explanation(self, include_internals: bool = False) -> dict[str, t.Any]:
        """
        Returns the explanation of the current AggregatedQueryConfig.
        If include_internals is True, returns a dict with 'explanation' (str) and 'internals' (list of dicts) keys.
        The 'internals' list contains the explanations of the internal queries, which may be recursive.
        """
        explanation = f"Selected `{self.score}` with reasoning: {self.reasoning}"
        internals = []
        if include_internals:
            # Collect internal explanations
            for query in self.queries:
                if isinstance(query, QueryConfig):
                    internals.append({
                        "explanation": query.to_explanation(),
                        "internals": []
                    })
                else:
                    internals.append(query.to_explanation(include_internals=include_internals))
        return {
            "explanation": explanation,
            "internals": internals
        }


class AggregatingConfig(BaseConfig):
    name: str
    subtype: str
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None
    _requires_async: bool = False  # Override this in subclasses that must be async

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "AggregatingConfig":
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

    def __call__(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        """
        Synchronous interface for aggregating configs that don't require async operations.
        For configs that require async operations, use async_call() instead.
        """
        if self._requires_async:
            raise RuntimeError(
                f"Config {self.name} requires async operations. Use 'await config.async_call(...)' instead of 'config(...)'."
            )
        
        # For sync configs, we can run the async_call in a sync context since they don't actually await anything
        import asyncio
        
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, so we can't use asyncio.run
            raise RuntimeError(
                f"Cannot use sync interface for {self.name} within an async context. Use 'await config.async_call(...)' instead."
            )
        except RuntimeError:
            # No running loop, so we can create one
            return asyncio.run(self.async_call(queries, **kwargs))

    def get_scores(self, queries: list[QueryConfig | AggregatedQueryConfig]) -> list[t.Any]:
        return [query.score for query in queries]

    def get_reasonings(
        self, queries: list[QueryConfig | AggregatedQueryConfig]
    ) -> list[ReasoningsType]:
        return [query.reasoning for query in queries]

    def coerce_score_types(self, scores: list[t.Any], new_type: type[t.Any]) -> list[t.Any]:
        return [new_type(score) for score in scores]

    def check_queries(self, queries: list[QueryConfig | AggregatedQueryConfig]) -> bool:
        # recursively check that queries are of the correct type and have been answered
        # NullQueryConfig is a special case; skip it
        for query in queries:
            if isinstance(query, NullQueryConfig):
                continue
            if isinstance(query, QueryConfig):
                if not query.been_answered:
                    breakpoint()
                    assert (
                        query.been_answered
                    ), f"Query {query.instruction} must be answered before aggregation"
                if self.valid_scoring_configs is not None:
                    assert (
                        type(query.scoring_config) in self.valid_scoring_configs
                    ), f"Invalid scoring config: {type(query.scoring_config)} for query in agg config {self.name} with valid configs {self.valid_scoring_configs}"
            elif isinstance(query, AggregatedQueryConfig):
                self.check_queries(query.queries)
            else:
                raise ValueError(f"Invalid query type: {type(query)}")
        return True

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        raise NotImplementedError(f"Aggregating config {self.name} must implement async_call")


class NullAggregatingConfig(AggregatingConfig):
    """
    Placeholder for no aggregation, just returns the first query as an "aggregated" query
    """

    name: str = "null"
    subtype: str = "null"
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = None

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        return AggregatedQueryConfig(
            queries=queries,
            score=queries[0].score,
            confidence=1,
        )


class MeanAggregatingConfig(AggregatingConfig):
    name: str = "mean"
    subtype: str = "mean"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    async def async_call(
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
            reasoning=f"Mean over {scores}",
        )


class MedianAggregatingConfig(AggregatingConfig):
    name: str = "median"
    subtype: str = "median"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    async def async_call(
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
            reasoning=f"Median over {scores}",
        )


class ModeAggregatingConfig(AggregatingConfig):
    name: str = "mode"
    subtype: str = "mode"
    valid_scoring_configs: list[type[ScoringConfig]] = discrete_scoring_configs

    async def async_call(
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
            reasoning=f"Mode over {scores}",
        )


class AllAggregatingConfig(AggregatingConfig):
    name: str = "all"
    subtype: str = "all"
    valid_scoring_configs: list[type[ScoringConfig]] = [BinaryScoringConfig]

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score = all(scores)
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=1 if score else 0,
            reasoning=f"All over {scores}",
        )


class AnyAggregatingConfig(AggregatingConfig):
    name: str = "any"
    subtype: str = "any"
    valid_scoring_configs: list[type[ScoringConfig]] = [BinaryScoringConfig]

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score = any(scores)
        return AggregatedQueryConfig(
            queries=queries,
            score=score,
            confidence=1 if score else 0,
            reasoning=f"Any over {scores}",
        )


class MaxAggregatingConfig(AggregatingConfig):
    name: str = "max"
    subtype: str = "max"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score = max(scores)
        return AggregatedQueryConfig(
            queries=queries, score=score, confidence=None, reasoning=f"Max over {scores}"
        )


class MinAggregatingConfig(AggregatingConfig):
    name: str = "min"
    subtype: str = "min"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scores = self.get_scores(queries)
        score = min(scores)
        return AggregatedQueryConfig(
            queries=queries, score=score, confidence=None, reasoning=f"Min over {scores}"
        )


class WeightedAggregatingConfig(AggregatingConfig):
    name: str = "weighted"
    subtype: str = "weighted"
    valid_scoring_configs: list[type[ScoringConfig]] = continuous_scoring_configs
    weights: t.Optional[list[float]] = None

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        assert (
            "weights" in kwargs or self.weights is not None
        ), f"Weights must be provided for weighted aggregation; got kwargs: {kwargs}"
        raise NotImplementedError(f"Aggregating config {self.name} must implement async_call")


class WeightedSumAggregatingConfig(WeightedAggregatingConfig):
    name: str = "weighted_sum"
    subtype: str = "weighted_sum"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    async def async_call(
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
            reasoning=f"Weighted sum over {scores} with weights {weights}",
        )


class WeightedAverageAggregatingConfig(WeightedAggregatingConfig):
    name: str = "weighted_average"
    subtype: str = "weighted_average"
    valid_scoring_configs: list[type[ScoringConfig]] = (
        continuous_scoring_configs + discrete_scoring_configs
    )

    async def async_call(
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
            reasoning=f"Weighted average over {scores} with weights {weights}",
        )


class LLMAggregatingConfig(AggregatingConfig):
    model: str
    name: str = "llm-aggregator"
    subtype: str = "llm-aggregator"
    aggregation_prompt: str = "Aggregate the provided results into a single result."
    valid_scoring_configs: t.Optional[list[type[ScoringConfig]]] = (
        None  # indicates all scoring configs are valid
    )
    _requires_async: bool = True  # LLM calls require async

    async def async_call(
        self, queries: list[QueryConfig | AggregatedQueryConfig], **kwargs: t.Any
    ) -> AggregatedQueryConfig:
        self.check_queries(queries)
        scoring_config = FreeTextScoringConfig()
        results_str = "\n".join(
            [f"- Score: {query.score}, Reasoning: {query.reasoning}" for query in queries]
        )
        prompt = f"""
{self.aggregation_prompt}
Results:
{results_str}
{scoring_config.to_prompt()}
""".strip()
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
            reasoning=answer.reasoning,
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
    "weighted_average": WeightedAverageAggregatingConfig,
    "llm": LLMAggregatingConfig,
}

PRESET_AGGREGATOR_CONFIGS = [
    NullAggregatingConfig(),
    MeanAggregatingConfig(),
    MedianAggregatingConfig(),
    ModeAggregatingConfig(),
    AllAggregatingConfig(),
    AnyAggregatingConfig(),
    MaxAggregatingConfig(),
    MinAggregatingConfig(),
]


class AggregatorConfigCollector(BaseConfigCollector[AggregatingConfig]):
    BaseConfigType: ClassVar[type[AggregatingConfig]] = AggregatingConfig
    data_key: ClassVar[str] = "aggregator_configs"
    preset_configs: ClassVar[list[AggregatingConfig]] = PRESET_AGGREGATOR_CONFIGS
