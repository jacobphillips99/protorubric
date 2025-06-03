import asyncio

from open_rubric.configs.aggregating import (
    AggregatedQueryConfig,
    AggregatorConfigCollector,
    LLMAggregatingConfig,
)
from open_rubric.configs.answers import BoolAnswerConfig
from open_rubric.configs.query import QueryConfig
from open_rubric.configs.scoring import ScoringConfig, ScoringConfigCollector

scoring_config_collector = ScoringConfigCollector.from_data([])
aggregator_config_collector = AggregatorConfigCollector.from_data([])

binary_scoring_config = scoring_config_collector.get_config_by_name("binary")
all_aggregator_config = aggregator_config_collector.get_config_by_name("all")


query1 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="1 + 1 = 2",
    scoring_config=binary_scoring_config,
    answer=BoolAnswerConfig(score=True, reasoning="The math is correct q1."),
)

query2 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="1 + 1 = 3",
    scoring_config=binary_scoring_config,
    answer=BoolAnswerConfig(score=False, reasoning="The math is incorrect q2."),
)

query3 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="2 + 1 = 3",
    scoring_config=binary_scoring_config,
    answer=BoolAnswerConfig(score=True, reasoning="The math is correct q3."),
)

query4 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="2 + 2 = 4",
    scoring_config=binary_scoring_config,
    answer=BoolAnswerConfig(score=True, reasoning="The math is correct q4."),
)

agg1 = AggregatedQueryConfig(
    queries=[query1, query2],
    score=False,
    reasoning="The math is incorrect agg1.",
    confidence=0.5,
)
agg2 = AggregatedQueryConfig(
    queries=[query3, query4],
    score=True,
    reasoning="The math is correct agg2.",
    confidence=0.5,
)

llm_aggregator_config = LLMAggregatingConfig(
    model="gpt-4o-mini",
    aggregation_prompt="Aggregate the provided results into a single result.",
)


async def main() -> AggregatedQueryConfig:
    return await llm_aggregator_config.async_call(queries=[agg1, agg2])


if __name__ == "__main__":
    res = asyncio.run(main())
    breakpoint()
