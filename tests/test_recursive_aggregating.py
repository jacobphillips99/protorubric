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

all_aggregator = aggregator_config_collector.get_config_by_name("all")
llm_aggregator_config = LLMAggregatingConfig(
    model="gpt-4o-mini",
    aggregation_prompt="Aggregate the provided results into a single result. Use the binary `all` to combine the results, meaning that all results must be true for the final answer to be true.",
)

async def main() -> dict[str, AggregatedQueryConfig]:
    return {"text": await llm_aggregator_config.async_call(queries=[agg1, agg2]), "all": await all_aggregator.async_call(queries=[agg1, agg2])}


if __name__ == "__main__":
    recursively_aggregated = asyncio.run(main())

    from open_rubric.configs.dependent_results import format_dependent_results
    text = format_dependent_results(recursively_aggregated, include_internals=True)
    breakpoint()
