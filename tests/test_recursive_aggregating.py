import asyncio

from open_rubric.configs.aggregating import (
    AggregatedQueryConfig,
    AggregatorConfigCollector,
    LLMAggregatingConfig,
)
from open_rubric.configs.dependent_results import format_dependent_results
from open_rubric.configs.evaluating import ModelEvaluatorConfig
from open_rubric.configs.query import QueryConfig
from open_rubric.configs.scoring import ScoringConfigCollector

scoring_config_collector = ScoringConfigCollector.from_data([])
aggregator_config_collector = AggregatorConfigCollector.from_data([])

binary_scoring_config = scoring_config_collector.get_config_by_name("binary")
all_aggregator_config = aggregator_config_collector.get_config_by_name("all")


query1 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="1 + 1 = 2",
    scoring_config=binary_scoring_config,
)

query2 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="1 + 1 = 3",
    scoring_config=binary_scoring_config,
)

query3 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="2 + 1 = 3",
    scoring_config=binary_scoring_config,
)

query4 = QueryConfig(
    instruction="Is the answer correct?",
    inputs="2 + 2 = 4",
    scoring_config=binary_scoring_config,
)
queries = [query1, query2, query3, query4]


all_aggregator_config = aggregator_config_collector.get_config_by_name("all")
llm_aggregator_config = LLMAggregatingConfig(
    model="gpt-4o-mini",
    aggregation_prompt="Aggregate the provided results into a single result. Use the binary `all` to combine the results, meaning that all results must be true for the final answer to be true.",
)
null_aggregator_config = aggregator_config_collector.get_config_by_name("null")


model_evaluator_config = ModelEvaluatorConfig(
    model="gpt-4o-mini",
    provider="openai",
    n_samples=1,
)


async def build_nested_aqc(
    queries: list[QueryConfig | AggregatedQueryConfig],
) -> dict[str, AggregatedQueryConfig]:
    # answer all queries together, then build an a nested AQC by iterating over each step
    answered_queries_list = [await model_evaluator_config.async_call(query) for query in queries]
    aqcs = [
        await null_aggregator_config.async_call(answered_queries)
        for answered_queries in answered_queries_list
    ]
    nested = aqcs[0]
    # make a nested AQC by aggregating over all the results
    for i in range(1, len(aqcs)):
        nested = await all_aggregator_config.async_call([nested, aqcs[i]])
    return dict(nested=nested)


async def build_dependent_results(
    queries: list[QueryConfig | AggregatedQueryConfig],
) -> dict[str, AggregatedQueryConfig]:
    # build a dependent results object by iteratively answering each query as dependent on the previous answers
    dependent_results: dict[str, AggregatedQueryConfig] = {}

    first_aqc = await null_aggregator_config.async_call(
        await model_evaluator_config.async_call(queries[0])
    )
    dependent_results["next_0"] = first_aqc
    for i in range(1, len(queries)):
        next_aqc = await null_aggregator_config.async_call(
            await model_evaluator_config.async_call(queries[i], dependent_results)
        )
        dependent_results[f"next_{i}"] = next_aqc
    return dependent_results


if __name__ == "__main__":
    # aqc = asyncio.run(build_nested_aqc(queries))
    results_dict = asyncio.run(build_dependent_results(queries))
    text = format_dependent_results(results_dict, include_internals=True)
    print(text)
    breakpoint()
