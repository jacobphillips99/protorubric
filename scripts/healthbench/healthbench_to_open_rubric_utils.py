import litellm

from protorubric.configs.aggregating import (
    ModeAggregatingConfig,
    NullAggregatingConfig,
    WeightedAverageAggregatingConfig,
    WeightedSumAggregatingConfig,
)
from protorubric.configs.evaluating import ModelEvaluatorConfig, PassThroughEvaluatorConfig
from protorubric.configs.query import NullQueryConfig, QueryConfig
from protorubric.configs.requirement import RequirementConfig, Requirements
from protorubric.configs.scoring import name_to_scoring_config
from protorubric.rubric import Rubric


def hb_rubric_to_requirement(rubric_item: dict, name: str, grader_model: str) -> RequirementConfig:
    scoring_config = name_to_scoring_config["binary"]()
    evaluator_config = ModelEvaluatorConfig(
        model=grader_model, provider=litellm.get_llm_provider(grader_model)[1]
    )
    aggregator_config = NullAggregatingConfig()
    query_config = QueryConfig(
        instruction=rubric_item["criterion"],
        inputs=None,
        example=None,
        scoring_config=scoring_config,
    )
    requirement_config = RequirementConfig(
        name=name,
        query=query_config,
        evaluator=evaluator_config,
        aggregator=aggregator_config,
    )
    return requirement_config


def make_rubric_from_hb_dicts(rubric_dicts: list[dict], grader_model: str) -> Rubric:
    requirement_configs = dict()
    weights = []
    for i, rubric_dict in enumerate(rubric_dicts):
        req = hb_rubric_to_requirement(
            rubric_dict, name=f"rubric_item_{i}", grader_model=grader_model
        )
        requirement_configs[req.name] = req
        weights.append(rubric_dict["points"])
    print(f"Weights: {weights}")

    # add an another Requirement that collects all of the above scores
    mode_collector_name = "mode_collector"
    mode_vote_collector = RequirementConfig(
        name=mode_collector_name,
        query=NullQueryConfig(),
        evaluator=PassThroughEvaluatorConfig(),
        aggregator=ModeAggregatingConfig(),
        dependency_names=list(requirement_configs.keys()),
    )

    weighted_average_collector_name = "weighted_average_collector"
    weighted_average_collector = RequirementConfig(
        name=weighted_average_collector_name,
        query=NullQueryConfig(),
        evaluator=PassThroughEvaluatorConfig(),
        aggregator=WeightedAverageAggregatingConfig(weights=weights),
        dependency_names=list(requirement_configs.keys()),
    )

    weighted_sum_collector_name = "weighted_sum_collector"
    weighted_sum_collector = RequirementConfig(
        name=weighted_sum_collector_name,
        query=NullQueryConfig(),
        evaluator=PassThroughEvaluatorConfig(),
        aggregator=WeightedSumAggregatingConfig(weights=weights),
        dependency_names=list(requirement_configs.keys()),
    )

    # add collectors to the requirement configs AFTER definition to avoid linkage
    requirement_configs[mode_vote_collector.name] = mode_vote_collector
    requirement_configs[weighted_average_collector.name] = weighted_average_collector
    requirement_configs[weighted_sum_collector.name] = weighted_sum_collector

    requirements = Requirements(requirements=requirement_configs)
    rubric = Rubric(requirements=requirements)
    return rubric
