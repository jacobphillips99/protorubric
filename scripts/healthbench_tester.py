import ast
import asyncio
import json
import time

import pandas as pd

from open_rubric.configs.aggregating import (
    ModeAggregatingConfig,
    NullAggregatingConfig,
    WeightedAverageAggregatingConfig,
    WeightedSumAggregatingConfig,
)
from open_rubric.configs.evaluating import ModelEvaluatorConfig, PassThroughEvaluatorConfig
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest
from open_rubric.configs.query import NULL_QUERY_CONFIG, QueryConfig
from open_rubric.configs.requirement import RequirementConfig, Requirements
from open_rubric.rubric import Rubric
from open_rubric.configs.scoring import name_to_scoring_config

rubric_path = "assets/example_rubrics/test_rubric.yaml"
healthbench_path = "assets/healthbench.jsonl"
healthbench_with_completions_path = "assets/hb_df.csv"

SAMPLER_MODEL = "gpt-4o-mini"  # the model to run for completions
# GRADER_MODEL = "gpt-4.1" # the model to run for grading
GRADER_MODEL = "gpt-4o"  # the model to run for grading


def load_healthbench_df() -> pd.DataFrame:
    with open(healthbench_path, "r") as f:
        hb_samples = [json.loads(line) for line in f]

    hb_df = pd.DataFrame(hb_samples)
    print(f"Rubric lengths: {hb_df.rubrics.apply(lambda x: len(x)).describe()}")
    points = hb_df.rubrics.apply(lambda x: [y["points"] for y in x]).explode()
    print(f"Points distribution: {points.describe()}")
    hb_df = hb_df.head(2)

    return hb_df


healthbench_df = load_healthbench_df()


def get_final_completion(prompt_value: list[dict[str, str]]) -> str:
    return prompt_value[-1]["content"]


# need to add a model completion
async def make_convo_with_response(row: pd.Series) -> list[dict[str, str]]:
    prompt: list[dict[str, str]] = row.prompt
    prepared_messages: list[ModelInput] = []
    for msg in prompt:
        prepared_messages.append(ModelInput(role=msg["role"], prompt=msg["content"]))
    model_request = ModelRequest(model=SAMPLER_MODEL, prepared_messages=prepared_messages)
    response = await MODEL.agenerate(model_request)
    response_text = response.texts[0]
    convo_with_response = prompt + [{"role": "assistant", "content": response_text}]
    return convo_with_response


async def run_completions(df: pd.DataFrame, update_in_place: bool = True) -> pd.DataFrame:
    print(f"Running {len(df)} completions")
    start_time = time.time()
    convos = await asyncio.gather(*[make_convo_with_response(row) for _, row in df.iterrows()])
    end_time = time.time()
    print(f"Completed {len(convos)} completions in {round(end_time - start_time, 2)} seconds")
    if update_in_place:
        df["convo_with_response"] = convos
        return df
    else:
        return pd.DataFrame({"convo_with_response": convos})


def construct_conversation_string(convo: list[dict] | str) -> str:
    if isinstance(convo, str):
        try:
            processed_convo = json.loads(convo)
        except json.JSONDecodeError:
            try:
                processed_convo = ast.literal_eval(convo)
            except Exception as e:
                raise ValueError(f"Could not parse conversation:{e} {convo}")
    if not isinstance(processed_convo, list):
        raise ValueError(f"Conversation is not a list: {processed_convo}")

    convo_str = "\n\n".join(
        [
            f"{m.get('role', 'UnknownRole')}: {m.get('content', 'NoContent')}"
            for m in processed_convo
        ]
    )
    return convo_str


async def grade_one_rubric_item(convo_str: str, rubric_item: dict) -> dict:
    GRADER_TEMPLATE = """
    You are a helpful assistant that grades a conversation between a patient and a doctor.
    The conversation is as follows:
    ------- CONVERSATION -------
    <<conversation>>
    ------- END CONVERSATION -------
    \n\nThe rubric item to grade is as follows:
    ------- RUBRIC ITEM -------
    <<rubric_item>>
    ------- END RUBRIC ITEM -------
    Respond with a JSON object with the following fields:
    - score: the score for the rubric item
    - reasoning: the reasoning for the score

    Example response:
    {
        "score": "true",
        "reasoning": "The patient was able to explain their symptoms and the doctor was able to provide a diagnosis."
    }

    Do NOT include any other text; do not use ```json or ``` or anything else.
    """
    final_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
        "<<rubric_item>>", json.dumps(rubric_item, indent=2)
    )
    model_request = ModelRequest(
        model=GRADER_MODEL, model_input=ModelInput(role="user", prompt=final_prompt)
    )
    response = await MODEL.agenerate(model_request)
    try:
        response_dict: dict[str, str] = json.loads(response.texts[0])
    except Exception as e:
        print(f"{e}, {response.texts[0]}")
        breakpoint()
    return response_dict


def get_rubric_items(row: pd.Series) -> list[dict]:
    rubric_items: list[dict] = ast.literal_eval(row.rubrics)
    return rubric_items


def hb_rubric_to_requirement(rubric_item: dict, name: str) -> RequirementConfig:
    scoring_config = name_to_scoring_config["binary"]()
    evaluator_config = ModelEvaluatorConfig(model=GRADER_MODEL, provider="openai")
    aggregator_config = NullAggregatingConfig()
    query_config = QueryConfig(
        instruction=rubric_item["criterion"],
        inputs=None,  # added later in .solve()
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


async def run_row(row: pd.Series, ours: bool) -> list[tuple[dict, dict]]:
    rubric_dicts = get_rubric_items(row)
    print(f"running {len(rubric_dicts)} rubric items")
    if not ours:
        results = await asyncio.gather(
            *[
                grade_one_rubric_item(construct_conversation_string(row.convo_with_response), item)
                for item in rubric_dicts
            ]
        )
        rubric_responses = [(rubric, result) for rubric, result in zip(rubric_dicts, results)]

    else:
        requirement_configs = dict()
        weights = []
        for i, rubric_dict in enumerate(rubric_dicts):
            req = hb_rubric_to_requirement(rubric_dict, name=f"rubric_item_{i}")
            requirement_configs[req.name] = req
            weights.append(rubric_dict["points"])
        print(f"Weights: {weights}")

        # add an another Requirement that collects all of the above scores
        mode_collector_name = "mode_collector"
        mode_vote_collector = RequirementConfig(
            name=mode_collector_name,
            query=NULL_QUERY_CONFIG,
            evaluator=PassThroughEvaluatorConfig(),
            aggregator=ModeAggregatingConfig(),
            dependency_names=list(requirement_configs.keys()),
        )

        weighted_average_collector_name = "weighted_average_collector"
        weighted_average_collector = RequirementConfig(
            name=weighted_average_collector_name,
            query=NULL_QUERY_CONFIG,
            evaluator=PassThroughEvaluatorConfig(),
            aggregator=WeightedAverageAggregatingConfig(weights=weights),
            dependency_names=list(requirement_configs.keys()),
        )

        weighted_sum_collector_name = "weighted_sum_collector"
        weighted_sum_collector = RequirementConfig(
            name=weighted_sum_collector_name,
            query=NULL_QUERY_CONFIG,
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

        results = await rubric.asolve(inputs=construct_conversation_string(row.convo_with_response))
        rubric_responses = [
            (req.model_dump(), results[req_name].model_dump())
            for req_name, req in requirement_configs.items()
        ]

    return rubric_responses


if __name__ == "__main__":
    # rubric = Rubric.from_yaml(rubric_path)
    # results = rubric.solve()
    # asyncio.run(run_completions(hb_df, update_in_place=True))
    # hb_df.to_csv(healthbench_with_completions_path, index=False)
    hb_df = pd.read_csv(healthbench_with_completions_path)
    hb_df = hb_df.head(1)
    OUR_METHOD = True

    rubric_responses = asyncio.run(run_row(hb_df.iloc[0], ours=OUR_METHOD))
    name_to_result = {
        req.get("name", f"rubric_item_{i}"): result
        for i, (req, result) in enumerate(rubric_responses)
    }
    breakpoint()
    name_to_scores = {k: v.get("score", None) for k, v in name_to_result.items()}
    if OUR_METHOD:
        print(f"Final Mode: {name_to_result['mode_collector'].score}")
        print(f"Final Weighted Average: {name_to_result['weighted_average_collector'].score}")
        print(f"Final Weighted Sum: {name_to_result['weighted_sum_collector'].score}")
    breakpoint()

    breakpoint()
