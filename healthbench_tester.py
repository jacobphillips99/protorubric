import ast
import asyncio
import json
import time

import pandas as pd

from open_rubric.aggregating import ModeAggregatingConfig, NullAggregatingConfig, WeightedAverageAggregatingConfig, WeightedSumAggregatingConfig
from open_rubric.evaluating import ModelEvaluatorConfig, PassThroughEvaluatorConfig
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest
from open_rubric.query import NULL_QUERY_CONFIG, QueryConfig
from open_rubric.requirement import RequirementConfig, Requirements
from open_rubric.rubric import Rubric
from open_rubric.scoring import name_to_scoring_config

rubric_path = "example_rubrics/test_rubric.yaml"
healthbench_sample_path = "example_rubrics/healthbench.jsonl"

with open(healthbench_sample_path, "r") as f:
    hb_samples = [json.loads(line) for line in f]

hb_df = pd.DataFrame(hb_samples)
print(f"Rubric lengths: {hb_df.rubrics.apply(lambda x: len(x)).describe()}")
points = hb_df.rubrics.apply(lambda x: [y["points"] for y in x]).explode()
print(f"Points distribution: {points.describe()}")
hb_df = hb_df.head(2)

SAMPLER_MODEL = "gpt-4o-mini"
GRADER_MODEL = "gpt-4.1"


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
    """
    final_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
        "<<rubric_item>>", json.dumps(rubric_item, indent=2)
    )
    model_request = ModelRequest(
        model=GRADER_MODEL, model_input=ModelInput(role="user", prompt=final_prompt)
    )
    response = await MODEL.agenerate(model_request)
    response_dict: dict[str, str] = json.loads(response.texts[0])
    return response_dict


def get_rubric_items(row: pd.Series) -> list[dict]:
    rubric_items = ast.literal_eval(row.rubrics)
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


async def run_row(row: pd.Series, ours: bool) -> list[dict]:
    rubric_dicts = get_rubric_items(row)
    print(f"running {len(rubric_dicts)} rubric items")
    if not ours:
        rubric_responses = await asyncio.gather(
            *[
                grade_one_rubric_item(construct_conversation_string(row.convo_with_response), item)
                for item in rubric_dicts
            ]
        )
    else:
        requirement_configs = dict()
        weights = []
        for i, rubric_dict in enumerate(rubric_dicts):
            req = hb_rubric_to_requirement(rubric_dict, str(i))
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
        rubric_responses = [results[req_name] for req_name in requirement_configs.keys()]
        rubric_scores = {k: v.score for k, v in results.items()}
        print(rubric_scores)
        print(f"Final Mode: {rubric_scores[mode_collector_name].score}")
        print(f"Final Weighted Average: {rubric_scores[weighted_average_collector_name].score}")
        print(f"Final Weighted Sum: {rubric_scores[weighted_sum_collector_name].score}")
        breakpoint()

    return rubric_responses


if __name__ == "__main__":
    # rubric = Rubric.from_yaml(rubric_path)
    # results = rubric.solve()
    # asyncio.run(run_completions(hb_df, update_in_place=True))
    # hb_df.to_csv("example_rubrics/hb_df.csv", index=False)
    hb_df = pd.read_csv("example_rubrics/hb_df.csv")
    hb_df = hb_df.head(1)

    rubric_responses = asyncio.run(run_row(hb_df.iloc[0], ours=True))
    # output = asyncio.run(hb_run_rubric_items(hb_df.iloc[0]))

    breakpoint()
