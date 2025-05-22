import ast
import asyncio
import json
import time

import pandas as pd

from open_rubric.aggregators import AggregatorConfigs, NullAggregatingConfig
from open_rubric.evaluators import EvaluatorConfigs, ModelEvaluatorConfig
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest
from open_rubric.query import QueryConfig
from open_rubric.requirement import RequirementConfig, Requirements
from open_rubric.rubric import Rubric
from open_rubric.scoring import ScoringConfigs, name_to_scoring_configs

# from open_rubric.rubric import Rubric

rubric_path = "example_rubrics/test_rubric.yaml"
healthbench_sample_path = "example_rubrics/healthbench.jsonl"

with open(healthbench_sample_path, "r") as f:
    hb_samples = [json.loads(line) for line in f]

hb_df = pd.DataFrame(hb_samples)
print(f"Rubric lengths: {hb_df.rubrics.apply(lambda x: len(ast.literal_eval(x))).describe()}")
points = hb_df.rubrics.apply(lambda x: [y["points"] for y in ast.literal_eval(x)]).explode()
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
                raise ValueError(f"Could not parse conversation: {convo}")
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
    for item in rubric_items:
        item["format"] = (
            """Return JUST a valid JSON dictionary like {{"score": 0.5, "reason": 'The conversation is somewhat helpful'}}"""
        )
        item.pop("points", None)
        item.pop("tags", None)
    return rubric_items


def hb_rubric_to_requirement(rubric_item: dict, name: str) -> dict:
    scoring_config = name_to_scoring_configs["binary"]
    evaluator_config = ModelEvaluatorConfig(model=GRADER_MODEL, provider="openai")
    aggregator_config = NullAggregatingConfig()

    query_config = QueryConfig(
        instruction=rubric_item["criterion"],
        inputs=None,  # TODO,
        example=None,  # TODO,
        scoring_config=scoring_config,
    )
    requirement_config = RequirementConfig.from_data(
        name=str, query=query_config, evaluator=evaluator_config, aggregator=aggregator_config
    )
    return requirement_config


async def run_row(row: pd.Series, ours: bool = False) -> list[dict]:
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

        requirement_configs = [
            hb_rubric_to_requirement(item, str(i)) for i, item in enumerate(rubric_dicts)
        ]
        available_scoring_configs = [req.query.scoring_config for req in requirement_configs]
        scoring_configs = ScoringConfigs(
            scoring_configs={config.name: config for config in available_scoring_configs}
        )

        available_evaluator_configs = [req.evaluator for req in requirement_configs]
        evaluator_configs = EvaluatorConfigs(
            evaluators={config.name: config for config in available_evaluator_configs}
        )

        available_aggregator_configs = [req.aggregator for req in requirement_configs]
        aggregator_configs = AggregatorConfigs(
            aggregators={config.name: config for config in available_aggregator_configs}
        )

        requirements = Requirements(requirements=requirement_configs)
        rubric = Rubric(
            requirements=requirements,
            scoring_configs=scoring_configs,
            evaluators=evaluator_configs,
            aggregators=aggregator_configs,
        )
        results = rubric.solve(inputs=construct_conversation_string(row.convo_with_response))
        rubric_responses = [results[req.name] for req in requirement_configs]
    return rubric_responses


if __name__ == "__main__":
    # rubric = Rubric.from_yaml(rubric_path)
    # results = rubric.solve()
    # asyncio.run(run_completions(hb_df, update_in_place=True))
    # hb_df.to_csv("example_rubrics/hb_df.csv", index=False)
    hb_df = pd.read_csv("example_rubrics/hb_df.csv")
    # output = asyncio.run(hb_run_rubric_items(hb_df.iloc[0]))

    breakpoint()
