import ast
import asyncio
import json
import os

import pandas as pd

from protorubric.models.model import MODEL
from protorubric.models.model_types import ModelInput, ModelRequest
from protorubric.rubric import Rubric

from .healthbench_to_protorubric_utils import make_rubric_from_hb_dicts
from .setup_healthbench import check_path_or_download, get_limited_df, run_completions

HEALTHBENCH_REMOTE_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
HEALTHBENCH_LOCAL_PATH = "assets/examples/healthbench/healthbench.jsonl"
HEALTHBENCH_WITH_COMPLETIONS_LOCAL_PATH = (
    "assets/examples/healthbench/healthbench_with_completions_df.csv"
)

SAMPLER_MODEL = "gpt-4o-mini"  # the model to run for completions
GRADER_MODEL = "gpt-4o"  # the model to run for grading
SAMPLE_N = 5


def construct_conversation_string(convo: list[dict] | str) -> str:
    # helper function to convert a conversation to a string -- handles mixed formats from dataframe saving and loading
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

    convo_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in processed_convo])
    return convo_str


async def grade_one_rubric_item(convo_str: str, rubric_item: dict) -> dict:
    # simple re-implementation of the healthbench grading method from https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
    GRADER_TEMPLATE = """
    You are a helpful assistant that grades a conversation between a patient and a doctor.
    The conversation is as follows:
    ------- CONVERSATION -------
    <<conversation>>
    ------- END CONVERSATION -------

    The rubric item to grade is as follows:
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
    return response_dict


def get_rubric_items(row: pd.Series) -> list[dict]:
    rubric_items: list[dict] = ast.literal_eval(row.rubrics)
    return rubric_items


async def run_row(row: pd.Series, ours: bool, grader_model: str) -> list[tuple[dict, dict]]:
    rubric_dicts = get_rubric_items(row)
    print(f"running {len(rubric_dicts)} rubric items")
    if not ours:
        # run the simple healthbench method
        hb_results = await asyncio.gather(
            *[
                grade_one_rubric_item(construct_conversation_string(row.convo_with_response), item)
                for item in rubric_dicts
            ]
        )
        return [(rubric, result) for rubric, result in zip(rubric_dicts, hb_results)]
    else:
        # construct our rubric and solve it
        rubric = make_rubric_from_hb_dicts(rubric_dicts, grader_model=grader_model)
        results = await rubric.asolve(inputs=construct_conversation_string(row.convo_with_response))
        rubric_responses = [
            (req.model_dump(), results[req_name].model_dump())
            for req_name, req in rubric.requirements.requirements.items()
        ]
        return rubric_responses


# make example rubric from hb_df
def make_hb_example() -> tuple[Rubric, str]:
    hb_df = pd.read_csv(HEALTHBENCH_WITH_COMPLETIONS_LOCAL_PATH)
    hb_df = hb_df.head(1)
    rubric_dicts = get_rubric_items(hb_df.iloc[0])
    rubric = make_rubric_from_hb_dicts(rubric_dicts, grader_model=GRADER_MODEL)
    inputs = construct_conversation_string(hb_df.iloc[0].convo_with_response)
    return rubric, inputs


if __name__ == "__main__":
    our_method = True
    just_test = False

    check_path_or_download(HEALTHBENCH_REMOTE_PATH, HEALTHBENCH_LOCAL_PATH)

    # run completions if they don't exist
    if not os.path.exists(HEALTHBENCH_WITH_COMPLETIONS_LOCAL_PATH):
        hb_df = get_limited_df(HEALTHBENCH_LOCAL_PATH, sample_n=SAMPLE_N)
        asyncio.run(run_completions(hb_df, sampler_model=SAMPLER_MODEL, update_in_place=True))
        hb_df.to_csv(HEALTHBENCH_WITH_COMPLETIONS_LOCAL_PATH, index=False)
    else:
        hb_df = pd.read_csv(HEALTHBENCH_WITH_COMPLETIONS_LOCAL_PATH)

    hb_df.rubrics = hb_df.rubrics.apply(ast.literal_eval)
    print(f"Rubric lengths: {hb_df.rubrics.apply(lambda x: len(x)).describe()}")
    points = hb_df.rubrics.apply(lambda x: [y["points"] for y in x]).explode()
    print(f"Points distribution: {points.describe()}")

    if just_test:
        # make example rubric and solve it
        rubric, inputs = make_hb_example()
        results = rubric.solve(inputs=inputs)
    else:
        hb_df = pd.read_csv(HEALTHBENCH_WITH_COMPLETIONS_LOCAL_PATH)
        hb_df = hb_df.head(1)

        rubric_responses = asyncio.run(
            run_row(hb_df.iloc[0], ours=our_method, grader_model=GRADER_MODEL)
        )
        name_to_result = {
            req.get("name", f"rubric_item_{i}"): result
            for i, (req, result) in enumerate(rubric_responses)
        }
        name_to_scores = {k: v.get("score", None) for k, v in name_to_result.items()}
        if our_method:
            print(f"Final Mode: {name_to_result['mode_collector']['score']}")
            print(
                f"Final Weighted Average: {name_to_result['weighted_average_collector']['score']}"
            )
            print(f"Final Weighted Sum: {name_to_result['weighted_sum_collector']['score']}")
    # breakpoint()
