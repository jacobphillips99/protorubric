import ast
import asyncio
import json
import time

import pandas as pd

from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest

# from open_rubric.rubric import Rubric

rubric_path = "example_rubrics/test_rubric.yaml"
healthbench_sample_path = "example_rubrics/healthbench.jsonl"

with open(healthbench_sample_path, "r") as f:
    hb_samples = [json.loads(line) for line in f]

# hb_df = pd.DataFrame(hb_samples)
# hb_df = hb_df.head(2)

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
    processed_convo: list[dict]
    if isinstance(convo, str):
        try:
            # Try parsing as standard JSON first (e.g., from json.dumps)
            processed_convo = json.loads(convo)
        except json.JSONDecodeError:
            # If JSON parsing fails, try ast.literal_eval
            # This handles strings like "[{'role': 'user', 'content': 'Hello'}]"
            # which are Python literals but not strict JSON.
            try:
                evaluated = ast.literal_eval(convo)
                # Ensure the evaluated result is what we expect (list of dicts)
                if not isinstance(evaluated, list) or not all(
                    isinstance(item, dict) for item in evaluated
                ):
                    raise TypeError(
                        f"Evaluated string is not a list of dictionaries, got {type(evaluated)}."
                    )
                processed_convo = evaluated
            except (ValueError, SyntaxError, TypeError) as e:
                # Handle cases where neither parsing method works
                error_message = (
                    f"The 'convo_with_response' column contains a string that is neither valid JSON "
                    f"nor a parseable Python literal representing a list of dictionaries. "
                    f"Content preview: '{convo[:100]}...'. Error: {e}"
                )
                raise ValueError(error_message) from e
    elif isinstance(convo, list) and all(isinstance(item, dict) for item in convo):
        processed_convo = convo
    else:
        raise TypeError(
            f"Expected 'convo_with_response' to be a string or a list of dictionaries, "
            f"but got {type(convo)}. Content preview: '{str(convo)[:100]}...'"
        )

    # At this point, processed_convo should be a list of dicts.
    # Adding a safeguard check, though the logic above should ensure this.
    if not (
        isinstance(processed_convo, list)
        and all(isinstance(item, dict) for item in processed_convo)
    ):
        raise TypeError(
            f"Internal error: processed_convo is not a list of dictionaries after parsing. Got: {type(processed_convo)}"
        )

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


if __name__ == "__main__":
    # rubric = Rubric.from_yaml(rubric_path)
    # results = rubric.solve()
    # asyncio.run(run_completions(hb_df, update_in_place=True))
    # hb_df.to_csv("example_rubrics/hb_df.csv", index=False)
    hb_df = pd.read_csv("example_rubrics/hb_df.csv")
    rubric_item = {
        "instructions": "Is the conversation helpful?",
        "scoring": "Output a number between 0 and 1, where 0 is the worst and 1 is the best.",
        "format": """Return JUST a valid JSON dictionary like {{"score": 0.5, "reason": 'The conversation is somewhat helpful'}}""",
    }
    asyncio.run(
        grade_one_rubric_item(
            construct_conversation_string(hb_df["convo_with_response"].iloc[0]), rubric_item
        )
    )
    # breakpoint()
