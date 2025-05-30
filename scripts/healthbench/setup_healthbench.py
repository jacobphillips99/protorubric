"""
Utility functions for loading and preparing the healthbench dataset.

Includes setup for running completions and converting the healthbench dataset to a pandas dataframe.
"""

import asyncio
import json
import time
import typing as t

import pandas as pd

from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest


def get_healthbench_df(path: str, sample_n: t.Optional[int] = None) -> pd.DataFrame:
    # load the healthbench jsonl file and convert to a pandas dataframe
    with open(path, "r") as f:
        lines = f if sample_n is None else (next(f) for _ in range(sample_n))
        hb_samples = [json.loads(line) for line in lines]

    hb_df = pd.DataFrame(hb_samples)
    print(f"Rubric lengths: {hb_df.rubrics.apply(lambda x: len(x)).describe()}")
    points = hb_df.rubrics.apply(lambda x: [y["points"] for y in x]).explode()
    print(f"Points distribution: {points.describe()}")
    return hb_df


async def make_convo_with_response(row: pd.Series, sampler_model: str) -> list[dict[str, str]]:
    # construct a new response and append to the conversation
    prompt: list[dict[str, str]] = row.prompt
    prepared_messages: list[ModelInput] = []
    for msg in prompt:
        prepared_messages.append(ModelInput(role=msg["role"], prompt=msg["content"]))
    model_request = ModelRequest(model=sampler_model, prepared_messages=prepared_messages)
    response = await MODEL.agenerate(model_request)
    response_text = response.texts[0]
    convo_with_response = prompt + [{"role": "assistant", "content": response_text}]
    return convo_with_response


async def run_completions(
    df: pd.DataFrame, sampler_model: str, update_in_place: bool = True
) -> pd.DataFrame:
    # run completions for each row in the dataframe -- rate limiting handled by llm_rate_limiter package inside MODEL
    print(f"Running {len(df)} completions")
    start_time = time.time()
    convos = await asyncio.gather(
        *[make_convo_with_response(row, sampler_model=sampler_model) for _, row in df.iterrows()]
    )
    end_time = time.time()
    print(f"Completed {len(convos)} completions in {round(end_time - start_time, 2)} seconds")
    if update_in_place:
        df["convo_with_response"] = convos
        return df
    else:
        return pd.DataFrame({"convo_with_response": convos})
