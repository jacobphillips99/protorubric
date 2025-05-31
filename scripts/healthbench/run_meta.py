import asyncio

import pandas as pd

from open_rubric.configs.aggregating import AggregatedQueryConfig
from open_rubric.rubric import Rubric
from scripts.healthbench.run import HEALTHBENCH_LOCAL_PATH, HEALTHBENCH_REMOTE_PATH

from .setup_healthbench import check_path_or_download, get_limited_df
from .setup_meta_healthbench import make_convo_with_completion, meta_hb_rubric_to_open_rubric

HEALTHBENCH_META_REMOTE_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl"
HEALTHBENCH_META_LOCAL_PATH = "assets/examples/healthbench/healthbench_meta.jsonl"

GRADER_MODEL = "gpt-4.1"


async def run_row(row: pd.Series) -> tuple[Rubric, dict[str, AggregatedQueryConfig]]:
    rubric = await meta_hb_rubric_to_open_rubric(
        meta_row.rubric, rubric_constructor_model=GRADER_MODEL, grader_model=GRADER_MODEL
    )
    convo_with_response = make_convo_with_completion(row)
    results = await rubric.asolve(convo_with_response)
    return rubric, results


if __name__ == "__main__":
    # download the healthbench file
    check_path_or_download(HEALTHBENCH_REMOTE_PATH, HEALTHBENCH_LOCAL_PATH)
    # download the healthbench meta file
    check_path_or_download(
        HEALTHBENCH_META_REMOTE_PATH,
        HEALTHBENCH_META_LOCAL_PATH,
    )
    hb_df = get_limited_df(HEALTHBENCH_LOCAL_PATH)
    hb_meta_df = get_limited_df(HEALTHBENCH_META_LOCAL_PATH)
    # binary labels: list of [bool]

    meta_row = hb_meta_df.iloc[1]
    rubric, results = asyncio.run(run_row(meta_row))
    print("our results: ")
    for k in [k for k in results.keys() if "answer" in k]:
        print(f"{k}: {results[k].score}, {results[k].aggregated_reasoning}")
    row_answers = meta_row.binary_labels
    print(f"\nhealthbench results: {row_answers}")
    # breakpoint()
