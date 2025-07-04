import asyncio
import copy
import os
import typing as t

from protorubric.constants import EVAL_BASE_DIR
from protorubric.eval.rubric_with_answers import RubricWithAnswers, generate_random_answers
from protorubric.rubric import Rubric

from .healthbench.run import make_hb_example

TEST_RUBRIC_PATH = "assets/examples/example_configs/test_rubric.yaml"
TEST_INPUT_CONVO = [
    {"role": "user", "content": "Tell me about a beautiful mountain scene."},
    {
        "role": "assistant",
        "content": """The majestic mountains stand tall against the azure sky, their snow-capped peaks piercing through wispy clouds.
Sunlight dances across ancient granite faces that have witnessed countless seasons pass. Below, dense forests of
pine and fir blanket the rugged slopes, providing shelter to diverse wildlife. Crystal clear streams, fed by
glacial melt, cascade down rocky channels, their waters eventually joining to form mighty rivers that have carved
these valleys over millennia. This timeless landscape serves as a humbling reminder of nature's raw power and beauty.""".strip(),
    },
]
TEST_INPUT_CONVO_STR = "\n".join([f"{m['role']}: {m['content']}" for m in TEST_INPUT_CONVO])

hb_rubric, hb_inputs = make_hb_example()


async def run(
    rubric_path: str,
    inputs: t.Any,
) -> None:
    rubric = Rubric.from_yaml(rubric_path)
    # outputs = await rubric.asolve(inputs)
    answers = generate_random_answers(rubric)

    rubric_with_answers_tf = RubricWithAnswers.from_rubric_and_answers(
        copy.deepcopy(rubric), answers, teacher_force=True
    )
    await rubric_with_answers_tf.asolve(inputs)
    rubric_with_answers_tf.save_pkl(os.path.join(EVAL_BASE_DIR, "teacher_force_results.pkl"))

    rubric_with_answers = RubricWithAnswers.from_rubric_and_answers(
        copy.deepcopy(rubric), answers, teacher_force=False
    )
    await rubric_with_answers.asolve(inputs)
    rubric_with_answers.save_pkl(os.path.join(EVAL_BASE_DIR, "raw_results.pkl"))


if __name__ == "__main__":
    # rubric, results = asyncio.run(run(TEST_RUBRIC_PATH, TEST_INPUTS))
    # rubric, results = asyncio.run(run(hb_rubric, hb_inputs))
    asyncio.run(run(TEST_RUBRIC_PATH, TEST_INPUT_CONVO_STR))
    # breakpoint()
