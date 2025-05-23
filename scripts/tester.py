import asyncio

from open_rubric.configs.aggregating import AggregatedQueryConfig
from open_rubric.eval.rubric_with_answers import RubricWithAnswers, generate_test_answers
from open_rubric.rubric import Rubric

TEST_RUBRIC_PATH = "assets/example_rubrics/test_rubric.yaml"
TEST_INPUTS = """The majestic mountains stand tall against the azure sky, their snow-capped peaks piercing through wispy clouds.
Sunlight dances across ancient granite faces that have witnessed countless seasons pass. Below, dense forests of
pine and fir blanket the rugged slopes, providing shelter to diverse wildlife. Crystal clear streams, fed by
glacial melt, cascade down rocky channels, their waters eventually joining to form mighty rivers that have carved
these valleys over millennia. This timeless landscape serves as a humbling reminder of nature's raw power and beauty.""".strip()

from .healthbench_tester import make_hb_example
hb_rubric, hb_inputs = make_hb_example()

async def run(rubric: Rubric | str, inputs: str) -> tuple[Rubric, list[AggregatedQueryConfig]]:
    rubric = Rubric.from_yaml(rubric) if isinstance(rubric, str) else rubric
    results = await rubric.asolve(inputs)
    return rubric, results


if __name__ == "__main__":
    # rubric, results = asyncio.run(run(TEST_RUBRIC_PATH, TEST_INPUTS))
    rubric, results = asyncio.run(run(hb_rubric, hb_inputs))
    answers = generate_test_answers(rubric)
    rubric_with_answers = RubricWithAnswers.from_rubric_and_answers(rubric, answers)
    rubric_with_answers.run_metrics()
    breakpoint()
