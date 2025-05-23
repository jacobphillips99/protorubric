import asyncio

from open_rubric.rubric import Rubric

TEST_RUBRIC_PATH = "assets/example_rubrics/test_rubric.yaml"
TEST_INPUTS = """The majestic mountains stand tall against the azure sky, their snow-capped peaks piercing through wispy clouds.
Sunlight dances across ancient granite faces that have witnessed countless seasons pass. Below, dense forests of
pine and fir blanket the rugged slopes, providing shelter to diverse wildlife. Crystal clear streams, fed by
glacial melt, cascade down rocky channels, their waters eventually joining to form mighty rivers that have carved
these valleys over millennia. This timeless landscape serves as a humbling reminder of nature's raw power and beauty.""".strip()


async def run() -> None:
    rubric = Rubric.from_yaml(TEST_RUBRIC_PATH)
    results = await rubric.asolve(TEST_INPUTS)
    breakpoint()
    print(results)


if __name__ == "__main__":
    asyncio.run(run())
    breakpoint()
