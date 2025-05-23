import asyncio

from open_rubric.rubric import Rubric

rubric_path = "assets/example_rubrics/test_rubric.yaml"


async def run() -> None:
    rubric = Rubric.from_yaml(rubric_path)
    inputs = "Hello, how are you? some more random test text"
    results = await rubric.asolve(inputs)
    breakpoint()
    print(results)


if __name__ == "__main__":
    asyncio.run(run())
    breakpoint()
