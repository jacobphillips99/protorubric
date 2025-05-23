from open_rubric.rubric import Rubric
import asyncio
rubric_path = "example_rubrics/test_rubric.yaml"    

async def run():
    rubric = Rubric.from_yaml(rubric_path)
    inputs = "Hello, how are you? some more random test text"
    results = await rubric.asolve(inputs)
    breakpoint()

if __name__ == "__main__":
    asyncio.run(run())
    breakpoint()
