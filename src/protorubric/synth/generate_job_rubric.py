"""
For a given job, create a test problem.
The test problem should be emblematic of a real problem the employee would solve.
Next, create a rubric for HOW the test problem should be solved.
This creates a rubric for a given job.
Next, we'll create a test solution and then autograde the rubric.

problem = LLM(job_description)
rubric = LLM(problem)
solution = LLM(problem)
results = autograde(solution, rubric)
"""

import asyncio

from protorubric.configs.aggregating import AggregatorConfigCollector
from protorubric.configs.evaluating import EvaluatorConfigCollector, ModelEvaluatorConfig
from protorubric.configs.requirement import RequirementConfig, Requirements
from protorubric.configs.scoring import ScoringConfigCollector
from protorubric.models.model import MODEL
from protorubric.models.model_types import ModelInput, ModelRequest, ModelResponse
from protorubric.rubric import Rubric

use_model_name = "gpt-4o-mini"

agg_collector = AggregatorConfigCollector.from_data_or_yaml(None)
eval_collector = EvaluatorConfigCollector.from_data_or_yaml(None)
scoring_collector = ScoringConfigCollector.from_data_or_yaml(None)

my_llm_evaluator = ModelEvaluatorConfig(
    name="my-llm-evaluator",
    type="llm",
    model=use_model_name,
)
eval_collector.add_config(my_llm_evaluator)


def setup_example() -> tuple[str, str, Rubric]:
    # lets handmake an example
    job = "financial analyst"
    problem = "is the valuation of Scale AI over 25 billion dollars?"

    default_llm_vars = {
        "evaluator": my_llm_evaluator.name,
        "aggregator": "null",
    }
    default_query_vars = {"scoring_config": "free_text"}
    # get the relevant information
    requirement_dicts = [
        {
            "name": "research",
            "query": {"instruction": "research the company", **default_query_vars},
            **default_llm_vars,
        },
        {
            "name": "ARR",
            "query": {"instruction": "determine the ARR of the company", **default_query_vars},
            **default_llm_vars,
        },
        {
            "name": "ARR multiples",
            "query": {
                "instruction": "determine ARR multiples of similar companies",
                **default_query_vars,
            },
            **default_llm_vars,
        },
    ]
    # determine the valuation of the company
    requirement_dicts.append(
        {
            "name": "valuation",
            "query": {
                "instruction": "Determine the dollar value of the company",
                **default_query_vars,
            },
            "dependency_names": [req["name"] for req in requirement_dicts],
            **default_llm_vars,
        }
    )
    # return the answer to the question
    requirement_dicts.append(
        {
            "name": "answer",
            "query": {"instruction": "return the answer to the question", **default_query_vars},
            "dependency_names": ["valuation"],
            **default_llm_vars,
        }
    )

    requirements_dict: dict[str, RequirementConfig] = {}
    for req_dict in requirement_dicts:
        requirements_dict[req_dict["name"]] = RequirementConfig.from_data(
            req_dict,
            evaluator_configs=eval_collector,
            aggregator_configs=agg_collector,
            scoring_configs=scoring_collector,
        )

    requirements = Requirements(requirements=requirements_dict)
    rubric = Rubric(requirements=requirements)
    _ = rubric.levels
    return job, problem, rubric


def get_completion(problem: str) -> str:
    req = ModelRequest(
        model=use_model_name,
        model_input=ModelInput(prompt=problem),
    )
    res: ModelResponse = asyncio.run(MODEL.agenerate(req))
    return res.texts[0]


if __name__ == "__main__":
    job, problem, rubric = setup_example()

    rubric_to_instructions = []
    i = 0
    for level in rubric.levels:
        for req in level:
            rubric_to_instructions.append(f"Step {i + 1} - {req.name}: {req.query.instruction}")
            i += 1
    rubric_to_instructions = "\n".join(rubric_to_instructions)

    problem_with_steps = f"""
You are a financial analyst.
You are given a problem and a rubric.
You need to solve the problem using the rubric.

Problem: {problem}
Rubric:
{rubric_to_instructions}

Return an answer for each step in the rubric as a JSON dictionary with the key being the step name and the value being the answer.
Do not include ```json or ``` at the beginning or end of the response.
""".strip()
    problem_completion = get_completion(problem_with_steps)

    full_convo = "\n".join(
        [
            f"User: {problem}",
            f"Assistant: {problem_completion}",
        ]
    )
    rubric_completion = rubric.solve(full_convo)
    breakpoint()
