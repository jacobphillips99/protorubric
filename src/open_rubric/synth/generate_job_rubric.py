"""
For a given job, create a test problem.
The test problem should be emblematic of a real problem the employee would solve.
Next, create a rubric for HOW the test problem should be solved.
This creates a rubric for a given job.
Next, we'll create a test solution and then autograde the rubric.

problem = LLM(job_description)
rubric = LLM(problem)
solution = LLM(problem)
autograde(solution, rubric)
"""

from open_rubric.configs.aggregating import AggregatorConfigCollector
from open_rubric.configs.evaluating import EvaluatorConfigCollector, ModelEvaluatorConfig   
from open_rubric.configs.query import QueryConfig
from open_rubric.configs.requirement import RequirementConfig, Requirements
from open_rubric.configs.scoring import ScoringConfigCollector
from open_rubric.rubric import Rubric


agg_collector = AggregatorConfigCollector.from_data_or_yaml(None)
eval_collector = EvaluatorConfigCollector.from_data_or_yaml(None)
scoring_collector = ScoringConfigCollector.from_data_or_yaml(None)

my_llm_evaluator = ModelEvaluatorConfig(
    name="my-llm-evaluator",
    type="llm",
    model="gpt-4o-mini",
)
eval_collector.add_config(my_llm_evaluator)

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
    {"name": "research", "query": {"instruction": "research the company", **default_query_vars}, **default_llm_vars},
    {
        "name": "ARR",
        "query": {"instruction": "determine the ARR of the company", **default_query_vars},
        **default_llm_vars,
    },
    {
        "name": "ARR multiples",
        "query": {"instruction": "determine ARR multiples of similar companies", **default_query_vars},
        **default_llm_vars,
    },
]
# determine the valuation of the company
requirement_dicts.append(
    {
        "name": "valuation",
        "query": {"instruction": "Determine the dollar value of the company", **default_query_vars},
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
breakpoint()