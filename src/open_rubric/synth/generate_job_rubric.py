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
from open_rubric.rubric import Rubric
from open_rubric.configs.aggregating import AggregatorConfigCollector
from open_rubric.configs.evaluating import EvaluatorConfigCollector
from open_rubric.configs.query import QueryConfig
from open_rubric.configs.scoring import ScoringConfigCollector
from open_rubric.configs.requirement import RequirementConfig

agg_collector = AggregatorConfigCollector.from_data_or_yaml(None)
eval_collector = EvaluatorConfigCollector.from_data_or_yaml(None)
scoring_collector = ScoringConfigCollector.from_data_or_yaml(None)

# lets handmake an example
job = "financial analyst"
problem = "is the valuation of Scale AI over 25 billion dollars?"

default_llm_vars = {"evaluator": {"type": "llm", "model": "gpt-4o-mini"}, "aggregator": "null", "scoring_config": {"type": "free_text"}}
# get the relevant information
requirement_dicts = [
    {"name": "research", "query": {"instruction": "research the company"}, **default_llm_vars},
    {"name": "ARR", "query": {"instruction": "determine the ARR of the company"}, **default_llm_vars},
    {"name": "ARR multiples", "query": {"instruction": "determine ARR multiples of similar companies"}, **default_llm_vars},
]
# determine the valuation of the company
requirement_dicts.append({"name": "valuation", "query": {"instruction": "Determine the dollar value of the company"}, "dependency_names": [req['name'] for req in requirement_dicts]})
# return the answer to the question
requirement_dicts.append({"name": "answer", "query": {"instruction": "return the answer to the question"}, "dependency_names": ['valuation']})

requirements: dict[str, RequirementConfig] = {}
for req_dict in requirement_dicts:
    requirements[req_dict["name"]] = RequirementConfig.from_data(req_dict, evaluator_configs=eval_collector, aggregator_configs=agg_collector, scoring_configs=scoring_collector)

rubric = Rubric.from_data(requirements)






