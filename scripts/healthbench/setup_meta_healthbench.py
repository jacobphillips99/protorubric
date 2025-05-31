import json

import litellm
import pandas as pd

from open_rubric.configs.aggregating import AllAggregatingConfig, LLMAggregatingConfig
from open_rubric.configs.evaluating import ModelEvaluatorConfig, PassThroughEvaluatorConfig
from open_rubric.configs.query import NULL_QUERY_CONFIG, QueryConfig
from open_rubric.configs.requirement import RequirementConfig, Requirements
from open_rubric.configs.scoring import BinaryScoringConfig
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelRequest
from open_rubric.rubric import Rubric


async def meta_hb_rubric_to_open_rubric(
    meta_hb_rubric: str, rubric_constructor_model: str, grader_model: str
) -> Rubric:
    # use an LLM to convert the paragraph-like meta healthbench rubric into a set of requirements for open-rubric
    prompt = f"""
Breakdown the following instruction set into a list of requirements.
The requirements should have a name and be phrased as questions that can be answered with a binary yes/no.
Respond with a JSON object representing a python dictionary.
The key of the dictionary should be the name; the value should be the question to be answered.
Do not include any other text; do not use ```json or ``` or anything else.

Here is the instruction set:
{meta_hb_rubric}
"""
    model_input = ModelInput(role="user", prompt=prompt)
    req = ModelRequest(model=rubric_constructor_model, model_input=model_input)
    response = await MODEL.agenerate(req)
    # construct the dictionary of the broken down meta rubric
    names_to_questions = json.loads(response.texts[0])

    # build our open-rubric requirements
    # the meta healthbench rubrics are aggreagated by "all" and are all "binary" scoring configs
    reqs: dict[str, RequirementConfig] = dict()
    evaluator_config = ModelEvaluatorConfig(
        model=grader_model, provider=litellm.get_llm_provider(grader_model)[1]
    )
    scoring_config = BinaryScoringConfig()
    agg_config = AllAggregatingConfig()
    for name, question in names_to_questions.items():
        reqs[name] = RequirementConfig(
            name=name,
            query=QueryConfig(instruction=question, scoring_config=scoring_config),
            evaluator=evaluator_config,
            aggregator=agg_config,
        )

    reqs["answer_bool"] = RequirementConfig(
        name="answer_bool",
        query=NULL_QUERY_CONFIG,
        dependency_names=list(names_to_questions.keys()),
        evaluator=PassThroughEvaluatorConfig(),
        aggregator=agg_config,
    )
    text_agg_prompt = "Summarize the available information and combine it into a single answer using the boolean 'all' function to combine all the boolean answers. Only answer 'true' if ALL provided information is scored as true. If any of the information is scored as false, answer 'false'."
    reqs["answer_text"] = RequirementConfig(
        name="answer_text",
        query=NULL_QUERY_CONFIG,
        dependency_names=list(names_to_questions.keys()),
        evaluator=PassThroughEvaluatorConfig(),
        aggregator=LLMAggregatingConfig(model=grader_model, aggregation_prompt=text_agg_prompt),
    )
    return Rubric(requirements=Requirements(requirements=reqs))


def make_convo_with_completion(row: pd.Series) -> str:
    # prompt: list[dict[str, str]] = row.prompt
    # prepared_messages: list[ModelInput] = []
    # for msg in prompt:
    #     prepared_messages.append(ModelInput(role=msg["role"], prompt=msg["content"]))
    # # add the completion from the row
    # prepared_messages.append(ModelInput(role="assistant", prompt=row.completion))
    # # healthbench actually submits all of these as one message in the "user"
    # # https://github.com/openai/simple-evals/blob/main/healthbench_meta_eval.py#L86
    # return prepared_messages
    row_prompt: list[dict[str, str]] = row.prompt
    row_prompt.append({"role": "assistant", "content": row.completion})
    prompt_str = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in row_prompt])
    return prompt_str
