# protorubric

Open-source tools for autograding rubrics with LLMs.

## Features
- Define and evaluate rubrics for LLM-generated responses using YAML configurations.
- Support for custom scoring strategies (binary, continuous, free-text, etc.).
- Flexible aggregation methods (mean, median, mode, custom LLM-based aggregators).
- Asynchronous and synchronous evaluation workflows.
- Visualization of rubric structure and evaluation progress.
- Integration with HealthBench dataset for medical dialogue evaluation.

## Installation
<details>
<summary>Click to expand</summary>
Requirements: Python 3.10 or higher.

1. Install the package and core dependencies:
   ```bash
   git clone <repo_url>
   cd protorubric
   pip install -r requirements.txt
   pip install -e .
   ```

2. (Optional) Install visualization dependencies:
   ```bash
   pip install -r requirements-viz.txt
   ```

3. Install Graphviz for network diagrams:
   - macOS: `brew install graphviz`
   - Ubuntu/Debian: `sudo apt-get install graphviz`
</details>

## Quick Start

### Define a rubric

Create a rubric file, either in YAML or code, describing scoring, evaluators, aggregators, and requirements.
Example in [`assets/examples/example_configs/test_rubric.yaml`](https://github.com/jacobphillips99/protorubric/blob/main/assets/examples/example_configs/test_rubric.yaml) or [`assets/examples/healthbench/healthbench_to_protorubric_utils.py`](https://github.com/jacobphillips99/protorubric/blob/main/assets/examples/healthbench/healthbench_to_protorubric_utils.py).

You can also create a rubric in code:
<details>
<summary>Style-guide Rubric example</summary>
Construct a rubric for grading a response based on grammar and tone.

```python
llm_judge = ModelEvaluatorConfig(model="gpt-4o", provider="openai")

grammar_requirement = RequirementConfig(
   name="grammar",
   query=QueryConfig(instruction="Is the response grammatically correct?", scoring_config="binary"),
   evaluator=llm_judge,
)

tone_requirement = RequirementConfig(
   name="tone",
   query=QueryConfig(instruction="What tone does the response have?", scoring_config="unit_vector"),
   evaluator=llm_judge,
)

overall_score_requirement = RequirementConfig(
   name="overall_score",
   aggregator=WeightedAverageAggregatingConfig(weights=[0.9, 0.1]),
   dependency_names=["grammar", "tone"],
)
rubric = Rubric(requirements=[grammar_requirement, tone_requirement, overall_score_requirement])
```

</details>

<details>
<summary>Job Rubric example</summary>

Construct a rubric for determining whether the valuation of Scale AI is over 25 billion dollars.

```python
research_requirement = RequirementConfig(
   name="research",
   query=QueryConfig(instruction="research the company", scoring_config="free_text"),
   evaluator=llm_judge,
)
arr_requirement = RequirementConfig(
   name="arr",
   query=QueryConfig(instruction="determine the ARR of the company", scoring_config="free_text"),
   evaluator=llm_judge,
   dependency_names=["research"],
)
arr_multiples_requirement = RequirementConfig(
   name="arr_multiples",
   query=QueryConfig(instruction="determine ARR multiples of similar companies", scoring_config="free_text"),
   evaluator=llm_judge,
   dependency_names=["research"],
)
valuation_requirement = RequirementConfig(
   name="valuation",
   query=QueryConfig(instruction="determine the valuation of the company", scoring_config="free_text"),
   evaluator=llm_judge,
   dependency_names=["arr", "arr_multiples"],
)
overall_result_requirement = RequirementConfig(
   name="overall_result",
   evaluator=binary,
   dependency_names=["valuation"],
)

rubric = Rubric(requirements=[research_requirement, arr_requirement, arr_multiples_requirement, valuation_requirement, overall_result_requirement])
```
</details>

### Evaluate a rubric
Rubrics take in an a "input" object, which is typically a conversation between a user and an assistant or a blob of text. The rubric will then evaluate the input based on the requirements in the rubric. We conduct evaluation asynchronously by determining a topological ordering of the requirements and evaluating them in order. This enables us to finish the evaluation in the shortest amount of time according to the critical path. For example, the dependency graph for a given rubric may look like this:
```bash
{a: [], b: [], c:[a], d:[a,b], e:[c, d]}
```
Instead of evaluating the requirements one-by-one, we can conduct a topological level-finding sort and then evaluate the requirements asynchronously according to the critical path, such as the following:
```bash
Level 0: a, b
Level 1: c, d
Level 2: e
```
This makes evaluation significantly faster, as we can evaluate the requirements in parallel, especially for large rubrics that are much wider than they are deep.



```python
from protorubric.rubric import Rubric

# Load rubric from YAML
rubric = Rubric.from_yaml("my_rubric.yaml")

# Prepare inputs (e.g., conversation string or text)
inputs = "role: user: Hello, how are you?\nrole: assistant: I'm fine, thank you!"

# Run evaluation synchronously
results = rubric.asolve(inputs)
```


## Visualization

Generate visual representations of the rubric DAG and component usage:

```python
from protorubric.viz.visualize import visualize_rubric

# visualize and save outputs under assets/viz_outputs/
visualizer, rubric = visualize_rubric(rubric=rubric, inputs=inputs, output_dir="assets/viz_outputs")
```

See `scripts/test_viz.py` for a runnable example.

![Rubric Architecture Visualization](assets/imgs/simple_rubric_viz.png)

## Evaluation with Ground Truth

Use `RubricWithAnswers` to compare rubric evaluation against known answers, or even invoke `teacher_force=True` to force the rubric to use the known answers when considering dependent requirements. This enables us to evaluate either the full-length performance of a rubric or model or break up the evaluation into multiple parts.

```python
from protorubric.eval.rubric_with_answers import RubricWithAnswers, generate_random_answers

rubric = Rubric.from_yaml("my_rubric.yaml")
answers = generate_random_answers(rubric)

# Teacher-forced evaluation
rwa_tf = RubricWithAnswers.from_rubric_and_answers(rubric, answers, teacher_force=True)
rwa_tf.solve(inputs)

# Standard evaluation
rwa = RubricWithAnswers.from_rubric_and_answers(rubric, answers, teacher_force=False)
rwa.solve(inputs)
```

## HealthBench Integration
TODO TODO

## Project Structure

- `src/protorubric/` — core library modules.
- `assets/` — example rubrics, evaluation results, and visualization outputs.
- `scripts/` — helper scripts for testing and HealthBench integration.
- `notebooks/` — example Jupyter notebooks.
- `requirements.txt` — core dependencies.
- `requirements-viz.txt` — visualization dependencies.
- `rate_limits.yaml` — default rate limit configurations.

## Development

- Formatting: Black, isort, Ruff, and Flake8 are pre-configured.
- Type checking: MyPy (Python >=3.10).

Run pre-commit checks before committing:
```bash
pre-commit run --all-files
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open issues and submit pull requests.
