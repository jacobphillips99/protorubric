# open-rubric

Open-source tools for autograding rubrics with LLMs.

## Features
- Define and evaluate rubrics for LLM-generated responses using YAML configurations.
- Support for custom scoring strategies (binary, continuous, free-text, etc.).
- Flexible aggregation methods (mean, median, mode, custom LLM-based aggregators).
- Asynchronous and synchronous evaluation workflows.
- Visualization of rubric structure and evaluation progress.
- Integration with HealthBench dataset for medical dialogue evaluation.

## Installation

Requirements: Python 3.10 or higher.

1. Install the package and core dependencies:
   ```bash
   git clone <repo_url>
   cd open-rubric
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

## Quick Start

### Define a rubric

Create a rubric file, either in YAML or code, describing scoring, evaluators, aggregators, and requirements.
Example in `assets/examples/example_configs/test_rubric.yaml` or `assets/examples/healthbench/healthbench_to_open_rubric_utils.py`

### Evaluate a rubric

```python
from open_rubric.rubric import Rubric

# Load rubric from YAML
rubric = Rubric.from_yaml("assets/examples/example_configs/test_rubric.yaml")

# Prepare inputs (e.g., conversation string or text)
inputs = "role: user: Hello, how are you?\nrole: assistant: I'm fine, thank you!"

# Run evaluation synchronously
results = rubric.solve(inputs)

print(results)
```

For asynchronous evaluation:
```python
import asyncio
from open_rubric.rubric import Rubric

rubric = Rubric.from_yaml("my_rubric.yaml")
results = asyncio.run(rubric.asolve(inputs))
```

## Visualization

Generate visual representations of the rubric DAG and component usage:

```python
from open_rubric.viz.visualize import visualize_rubric

# visualize and save outputs under assets/viz_outputs/
visualizer, rubric = visualize_rubric(rubric=rubric, inputs=inputs, output_dir="assets/viz_outputs")
```

See `scripts/test_viz.py` for a runnable example.

## Evaluation with Ground Truth

Use `RubricWithAnswers` to compare rubric evaluation against known answers, or even invoke `teacher_force=True` to force the rubric to use the known answers when considering dependent requirements:

```python
from open_rubric.eval.rubric_with_answers import RubricWithAnswers, generate_test_answers

rubric = Rubric.from_yaml("my_rubric.yaml")
answers = generate_test_answers(rubric)

# Teacher-forced evaluation
rwa_tf = RubricWithAnswers.from_rubric_and_answers(rubric, answers, teacher_force=True)
rwa_tf.solve(inputs)

# Standard evaluation
rwa = RubricWithAnswers.from_rubric_and_answers(rubric, answers, teacher_force=False)
rwa.solve(inputs)
```

## HealthBench Integration

Convert and evaluate HealthBench medical dialogue tasks:

```bash
python scripts/healthbench/run.py
```

Or use `make_hb_example()` in your code to generate a sample rubric and inputs:

```python
from scripts.healthbench.run import make_hb_example

rubric, inputs = make_hb_example()
results = rubric.solve(inputs)
```

## Project Structure

- `src/open_rubric/` — core library modules.
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
