from open_rubric.configs.evaluating import EvaluatorConfigCollector

TEST_EVALUATOR_PATH = "assets/examples/example_configs/my_llm_committee_evaluator.yaml"

evaluator_config_collector = EvaluatorConfigCollector.from_data_or_yaml(TEST_EVALUATOR_PATH)

print(evaluator_config_collector.configs.keys())

assert "my-llm-committee" in evaluator_config_collector.configs.keys()
committee_config = evaluator_config_collector.get_config_by_name("my-llm-committee")
assert committee_config.type == "llm-ensemble"
model_names = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
found_model_names = [model.model for model in committee_config.models]

assert set(found_model_names) == set(model_names), f"Found model names {found_model_names} do not match expected model names {model_names}"

breakpoint()