from protorubric.configs.scoring import (
    PRESET_SCORING_CONFIGS,
    ScoringConfig,
    ScoringConfigCollector,
)

TEST_SCORING_PATH = "assets/examples/example_configs/my_scoring_config.yaml"

scoring_config_collector = ScoringConfigCollector.from_data_or_yaml(TEST_SCORING_PATH)

print(scoring_config_collector.configs.keys())

preset_names = [config.name for config in PRESET_SCORING_CONFIGS]
for preset_name in preset_names:
    assert (
        preset_name in scoring_config_collector.configs.keys()
    ), f"Preset config {preset_name} not found in collector with configs {scoring_config_collector.configs.keys()}"

first_tier_names = ["tone-scoring-config", "job-quality-num-scoring-config"]
second_tier_names = ["likert"]

for first_tier_name in first_tier_names:
    assert (
        first_tier_name in scoring_config_collector.configs.keys()
    ), f"First tier config {first_tier_name} not found in collector with configs {scoring_config_collector.configs.keys()}"

for second_tier_name in second_tier_names:
    assert (
        second_tier_name in scoring_config_collector.configs.keys()
    ), f"Second tier config {second_tier_name} not found in collector with configs {scoring_config_collector.configs.keys()}"

scoring_config = scoring_config_collector.get_config_by_name("tone-scoring-config")
assert scoring_config.subtype == "categorical"
assert set(scoring_config.options) == set(["positive", "neutral", "negative"])

print("All tests passed")
breakpoint()
