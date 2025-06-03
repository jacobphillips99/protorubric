from open_rubric.configs.aggregating import PRESET_AGGREGATOR_CONFIGS, AggregatorConfigCollector

TEST_AGGREGATOR_PATH = "assets/examples/example_configs/my_aggregator_config.yaml"

aggregating_config_collector = AggregatorConfigCollector.from_data_or_yaml(TEST_AGGREGATOR_PATH)

print(aggregating_config_collector.configs.keys())

preset_names = [config.name for config in PRESET_AGGREGATOR_CONFIGS]
for preset_name in preset_names:
    assert (
        preset_name in aggregating_config_collector.configs.keys()
    ), f"Preset config {preset_name} not found in collector with configs {aggregating_config_collector.configs.keys()}"

assert "ninety-ten-aggregator" in aggregating_config_collector.configs.keys()
ninety_ten_aggregator = aggregating_config_collector.get_config_by_name("ninety-ten-aggregator")
assert ninety_ten_aggregator.subtype == "weighted_sum"
assert ninety_ten_aggregator.weights == [0.9, 0.1]
breakpoint()
