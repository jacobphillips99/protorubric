from open_rubric.rubric import Rubric

TEST_RUBRIC_PATH = "assets/examples/example_configs/test_rubric.yaml"

rubric = Rubric.from_data_or_yaml(TEST_RUBRIC_PATH)
level_sorted_reqs = rubric.setup_graph(inputs="test inputs")
correct_level_sorted_reqs = [
    ["grammar", "tone", "helpfulness"],
    ["job_quality"],
    ["overall_quality"],
    ["summarizer"],
]

assert len(level_sorted_reqs) == len(
    correct_level_sorted_reqs
), f"Expected {len(correct_level_sorted_reqs)} levels, got {len(level_sorted_reqs)}"
for i, level in enumerate(level_sorted_reqs):
    assert set([req.name for req in level]) == set(
        correct_level_sorted_reqs[i]
    ), f"Expected {correct_level_sorted_reqs[i]} requirements in level {i}, got {level}"

breakpoint()
