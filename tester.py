from open_rubric.rubric import Rubric

rubric_path = "example_rubrics/test_rubric.yaml"

if __name__ == "__main__":
    rubric = Rubric.from_yaml(rubric_path)
    results = rubric.solve()
    # breakpoint()
