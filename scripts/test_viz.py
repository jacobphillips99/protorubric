from open_rubric.rubric import Rubric
from open_rubric.viz.visualize import VIZ_OUTPUT_DIR, visualize_rubric

from .healthbench_tester import make_hb_example
from .tester import TEST_INPUTS, TEST_RUBRIC_PATH

if __name__ == "__main__":
    version = "test"
    add_inputs = True
    if version == "test":
        path = TEST_RUBRIC_PATH
        inputs = TEST_INPUTS
        rubric = Rubric.from_yaml(path)
    elif version == "hb":
        rubric, inputs = make_hb_example()
    else:
        raise ValueError(f"Invalid version: {version}")
    
    payload = {"rubric": rubric, "output_dir": VIZ_OUTPUT_DIR / version}
    if add_inputs:
        payload["inputs"] = inputs
    visualize_rubric(**payload)
