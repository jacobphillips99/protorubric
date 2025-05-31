import os

from open_rubric.constants import VIZ_OUTPUT_DIR
from open_rubric.rubric import Rubric
from open_rubric.viz.visualize import visualize_rubric

from .healthbench.run import make_hb_example
from .tester import TEST_INPUT_CONVO_STR, TEST_RUBRIC_PATH

if __name__ == "__main__":
    version = "test"
    add_inputs = True
    if version == "test":
        path = TEST_RUBRIC_PATH
        inputs = TEST_INPUT_CONVO_STR
        rubric = Rubric.from_yaml(path)
    elif version == "hb":
        rubric, inputs = make_hb_example()
    else:
        raise ValueError(f"Invalid version: {version}")

    payload = {"rubric": rubric, "output_dir": os.path.join(VIZ_OUTPUT_DIR, version)}
    if add_inputs:
        payload["inputs"] = inputs
    visualizer, rubric = visualize_rubric(**payload)
    # breakpoint()
