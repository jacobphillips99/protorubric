import os

# Get the project root directory dynamically; construct the relevant paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
EVAL_BASE_DIR = os.path.join(ASSETS_DIR, "eval")
VIZ_OUTPUT_DIR = os.path.join(ASSETS_DIR, "viz_outputs")
