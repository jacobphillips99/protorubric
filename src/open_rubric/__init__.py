"""
Open Rubric is a tool for creating and evaluating rubrics for LLM evaluation.
"""

__version__ = "0.1.0"

import os
import logging 

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)