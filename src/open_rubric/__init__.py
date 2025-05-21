"""
Open Rubric is a tool for creating and evaluating rubrics for LLM evaluation.
"""

__version__ = "0.1.0"

import logging
import os

import litellm

logger = logging.getLogger(__name__)

log_level_name = os.getenv("OPEN_RUBRIC_LOG_LEVEL", "ERROR").upper()
log_level = getattr(logging, log_level_name, logging.ERROR)
logger.setLevel(log_level)

if not logger.handlers and not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"%(asctime)s - {__name__} - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

litellm._logging._disable_debugging()
