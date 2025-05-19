from open_rubric.configs.base import BaseConfig
from open_rubric.configs.scoring import ScoringConfig
import typing as t

class RequirementConfig(BaseConfig):
    name: str
    instruction: str
    example: str
    scoring_config: ScoringConfig
    score: t.Any
    dependency_names: t.Optional[list[str]] = None





