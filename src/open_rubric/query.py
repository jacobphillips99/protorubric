import typing as t

import yaml

from open_rubric.answers import AnswerConfig
from open_rubric.base import BaseConfig
from open_rubric.scoring import ScoringConfig


class QueryConfig(BaseConfig):
    instruction: t.Optional[str] = None
    inputs: t.Optional[t.Any] = None
    example: t.Optional[str] = None
    scoring_config: t.Optional[ScoringConfig] = None
    _answer: t.Optional[AnswerConfig] = None

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "QueryConfig":
        if "scoring_config" in data and isinstance(data["scoring_config"], str):
            data["scoring_config"] = kwargs["scoring_configs"].get_config_by_name(
                data["scoring_config"]
            )
        return super().from_data(data, **kwargs)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "QueryConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    @property
    def been_answered(self) -> bool:
        return self._answer is not None

    @property
    def score(self) -> t.Any:
        return self._answer.score if self._answer else None

NULL_QUERY_CONFIG = QueryConfig(
    instruction=None,
    inputs=None,
    example=None,
    scoring_config=None,
    _answer=None,
)