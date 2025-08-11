import typing as t

import yaml

from protorubric.configs.answers import AnswerConfig
from protorubric.configs.base import BaseConfig
from protorubric.configs.scoring import ScoringConfig


class QueryConfig(BaseConfig):
    instruction: str
    inputs: t.Optional[t.Any] = None
    example: t.Optional[str] = None
    scoring_config: ScoringConfig
    answer: t.Optional[AnswerConfig] = None

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "QueryConfig":
        if data.get("instruction") is None:
            # null query
            return NullQueryConfig(
                scoring_config=data.get("scoring_config", ScoringConfig()),
                answer=data.get("answer", None),
            )
        return super().from_data(data, **kwargs)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "QueryConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    @property
    def been_answered(self) -> bool:
        return self.answer is not None

    @property
    def score(self) -> t.Any:
        return self.answer.score if self.answer else None

    @property
    def reasoning(self) -> str | None:
        return self.answer.reasoning if self.answer else None

    def to_explanation(self, **kwargs: t.Any) -> str:
        assert self.been_answered, "Cannot generate explanation for unanswered query!"
        scoring_method_description = self.scoring_config.to_description()
        return f"Selected `{self.score}` with reasoning: {self.reasoning} using {scoring_method_description}."


class NullQueryConfig(QueryConfig):
    """
    Special null query configuration that skips answering the query and
    optionally includes a provided score and answer.
    """

    def __init__(self, **kwargs: t.Any) -> None:
        super().__init__(
            instruction="None",
            inputs=None,
            example=None,
            scoring_config=kwargs.get("scoring_config", ScoringConfig()),
            answer=kwargs.get("answer", None),
            **{k: v for k, v in kwargs.items() if k not in ["scoring_config", "answer"]},
        )
