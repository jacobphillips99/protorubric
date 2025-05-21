import typing as t

from open_rubric.base import BaseConfig

ScoreType = t.TypeVar("ScoreType")


class AnswerConfig(BaseConfig, t.Generic[ScoreType]):
    score: ScoreType
    reasoning: t.Optional[str] = None


# Create type aliases for the answer configs
AnyAnswerConfig = AnswerConfig[t.Any]
StringAnswerConfig = AnswerConfig[str]
BoolAnswerConfig = AnswerConfig[bool]
IntAnswerConfig = AnswerConfig[int]
FloatAnswerConfig = AnswerConfig[float]
NumberAnswerConfig = t.Union[AnswerConfig[int], AnswerConfig[float]]
