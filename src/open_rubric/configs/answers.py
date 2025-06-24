import typing as t

from protorubric.configs.base import BaseConfig

ScoreType = t.TypeVar("ScoreType")


class AnswerConfig(BaseConfig, t.Generic[ScoreType]):
    score: ScoreType
    reasoning: t.Optional[str] = None


# provide a concrete answer config for requests
class AnyAnswerConfig(BaseConfig):
    score: t.Union[str, bool, int, float]
    reasoning: t.Optional[str] = None


# Create type aliases for the answer configs
StringAnswerConfig = AnswerConfig[str]
BoolAnswerConfig = AnswerConfig[bool]
IntAnswerConfig = AnswerConfig[int]
FloatAnswerConfig = AnswerConfig[float]

ANSWER_TYPE_TO_PYTHON_TYPE: dict[t.Type[AnswerConfig], t.Type[t.Any]] = {
    StringAnswerConfig: str,
    BoolAnswerConfig: bool,
    IntAnswerConfig: int,
    FloatAnswerConfig: float,
}
