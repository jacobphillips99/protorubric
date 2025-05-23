"""
Determines methods of scoring rubric items, like unit_scalar, binary, categorical, etc.
Defines basic interface for scoring configs and base classes for discrete and continuous scoring configs.
Configs can be defined in YAML files and loaded into the scoring configs object; recursive loading is supported.

See assets/example_rubrics/test_rubric.yaml for an example of a top-level scoring config which contains other scoring configs, like assets/example_rubrics/my_scoring_config.yaml.
"""

import json
import typing as t

import yaml
from pydantic import model_validator

from open_rubric.answers import (
    AnyAnswerConfig,
    BoolAnswerConfig,
    FloatAnswerConfig,
    StringAnswerConfig,
)
from open_rubric.base import BaseConfig


class ScoringConfig(BaseConfig):
    name: str = ""
    subtype: str = ""
    type: t.Literal["discrete", "continuous", "base"] = "base"

    @classmethod
    def from_data(cls, data: dict | str, **kwargs: t.Any) -> "ScoringConfig":
        # named scoring configs, like unit_scalar, binary, etc can be accessed by name
        if isinstance(data, str):
            if data in name_to_scoring_config:
                return name_to_scoring_config[data](**kwargs)
            else:
                raise ValueError(f"Cannot find scoring config with name {data}")
        elif isinstance(data, dict):
            subtype = data.pop("subtype")
            name = data.pop("name", subtype)
            if subtype in name_to_scoring_config:
                return name_to_scoring_config[subtype](**{**data, "name": name})
            else:
                raise ValueError(f"Cannot find scoring config with subtype {subtype}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "ScoringConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_data(config, **kwargs)

    def to_prompt(self) -> str:
        raise NotImplementedError(f"Scoring config {self.name} must implement to_prompt")


class DiscreteScoringConfig(ScoringConfig):
    name: str = "discrete"
    subtype: str = "discrete"
    type: t.Literal["discrete"] = "discrete"
    options: list[t.Any]

    @model_validator(mode="after")
    def check_options(self) -> "DiscreteScoringConfig":
        options = self.options
        assert isinstance(options, list), f"Options must be a list; got {type(options)}"
        type_of_options = type(options[0])
        assert all(
            isinstance(option, type_of_options) for option in options
        ), f"All options must be of type {type_of_options}; got {options}"
        return self

    def to_prompt(self) -> str:
        return f"""
Score the response based on the following options: {self.options} and provide a reasoning for your score.
Respond with a valid JSON object with the following keys:
- score: the score you chose from the options
- reasoning: the reasoning for your score
Use the following format:

{{
    "score": <score>,
    "reasoning": <reasoning>
}}

Do not include and other text; do not use "```json" or "```" anywhere in your response.
""".strip()

    def parse_response(self, response: str) -> AnyAnswerConfig:
        data = json.loads(response)
        type_of_options = type(self.options[0])
        parsed_score = type_of_options(data["score"])
        assert parsed_score in self.options, f"Response {response} not in options {self.options}"
        reasoning = data.get("reasoning", None)
        return AnyAnswerConfig(score=parsed_score, reasoning=reasoning)


class BinaryScoringConfig(DiscreteScoringConfig):
    name: str = "binary"
    subtype: str = "binary"
    options: list[str] = ["true", "false"]  # JSON compatible

    def parse_response(self, response: str) -> BoolAnswerConfig:
        data = json.loads(response)
        if isinstance(data["score"], bool):
            data["score"] = "true" if data["score"] else "false"
        assert data["score"] in self.options, f"Response {response} not in options {self.options}"
        reasoning = data.get("reasoning", None)
        return BoolAnswerConfig(score=(data["score"] == "true"), reasoning=reasoning)


class CategoricalScoringConfig(DiscreteScoringConfig):
    name: str = "categorical"
    subtype: str = "categorical"
    options: list[str]

    def parse_response(self, response: str) -> StringAnswerConfig:
        data = json.loads(response)
        assert data["score"] in self.options, f"Response {response} not in options {self.options}"
        reasoning = data.get("reasoning", None)
        return StringAnswerConfig(score=data["score"], reasoning=reasoning)


class ContinuousScoringConfig(ScoringConfig):
    name: str = "continuous"
    subtype: str = "continuous"
    type: t.Literal["continuous"] = "continuous"
    min: t.Optional[t.Union[int, float]]
    max: t.Optional[t.Union[int, float]]
    inclusive_min: bool = True
    inclusive_max: bool = True

    def to_prompt(self) -> str:
        scoring_range = f"{'[' if self.inclusive_min else '('}{self.min}, {self.max}{']' if self.inclusive_max else ')'}"
        return f"""
Score the response by providing an answer based on the following range: {scoring_range}. Respond with ONLY that number.
Respond with a valid JSON object with the following keys:
- score: the score you chose from the range
- reasoning: the reasoning for your score
Use the following format:

{{
    "score": <score>,
    "reasoning": <reasoning>
}}

Do not include and other text; do not use "```json" or "```" anywhere in your response.
""".strip()

    def parse_response(self, response: str) -> FloatAnswerConfig:
        data = json.loads(response)
        score_value = float(data["score"])
        if self.min is not None:
            assert (
                score_value > self.min if not self.inclusive_min else score_value >= self.min
            ), f"Score {score_value} is less than minimum {self.min}"
        if self.max is not None:
            assert (
                score_value < self.max if not self.inclusive_max else score_value <= self.max
            ), f"Score {score_value} is greater than maximum {self.max}"
        return FloatAnswerConfig(score=score_value)


class UnitScalarScoringConfig(ContinuousScoringConfig):
    name: str = "unit_scalar"
    subtype: str = "unit_scalar"
    min: t.Union[int, float] = 0
    max: t.Union[int, float] = 1


subtype_to_discrete_scoring_configs = {
    "binary": BinaryScoringConfig,
    "categorical": CategoricalScoringConfig,
    "discrete": DiscreteScoringConfig,
}
discrete_scoring_configs = list(subtype_to_discrete_scoring_configs.values())

subtype_to_continuous_scoring_configs = {
    "continuous": ContinuousScoringConfig,
    "unit_scalar": UnitScalarScoringConfig,
}
continuous_scoring_configs = list(subtype_to_continuous_scoring_configs.values())
name_to_scoring_config = {
    **subtype_to_discrete_scoring_configs,
    **subtype_to_continuous_scoring_configs,
}
preset_scoring_configs = [BinaryScoringConfig(), UnitScalarScoringConfig()]


class ScoringConfigs(BaseConfig):
    scoring_configs: dict[str, ScoringConfig]

    @classmethod
    def from_data(cls, data: list[dict | str] | dict, **kwargs: t.Any) -> "ScoringConfigs":
        if isinstance(data, dict):
            if "scoring_configs" in data:
                list_data: list[dict | str] = data["scoring_configs"]
        else:
            list_data = data
        configs: list[ScoringConfig] = []
        for item in list_data:
            if isinstance(item, str) and item.endswith(
                ".yaml"
            ):  # recursive loading of scoring configs
                configs.extend(ScoringConfigs.from_yaml(item, **kwargs).scoring_configs.values())
            else:
                configs.append(ScoringConfig.from_data(item, **kwargs))
        all_names = [config.name for config in configs]
        for present_config in preset_scoring_configs:
            if present_config.name not in all_names:
                configs.append(present_config)
        # todo: check for duplicate names
        return cls(scoring_configs={config.name: config for config in configs})

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "ScoringConfigs":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            data = data["scoring_configs"]
        return cls.from_data(data, **kwargs)

    def get_config_by_name(self, name: str) -> ScoringConfig:
        return self.scoring_configs[name]

    def get_configs_by_subtype(self, subtype: str) -> list[ScoringConfig]:
        return [config for config in self.scoring_configs.values() if config.subtype == subtype]

    def get_configs_by_type(self, type: t.Literal["discrete", "continuous"]) -> list[ScoringConfig]:
        return [config for config in self.scoring_configs.values() if config.type == type]
