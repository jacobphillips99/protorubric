import typing as t

import yaml

from open_rubric.base import BaseConfig

"""
todo others -- likert (discrete and ordered)
"""


class ScoringConfig(BaseConfig):
    name: str = ""
    subtype: str = ""
    type: t.Literal["discrete", "continuous"]

    @classmethod
    def from_data(cls, data: dict | str, **kwargs: t.Any) -> "ScoringConfig":
        # named scoring configs, like unit_scalar, binary, etc can be accessed by name
        if isinstance(data, str):
            if data in subtype_to_scoring_configs:
                return subtype_to_scoring_configs[data](**kwargs)
            else:
                raise ValueError(f"Cannot find scoring config with name {data}")
        elif isinstance(data, dict):
            subtype = data.pop("subtype")
            name = data.pop("name", subtype)
            if subtype in subtype_to_scoring_configs:
                return subtype_to_scoring_configs[subtype](**{**data, "name": name})
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

    # TODO
    def to_prompt(self) -> str:
        return f"""
Score the response based on the following options: {self.options}
Respond with ONLY one of the options.
        """.strip()
    
    def parse_response(self, response: str) -> t.Any:
        assert response in self.options, f"Response {response} not in options {self.options}"
        return response


class BinaryScoringConfig(DiscreteScoringConfig):
    name: str = "binary"
    subtype: str = "binary"
    options: list[str] = ['true', 'false'] # JSON compatible


class CategoricalScoringConfig(DiscreteScoringConfig):
    name: str = "categorical"
    subtype: str = "categorical"
    options: list[str]


class ContinuousScoringConfig(ScoringConfig):
    name: str = "continuous"
    subtype: str = "continuous"
    type: t.Literal["continuous"] = "continuous"
    min: t.Optional[t.Union[int, float]]
    max: t.Optional[t.Union[int, float]]
    inclusive_min: bool = True
    inclusive_max: bool = True

    # TODO
    def to_prompt(self) -> str:
        scoring_range = f"{'[' if self.inclusive_min else '('}{self.min}, {self.max}{']' if self.inclusive_max else ')'}"
        return f"""Score the response by providing an answer based on the following range: {scoring_range}. Respond with ONLY that number.""".strip()

    def parse_response(self, response: str) -> t.Union[int, float]:
        score = float(response)
        assert score > self.min if self.inclusive_min else score >= self.min, f"Score {score} is less than minimum {self.min}"
        assert score < self.max if self.inclusive_max else score <= self.max, f"Score {score} is greater than maximum {self.max}"
        return score



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
subtype_to_scoring_configs = {
    **subtype_to_discrete_scoring_configs,
    **subtype_to_continuous_scoring_configs,
}
all_scoring_configs = list(subtype_to_scoring_configs.values())


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
            if isinstance(item, str) and item.endswith(".yaml"):  # recursive!
                configs.extend(ScoringConfigs.from_yaml(item, **kwargs).scoring_configs.values())
            else:
                configs.append(ScoringConfig.from_data(item, **kwargs))
        all_names = [config.name for config in configs]
        assert len(all_names) == len(set(all_names)), f"Duplicate scoring config names! {all_names}"
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
