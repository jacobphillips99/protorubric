import typing as t

import yaml

from open_rubric.configs.base import BaseConfig


class ScoringConfig(BaseConfig):
    name: str = ""
    subtype: str = ""
    type: t.Literal["discrete", "continuous"]

    @classmethod
    def from_dict_or_str(cls, data: dict | str) -> "ScoringConfig":
        # named scoring configs, like unit_scalar, binary, etc can be accessed by name
        if isinstance(data, str):
            if data in subtype_to_scoring_configs:
                return subtype_to_scoring_configs[data]()
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
    def from_yaml(cls, path: str) -> "ScoringConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict_or_str(config)


class DiscreteScoringConfig(ScoringConfig):
    name: str = "discrete"
    subtype: str = "discrete"
    type: t.Literal["discrete"] = "discrete"
    options: list[t.Any]


class BinaryScoringConfig(DiscreteScoringConfig):
    name: str = "binary"
    subtype: str = "binary"
    options: list[bool] = [True, False]


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

subtype_to_continuous_scoring_configs = {
    "continuous": ContinuousScoringConfig,
    "unit_scalar": UnitScalarScoringConfig,
}

subtype_to_scoring_configs = {
    **subtype_to_discrete_scoring_configs,
    **subtype_to_continuous_scoring_configs,
}


class ScoringConfigs(BaseConfig):
    scoring_configs: dict[str, ScoringConfig]

    @classmethod
    def from_dicts(cls, data: list[dict | str]) -> "ScoringConfigs":
        configs = [ScoringConfig.from_dict_or_str(item) for item in data]
        all_names = [config.name for config in configs]
        assert len(all_names) == len(set(all_names)), f"Duplicate scoring config names! {all_names}"
        return cls(scoring_configs={config.name: config for config in configs})

    @classmethod
    def from_yaml(cls, path: str) -> "ScoringConfigs":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            data = data["scoring_configs"]
        return cls.from_dicts(data)

    def get_config_by_name(self, name: str) -> ScoringConfig:
        return self.scoring_configs[name]

    def get_configs_by_subtype(self, subtype: str) -> list[ScoringConfig]:
        return [config for config in self.scoring_configs.values() if config.subtype == subtype]

    def get_configs_by_type(self, type: t.Literal["discrete", "continuous"]) -> list[ScoringConfig]:
        return [config for config in self.scoring_configs.values() if config.type == type]
