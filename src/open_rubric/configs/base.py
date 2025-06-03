import typing as t
from typing import ClassVar, Generic, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """
    Base config type
    """

    @classmethod
    def from_data(cls, data: t.Any, **kwargs: t.Any) -> "BaseConfig":
        return cls(**data, **kwargs)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "BaseConfig":
        assert path.endswith(".yaml"), f"Path must end with .yaml; got {path}"
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    @classmethod
    def from_data_or_yaml(cls, data: t.Any | str, **kwargs: t.Any) -> "BaseConfig":
        if isinstance(data, str) and data.endswith(".yaml"):
            print(f"loading from yaml: {data}")
            return cls.from_yaml(data, **kwargs)
        else:
            return cls.from_data(data, **kwargs)


class BaseConfigCollector(BaseConfig, Generic[T]):
    """
    Base config "collector" type. Contains a dict of subconfigs; allows for parsing of recursive config types
    """
    configs: dict[str, T]
    BaseConfigType: ClassVar[type[T]]
    data_key: ClassVar[str] = "base_configs"
    preset_configs: ClassVar[list[T]] = []

    @classmethod
    def from_data(cls, data: list[dict | str] | dict, **kwargs: t.Any) -> "BaseConfigCollector[T]":
        if isinstance(data, dict):
            if cls.data_key in data:
                list_data: list[dict | str] = data[cls.data_key]
        else:
            list_data = data
        configs: list[T] = []
        for item in list_data:
            if isinstance(item, str) and item.endswith(
                ".yaml"
            ):  # recursive loading of scoring configs
                configs.extend(cls.from_yaml(item, **kwargs).configs.values())
            else:
                configs.append(cls.BaseConfigType.from_data(item, **kwargs))
        found_names = [config.name for config in configs]
        for present_config in cls.preset_configs:
            if present_config.name not in found_names:
                configs.append(present_config)
        return cls(configs={config.name: config for config in configs})

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "BaseConfigCollector[T]":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            data = data[cls.data_key]
        return cls.from_data(data, **kwargs)

    def get_config_by_name(self, name: str) -> T:
        if name not in self.configs:
            raise ValueError(
                f"Cannot find config with name {name}; got configs: {self.configs.keys()}"
            )
        return self.configs[name]
