import typing as t

import yaml
from pydantic import BaseModel


class BaseConfig(BaseModel):
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
