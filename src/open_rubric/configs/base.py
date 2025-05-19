import typing as t

import yaml
from pydantic import BaseModel


class BaseConfig(BaseModel):
    @classmethod
    def from_data(cls, data: dict[str, t.Any]) -> "BaseConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "BaseConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_data_or_yaml(cls, data: dict[str, t.Any] | str) -> "BaseConfig":
        if isinstance(data, str) and data.endswith(".yaml"):
            return cls.from_yaml(data)
        else:
            return cls.from_data(data)
    

