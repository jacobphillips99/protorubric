from pydantic import BaseModel
import typing as t
import yaml

class BaseConfig(BaseModel):

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "BaseConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "BaseConfig":
        with open(path, "r") as f:
            data = yaml.load(f)
        return cls.from_dict(data)
