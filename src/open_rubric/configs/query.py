import typing as t
import yaml
from open_rubric.configs.base import BaseConfig
from open_rubric.configs.scoring import ScoringConfig

class QueryConfig(BaseConfig):
    instruction: str
    inputs: t.Optional[list[str] | str] = None
    example: t.Optional[str] = None
    scoring_config: ScoringConfig
    dependency_names: t.Optional[list[str]] = None
    _score: t.Optional[t.Any] = None

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "QueryConfig":
        if "scoring_config" in data:
            data["scoring_config"] = kwargs["scoring_configs"].get_config_by_name(data["scoring_config"])
        return super().from_data(data, **kwargs)
    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "QueryConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)
