import typing as t

from open_rubric.configs.base import BaseConfig


class ScoringConfig(BaseConfig):
    name: str
    type: t.Literal["discrete", "continuous"]

    @classmethod
    def from_yaml(cls, path: str) -> "ScoringConfig":
        breakpoint()

class DiscreteScoringConfig(ScoringConfig):
    options: list[t.Any]
    type: t.Literal["discrete"]

class BinaryScoringConfig(DiscreteScoringConfig):
    options: list[bool] = [True, False]
    name: str = "binary"

class CategoricalScoringConfig(DiscreteScoringConfig):
    options: list[str]
    name: str = "categorical"

class ContinuousScoringConfig(ScoringConfig):
    min: t.Optional[t.Union[int, float]]
    max: t.Optional[t.Union[int, float]]
    type: t.Literal["continuous"]
    name: str = "numerical"

class UnitScalarScoringConfig(ContinuousScoringConfig):
    min: 0
    max: 1
    name: str = "unit_scalar"


discrete_scoring_configs = [DiscreteScoringConfig, BinaryScoringConfig, CategoricalScoringConfig]
name_to_discrete_scoring_config = {config.name: config for config in discrete_scoring_configs}

continuous_scoring_configs = [ContinuousScoringConfig, UnitScalarScoringConfig]
name_to_continuous_scoring_config = {config.name: config for config in continuous_scoring_configs}

all_scoring_configs = discrete_scoring_configs + continuous_scoring_configs
name_to_scoring_config = {config.name: config for config in all_scoring_configs}
