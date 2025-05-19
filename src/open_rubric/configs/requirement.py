from open_rubric.configs.base import BaseConfig
from open_rubric.configs.scoring import ScoringConfig, ScoringConfigs 
import typing as t
import yaml

class RequirementConfig(BaseConfig):
    name: str
    instruction: str
    example: t.Optional[str] = None
    scoring_config: ScoringConfig
    _score: t.Optional[t.Any] = None
    dependency_names: t.Optional[list[str]] = None


class Requirements(BaseConfig):
    requirements: dict[str, RequirementConfig]
    dependencies: dict[str, t.Optional[list[str]]]

    @classmethod
    def from_data(cls, data: list[dict] | dict, scoring_configs: ScoringConfigs) -> "Requirements":
        reqs = []
        for req in data:
            # replace string scoring_config with ScoringConfig object
            req['scoring_config'] = scoring_configs.get_config_by_name(req['scoring_config'])
            reqs.append(RequirementConfig.from_dict(req))
        all_names = [req.name for req in reqs]
        assert len(all_names) == len(set(all_names)), f"Duplicate requirement names! {all_names}"
        requirement_dict = {req.name: req for req in reqs}
        dependency_dict = {req.name: req.dependency_names for req in reqs if req.dependency_names is not None}
        return cls(requirements=requirement_dict, dependencies=dependency_dict)

    @classmethod
    def from_yaml(cls, path: str, scoring_configs: ScoringConfigs) -> "Requirements":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            data = data["requirements"]
        return cls.from_data(data, scoring_configs)

    def get_requirement_by_name(self, name: str) -> RequirementConfig:
        return self.requirements[name]

    def get_dependencies_by_name(self, name: str) -> list[str]:
        return self.dependencies[name]
    
    