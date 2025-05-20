from open_rubric.configs.base import BaseConfig
import typing as t
from pydantic import model_validator

from open_rubric.configs.query import QueryConfig

class EvaluatorConfig(BaseConfig):
    name: str
    type: str

    def __call__(self, query: QueryConfig, dependent_results: t.Optional[dict[str, t.Any]] = None, **kwargs: t.Any) -> list[QueryConfig]:
        raise NotImplementedError(f"Evaluator {self.name} must implement __call__")
    
class ModelEvaluatorConfig(EvaluatorConfig):
    model: str
    n_samples: int = 1

    def __call__(self, query: QueryConfig, dependent_results: t.Optional[dict[str, t.Any]] = None, **kwargs: t.Any) -> list[QueryConfig]:
        pass

class EnsembledModelEvaluatorConfig(EvaluatorConfig):
    models: list[ModelEvaluatorConfig | str]
    n_samples_per_model: t.Optional[int] = None

    @model_validator(mode="after")
    def validate_models(self) -> "EnsembledModelEvaluatorConfig":
        # turn strings into ModelEvaluator objects and set global n_samples if provided
        output_models: list[ModelEvaluatorConfig] = []
        global_n_samples = self.n_samples_per_model if self.n_samples_per_model is not None else 1
        for model in self.models:
            if isinstance(model, str):
                output_models.append(ModelEvaluatorConfig(model=model, n_samples=global_n_samples))
            else:
                output_models.append(model)
        self.models = output_models
        return self









