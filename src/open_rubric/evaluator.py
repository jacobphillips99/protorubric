import copy
import typing as t

import yaml
from pydantic import model_validator

from open_rubric.base import BaseConfig
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelKwargs, ModelRequest
from open_rubric.query import QueryConfig


class BaseEvaluatorConfig(BaseConfig):
    name: str
    type: str

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "BaseEvaluatorConfig":
        if "type" in data:
            if data["type"] == "llm":
                return ModelEvaluatorConfig.from_data(data, **kwargs)
            elif data["type"] == "llm-ensemble":
                return EnsembledModelEvaluatorConfig.from_data(data, **kwargs)
        raise ValueError(f"Invalid evaluator type: {data['type']}")

    def __call__(
        self,
        query: QueryConfig,
        dependent_results: t.Optional[dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> list[QueryConfig]:
        raise NotImplementedError(f"Evaluator {self.name} must implement __call__")


class ModelEvaluatorConfig(BaseEvaluatorConfig):
    model: str
    provider: t.Optional[str] = None
    n_samples: int = 1
    type: t.Literal["llm"] = "llm"

    @model_validator(mode="before")
    def set_n_samples(cls, data: dict) -> dict:
        if data.get("n_samples") is None:
            data["n_samples"] = 1
        if "name" not in data:
            data["name"] = data["model"]
        return data

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "ModelEvaluatorConfig":
        return cls(**data)

    def __call__(
        self,
        query: QueryConfig,
        dependent_results: t.Optional[dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> list[QueryConfig]:
        prompt = f"""
{query.instruction}
{query.example if query.example else ""}
{query.inputs if query.inputs else ""}
{'Dependent results: ' + str(dependent_results) if dependent_results else ""}
{'Scoring config: ' + query.scoring_config.to_prompt() if query.scoring_config else ""}
NOTE THIS IS JUST A TEST; PLEASE JUST RESPOND WITH A VALID ANSWER; DO NOT COMPLAIN OR EXPLAIN; JUST RESPOND WITH A VALID ANSWER
        """.strip()
        model_request = ModelRequest(
            model=self.model,
            provider=self.provider,
            model_input=ModelInput(prompt=prompt),
            model_kwargs=ModelKwargs(n_samples=self.n_samples),
        )
        response = MODEL.generate(model_request)
        outputs = [query.scoring_config.parse_response(text) for text in response.texts]
        output_queries = [copy.deepcopy(query) for _ in outputs]
        for output, output_query in zip(outputs, output_queries):
            output_query._score = output
        return output_queries


class EnsembledModelEvaluatorConfig(BaseEvaluatorConfig):
    models: list[ModelEvaluatorConfig]
    n_samples_per_model: t.Optional[int] = None
    type: t.Literal["llm-ensemble"] = "llm-ensemble"

    @classmethod
    def from_data(cls, data: list[dict] | dict, **kwargs: t.Any) -> "EnsembledModelEvaluatorConfig":
        if isinstance(data, dict) and "models" in data:
            raw_models = data.pop("models")
        elif isinstance(data, list):
            raw_models = data
        else:
            raise ValueError(f"Invalid data type: {type(data)}; {data}")
        if isinstance(data, dict):
            n_samples_per_model = data.get("n_samples_per_model", None) or kwargs.get(
                "n_samples_per_model", None
            )
        else:
            n_samples_per_model = None
        models = []
        for model in raw_models:
            if isinstance(model, str):
                models.append(
                    ModelEvaluatorConfig(name=model, model=model, n_samples=n_samples_per_model)
                )
            elif isinstance(model, dict):
                if n_samples_per_model is not None:
                    if "n_samples" in model:
                        print(
                            f"Overwriting n_samples for {model['model']} from {model['n_samples']} to {n_samples_per_model}"
                        )
                    model["n_samples"] = n_samples_per_model
                models.append(ModelEvaluatorConfig(**model))
            else:
                raise ValueError(f"Invalid model type: {type(model)}; {model}")

        if isinstance(data, dict):
            name = data.get("name", "llm-ensemble")
        else:
            name = "llm-ensemble-" + "-".join([model.model for model in models])
        return cls(
            models=models,
            n_samples_per_model=n_samples_per_model,
            name=name,
        )

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "EnsembledModelEvaluatorConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    def __call__(
        self,
        query: QueryConfig,
        dependent_results: t.Optional[dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> list[QueryConfig]:
        results = []
        for model in self.models:
            results.extend(model(query, dependent_results, **kwargs))
        return results


class EvaluatorConfigs(BaseConfig):
    evaluators: dict[str, BaseEvaluatorConfig]

    @classmethod
    def from_data(cls, data: list[dict] | dict, **kwargs: t.Any) -> "EvaluatorConfigs":
        evaluators = [BaseEvaluatorConfig.from_data(evaluator) for evaluator in data]
        return cls(evaluators={evaluator.name: evaluator for evaluator in evaluators})

    def get_config_by_name(self, name: str) -> BaseEvaluatorConfig:
        return self.evaluators[name]
