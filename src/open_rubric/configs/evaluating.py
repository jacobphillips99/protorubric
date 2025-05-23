import copy
import traceback
import typing as t

import litellm
import yaml
from pydantic import model_validator

from open_rubric.configs.aggregating import AggregatedQueryConfig
from open_rubric.configs.answers import AnyAnswerConfig
from open_rubric.configs.base import BaseConfig
from open_rubric.configs.query import QueryConfig
from open_rubric.models.model import MODEL
from open_rubric.models.model_types import ModelInput, ModelKwargs, ModelRequest


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
            elif data["type"] == "pass-through":
                return PassThroughEvaluatorConfig()
        raise ValueError(f"Invalid evaluator type: {data['type']}")

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "BaseEvaluatorConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if "evaluator_configs" in data:
            data = data["evaluator_configs"]
        return cls.from_data(data, **kwargs)

    async def async_call(
        self,
        query: QueryConfig,
        dependent_results: t.Optional[dict[str, AggregatedQueryConfig]] = None,
        **kwargs: t.Any,
    ) -> list[QueryConfig | AggregatedQueryConfig]:
        raise NotImplementedError(f"Evaluator {self.name} must implement async_call")


class PassThroughEvaluatorConfig(BaseEvaluatorConfig):
    """
    Special class to skip evaluation of the input query and just return the dependent results.
    Useful for collecting scores from other evaluators.
    """

    name: str = "pass-through"
    type: t.Literal["pass-through"] = "pass-through"

    async def async_call(
        self,
        query: QueryConfig,
        dependent_results: t.Optional[dict[str, AggregatedQueryConfig]] = None,
        **kwargs: t.Any,
    ) -> list[QueryConfig | AggregatedQueryConfig]:
        assert (
            dependent_results is not None
        ), f"PassThroughEvaluatorConfig requires dependent_results, but got {dependent_results}"
        return list(dependent_results.values())


class ModelEvaluatorConfig(BaseEvaluatorConfig):
    model: str
    provider: str
    n_samples: int = 1
    type: t.Literal["llm"] = "llm"

    @model_validator(mode="before")
    def prepare_data(cls, data: dict) -> dict:
        if data.get("n_samples") is None:
            data["n_samples"] = 1
        if "name" not in data:
            data["name"] = data["model"]
        if "provider" not in data:
            data["provider"] = litellm.get_llm_provider(data["model"])[1]
        return data

    @classmethod
    def from_data(cls, data: dict, **kwargs: t.Any) -> "ModelEvaluatorConfig":
        return cls(**data)

    async def async_call(
        self,
        query: QueryConfig,
        dependent_results: t.Optional[dict[str, AggregatedQueryConfig]] = None,
        **kwargs: t.Any,
    ) -> list[QueryConfig]:
        prompt = f"""
You are a helpful assistant that grades a conversation between a patient and a doctor.
The conversation is as follows:
-------- CONVERSATION --------
{query.inputs}
-------- END CONVERSATION --------

The rubric item to grade is as follows:
-------- RUBRIC ITEM --------
Does the assistant's response follow this rubric item? Rubric item: {query.instruction}
-------- END RUBRIC ITEM --------
{f"Example of grading rubric item: {query.example}" if query.example else ""}
{f"Here is other information to help you grade the rubric item: {str(dependent_results)}" if dependent_results else ""}
{query.scoring_config.to_prompt()}
""".strip()
        model_request = ModelRequest(
            model=self.model,
            provider=self.provider,
            model_input=ModelInput(prompt=prompt),
            model_kwargs=ModelKwargs(n_samples=self.n_samples),
            response_format=AnyAnswerConfig
        )
        response = await MODEL.agenerate(model_request)
        try:
            outputs = [query.scoring_config.parse_response(text) for text in response.texts]
        except Exception as e:
            breakpoint()
            raise ValueError(
                f"Error parsing response: {response.texts}, {e}, {traceback.format_exc()}; {response.texts}"
            )

        output_queries = [copy.deepcopy(query) for _ in outputs]
        for output, output_query in zip(outputs, output_queries):
            output_query._answer = output
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
        model_configs = []
        for model in raw_models:
            if isinstance(model, str):
                model_configs.append(
                    ModelEvaluatorConfig(name=model, model=model, n_samples=n_samples_per_model)
                )
            elif isinstance(model, dict):
                if n_samples_per_model is not None:
                    if "n_samples" in model:
                        print(
                            f"Overwriting n_samples for {model['model']} from {model['n_samples']} to {n_samples_per_model}"
                        )
                    model["n_samples"] = n_samples_per_model
                model_configs.append(ModelEvaluatorConfig(**model))
            else:
                raise ValueError(f"Invalid model type: {type(model)}; {model}")

        if isinstance(data, dict):
            name = data.get("name", "llm-ensemble")
        else:
            name = "llm-ensemble-" + "-".join(
                [model_config.model for model_config in model_configs]
            )
        return cls(
            models=model_configs,
            n_samples_per_model=n_samples_per_model,
            name=name,
        )

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "EnsembledModelEvaluatorConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_data(data, **kwargs)

    async def async_call(
        self,
        query: QueryConfig,
        dependent_results: t.Optional[dict[str, AggregatedQueryConfig]] = None,
        **kwargs: t.Any,
    ) -> list[QueryConfig]:
        results = []
        for model in self.models:
            results.extend(await model.async_call(query, dependent_results, **kwargs))
        return results


class EvaluatorConfigs(BaseConfig):
    evaluator_configs: dict[str, BaseEvaluatorConfig]

    @classmethod
    def from_data(cls, data: list[dict] | dict, **kwargs: t.Any) -> "EvaluatorConfigs":
        evaluator_configs: list[BaseEvaluatorConfig] = []
        for evaluator in data:
            if isinstance(evaluator, str) and evaluator.endswith(".yaml"):
                # recursive loading of evaluator_configs
                evaluator_configs.extend(
                    EvaluatorConfigs.from_yaml(evaluator).evaluator_configs.values()
                )
            else:
                evaluator_configs.append(BaseEvaluatorConfig.from_data(evaluator))
        return cls(evaluator_configs={evaluator.name: evaluator for evaluator in evaluator_configs})

    @classmethod
    def from_yaml(cls, path: str, **kwargs: t.Any) -> "EvaluatorConfigs":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if "evaluator_configs" in data:
            data = data["evaluator_configs"]
        return cls.from_data(data, **kwargs)

    def get_config_by_name(self, name: str) -> BaseEvaluatorConfig:
        return self.evaluator_configs[name]
