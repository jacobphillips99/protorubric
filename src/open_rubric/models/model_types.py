import typing as t

import litellm
from pydantic import BaseModel, Field, model_validator
from litellm.types.utils import ModelResponse as LiteLLMResponse


class ImageInput(BaseModel):
    """Image input for VLM requests."""

    data: t.Union[str, bytes]  # Base64 encoded string or raw bytes
    mime_type: str = "image/png"


class ModelInput(BaseModel):
    prompt: str
    images: t.Optional[list[ImageInput]] = None

class ModelKwargs(BaseModel):
    n_samples: t.Optional[int] = None
    temperature: t.Optional[float] = None
    max_tokens: t.Optional[int] = 2**13 # 8192
    top_p: t.Optional[float] = None
    top_k: t.Optional[int] = None
    seed: t.Optional[int] = None

class ModelRequest(BaseModel):
    model: str
    model_input: ModelInput
    provider: str
    model_kwargs: ModelKwargs = Field(default_factory=ModelKwargs)
    timeout: float = 120.0
    extra_kwargs: t.Optional[dict[str, t.Any]] = None
    system_prompt: t.Optional[str] = None

    @model_validator(mode="before")
    def set_provider(cls, data: dict) -> dict:
        if not data.get("provider"):
            model = data['model']
            # result is (model, provider, key, endpoint)
            provider = litellm.get_llm_provider(model)[1] 
            data['provider'] = provider
        return data

class ModelResponse(BaseModel):
    texts: list[str]
    model: str
    response_ms: float
    provider: t.Optional[str] = None
    usage: dict[str, t.Any] = Field(default_factory=dict)
    raw_response: t.Optional[dict[str, t.Any]] = None
    stop_reason: t.Optional[str] = None

    @classmethod
    def from_litellm_response(cls, litellm_response: LiteLLMResponse, request: ModelRequest, response_ms: float) -> "ModelResponse":
        texts = [choice.message.content for choice in litellm_response.choices] # TODO -- multiple return choices?
        usage = litellm_response.usage.model_dump() if hasattr(litellm_response, "usage") else {}
        return cls(
            texts=texts,
            model=request.model,
            provider=request.provider,
            usage=usage,
            response_ms=response_ms,
            raw_response=litellm_response.model_dump() if hasattr(litellm_response, "model_dump") else None,
        )


def construct_payload(model_request: ModelRequest, messages: list[dict[str, t.Any]]) -> dict[str, t.Any]:
    # have to add model specific kwargs here 
    payload = {
        "model": model_request.model,
        "messages": messages
    }
    if model_request.model_kwargs:
        for k, v in model_request.model_kwargs.model_dump().items():
            if v is not None:
                if k == 'n_samples':
                    if model_request.provider.lower() in ["anthropic"]:
                        raise ValueError("Anthropic does not support n samples")
                    payload["n"] = v
                else:
                    payload[k] = v
    if model_request.extra_kwargs:
        for k, v in model_request.extra_kwargs.items():
            if v is not None:
                payload[k] = v

    # some models have specific kwargs! e.g. thinking models have max_completion_tokens and reject temperature
    if model_request.model in ["o4", "o3", "o4-mini", "o3-mini"]:
        payload["max_completion_tokens"] = model_request.model_kwargs.max_tokens
        payload["temperature"] = 1.0
        payload.pop("top_p", None)
    else:
        payload["max_tokens"] = model_request.model_kwargs.max_tokens
    if "gpt" in model_request.model:
        payload["max_tokens"] = min(payload["max_tokens"], 16384)
    elif "claude-3-5" in model_request.model:
        payload["max_tokens"] = min(payload["max_tokens"], 8192)

    return payload