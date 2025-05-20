from pydantic import BaseModel, Field
import typing as t
import os
from llm_rate_limiter.configs import RateLimitConfig
from llm_rate_limiter.rate_limit import rate_limiter
import logging 
import asyncio
import traceback

logger = logging.getLogger(__name__)


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
    max_tokens: t.Optional[int] = None
    top_p: t.Optional[float] = None
    top_k: t.Optional[int] = None
    seed: t.Optional[int] = None

class ModelRequest(BaseModel):
    model: str
    model_input: ModelInput
    provider: t.Optional[str] = None
    model_kwargs: t.Optional[ModelKwargs] = None
    timeout: t.Optional[float] = None
    extra_kwargs: t.Optional[dict[str, t.Any]] = None
    system_prompt: t.Optional[str] = None

class ModelResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: dict[str, t.Any] = Field(default_factory=dict)
    response_ms: int = 0
    raw_response: t.Optional[dict[str, t.Any]] = None
    stop_reason: t.Optional[str] = None

class Model:
    def __init__(self) -> None:
        self._check_api_keys()

    def _check_api_keys(self) -> None:
        """Check if API keys are set for the available providers."""
        for provider in rate_limiter.providers:
            if f"{provider.upper()}_API_KEY" not in os.environ:
                raise ValueError(f"API key for {provider} is not set!")    

    async def _acheck_model_available(self, model: str) -> bool:
        model_request = ModelRequest(model=model, model_input=ModelInput(prompt="Hello, world!"), model_kwargs=ModelKwargs(n_samples=1, max_tokens=1))
        try:
            output = await self.generate(model_request)
            return output
        except Exception as e:
            logger.error(
                f"Error checking model availability for {model}: {str(e)}; {traceback.format_exc()}"
            )
            return False

    def check_models_available(self):
        models = sum(list(rate_limiter.providers_to_models.values()), [])
        return asyncio.run(asyncio.gather(*[self._acheck_model_available(model) for model in models]))
    
    def _prepare_user_content(self, model_input: ModelInput) -> list[dict[str, t.Any]]:
        text_content = {"type": "text", "text": model_input.prompt}
        if model_input.images:
            image_content = [{"type": "image_url", "image_url": {"url": image.data}} for image in model_input.images]
            return [text_content, *image_content]
        return [text_content]

    
    def _prepare_messages(self, model_request: ModelRequest) -> list[dict[str, t.Any]]:
        messages = []
        if model_request.system_prompt:
            messages.append({"role": "system", "content": model_request.system_prompt})

        content = self._prepare_user_content(model_request.model_input)
        content.append({"type": "text", "text": "Begin your response below."})
        user_message: dict[str, t.Any] = {"role": "user", "content": content}
        messages.append(user_message)
        return messages
    
    def _get_estimated_tokens(self, model_request: ModelRequest) -> int:
        text_tokens = len(model_request.model_input.prompt) // 2
        # TODO! make a better estimate. add estimate tokens to llm_rate_limiter package
        image_tokens = len(model_request.model_input.images) * 500
        max_tokens = model_request.model_kwargs.max_tokens
        # assume half usage of max tokens
        return (text_tokens + image_tokens) + int(max_tokens / 2)
    
    async def agenerate(self, model_request: ModelRequest) -> t.Any:
        provider = model_request.provider
        messages = self._prepare_messages(model_request)
        estimated_consumption = self._get_estimated_tokens(model_request)
        await rate_limiter.wait_and_acquire(
            provider, model_request.model, tokens=estimated_consumption
        )
        try:
            response = await litellm.completion()

        return response

if __name__ == "__main__":
    model = Model()
    out = model.check_models_available()
    breakpoint()
    