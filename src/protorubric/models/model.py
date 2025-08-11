import asyncio
import logging
import os
import time
import traceback
import typing as t

import litellm
from llm_rate_limiter.rate_limit import rate_limiter

from protorubric.models.cache import ASYNC_REQUEST_CACHE
from protorubric.models.model_types import (
    ModelInput,
    ModelKwargs,
    ModelRequest,
    ModelResponse,
    construct_payload,
)

logger = logging.getLogger(__name__)


class Model:
    def __init__(self) -> None:
        self._check_api_keys()

    def _check_api_keys(self) -> None:
        """Check if API keys are set for the available providers."""
        for provider in rate_limiter.providers:
            if f"{provider.upper()}_API_KEY" not in os.environ:
                raise ValueError(f"API key for {provider} is not set!")

    async def _acheck_model_available(self, provider: str, model: str) -> ModelResponse | bool:
        model_request = ModelRequest(
            provider=provider,
            model=model,
            model_input=ModelInput(prompt="Hello, world!"),
            model_kwargs=ModelKwargs(max_tokens=5),
        )
        try:
            output = await self.agenerate(model_request, invalidate_cache=True)
            return output
        except Exception as e:
            logger.error(
                f"Error checking model availability for {model}: {str(e)}; {traceback.format_exc()}"
            )
            return False

    async def acheck_models_available(self) -> list[ModelResponse | bool]:
        provider_to_models = rate_limiter.providers_to_models
        payloads = [
            dict(provider=provider, model=model)
            for provider in rate_limiter.providers
            for model in provider_to_models[provider]
        ]
        return await asyncio.gather(
            *[self._acheck_model_available(**payload) for payload in payloads]
        )

    def check_models_available(self) -> list[ModelResponse | bool]:
        return asyncio.run(self.acheck_models_available())

    def _prepare_content(self, model_input: ModelInput) -> list[dict[str, t.Any]]:
        text_content = {"type": "text", "text": model_input.prompt}
        if model_input.images:
            image_content = [
                {"type": "image_url", "image_url": {"url": image.data}}
                for image in model_input.images
            ]
            return [text_content, *image_content]
        return [text_content]

    def _prepare_messages(self, model_request: ModelRequest) -> list[dict[str, t.Any]]:
        messages = []
        if model_request.system_prompt:
            messages.append({"role": "system", "content": model_request.system_prompt})

        if model_request.prepared_messages:
            for prepared_message in model_request.prepared_messages:
                content = self._prepare_content(prepared_message)
                prepared_message_dict: dict[str, t.Any] = {
                    "role": prepared_message.role,
                    "content": content,
                }
                messages.append(prepared_message_dict)

        if model_request.model_input:
            content = self._prepare_content(model_request.model_input)
            content.append(
                {"type": "text", "text": "Answer the above and begin your response below."}
            )
            message: dict[str, t.Any] = {"role": model_request.model_input.role, "content": content}
            messages.append(message)
        return messages

    def _get_estimated_tokens(self, model_request: ModelRequest) -> int:
        text_tokens = 0
        image_tokens = 0
        # TODO! make a better estimate. add estimate tokens to llm_rate_limiter package
        if model_request.model_input:
            text_tokens += len(model_request.model_input.prompt) // 2
            image_tokens += (
                len(model_request.model_input.images) * 500
                if model_request.model_input.images
                else 0
            )
        if model_request.prepared_messages:
            for prepared_message in model_request.prepared_messages:
                text_tokens += len(prepared_message.prompt) // 2
                image_tokens += len(prepared_message.images) * 500 if prepared_message.images else 0

        max_tokens = model_request.model_kwargs.max_tokens
        # assume half usage of max tokens
        return (text_tokens + image_tokens) + int(max_tokens / 2)

    async def agenerate(
        self, model_request: ModelRequest, invalidate_cache: bool = False
    ) -> ModelResponse:
        env_invalidate_cache = os.environ.get("protorubric_INVALIDATE_CACHE") == "True"
        if not invalidate_cache and not env_invalidate_cache:
            cached_response = await ASYNC_REQUEST_CACHE.aget(model_request)
            if cached_response:
                return cached_response

        provider = model_request.provider
        messages = self._prepare_messages(model_request)
        estimated_token_consumption = self._get_estimated_tokens(model_request)
        tic = time.time()
        await rate_limiter.wait_and_acquire(
            provider, model_request.model, tokens=estimated_token_consumption
        )
        try:
            # construct and await litellm request
            payload = construct_payload(model_request, messages)
            litellm_response = await litellm.acompletion(**payload)
            response_ms = time.time() - tic
            model_response: ModelResponse = ModelResponse.from_litellm_response(
                litellm_response, model_request, response_ms
            )
            # update the rate limiter with the actual token consumption
            token_usage = model_response.usage.get("total_tokens", estimated_token_consumption)
            rate_limiter.record_usage(provider, model_request.model, token_usage)
            await ASYNC_REQUEST_CACHE.aput(model_request, model_response)
            return model_response
        except Exception as e:
            logger.error(
                f"Error generating response for {model_request.model}: {str(e)}; {traceback.format_exc()}"
            )
            raise e

    def generate(self, model_request: ModelRequest) -> ModelResponse:
        return asyncio.run(self.agenerate(model_request))


MODEL = Model()
