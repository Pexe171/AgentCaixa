"""Gateway para provedores de LLM."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class LLMOutput:
    text: str
    provider: str
    model: str


ToolExecutor = Callable[[str, dict[str, Any]], Any]


def _request_with_retries(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    max_attempts: int = 4,
    initial_backoff_s: float = 0.4,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ) as exc:
            should_retry = attempt < max_attempts
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                should_retry = should_retry and status in {429, 500, 502, 503, 504}
            if not should_retry:
                raise

            backoff_s = initial_backoff_s * (2 ** (attempt - 1))
            time.sleep(backoff_s)
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Falha inesperada de requisição HTTP.")


class OpenAILLMGateway:
    """Cliente mínimo da API da OpenAI usando HTTP."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 20.0) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout_s) as client:
            return _request_with_retries(
                client=client,
                url="https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
            )

    def _extract_tool_calls(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        raw_output = data.get("output", [])
        if not isinstance(raw_output, list):
            return []
        return [
            item
            for item in raw_output
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_executor: ToolExecutor,
    ) -> list[dict[str, str]]:
        tool_outputs: list[dict[str, str]] = []
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name", "")).strip()
            call_id = str(tool_call.get("call_id", "")).strip()
            if not tool_name or not call_id:
                continue

            raw_arguments = tool_call.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            tool_result = tool_executor(tool_name, arguments)
            serialized_output = (
                tool_result
                if isinstance(tool_result, str)
                else json.dumps(tool_result, ensure_ascii=False)
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": serialized_output,
                }
            )
        return tool_outputs

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> LLMOutput:
        payload: dict[str, Any] = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if tools:
            payload["tools"] = tools

        data = self._send_request(payload=payload)

        for _ in range(5):
            answer_text = str(data.get("output_text", "")).strip()
            if answer_text:
                return LLMOutput(text=answer_text, provider="openai", model=self._model)

            tool_calls = self._extract_tool_calls(data)
            if not tool_calls:
                break
            if tool_executor is None:
                raise ValueError(
                    "A resposta solicitou tool calls, mas nenhum "
                    "tool_executor foi fornecido."
                )

            tool_outputs = self._execute_tool_calls(
                tool_calls, tool_executor=tool_executor
            )
            if not tool_outputs:
                break

            next_payload: dict[str, Any] = {
                "model": self._model,
                "input": tool_outputs,
                "previous_response_id": data.get("id"),
            }
            if tools:
                next_payload["tools"] = tools
            data = self._send_request(payload=next_payload)

        raise ValueError("Resposta da OpenAI sem conteúdo em output_text.")


class OllamaLLMGateway:
    """Gateway para execução local via Ollama."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> LLMOutput:
        del tools, tool_executor
        payload = {
            "model": self._model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
        }

        with httpx.Client(timeout=self._timeout_s) as client:
            data = _request_with_retries(
                client=client,
                url=f"{self._base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                payload=payload,
            )

        answer_text = str(data.get("response", "")).strip()
        if not answer_text:
            raise ValueError("Resposta do Ollama sem conteúdo em response.")

        return LLMOutput(text=answer_text, provider="ollama", model=self._model)
