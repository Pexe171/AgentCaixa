"""Gateway para provedores de LLM."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class LLMOutput:
    text: str
    provider: str
    model: str


class OpenAILLMGateway:
    """Cliente mínimo da API da OpenAI usando HTTP."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 20.0) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    def generate(self, system_prompt: str, user_prompt: str) -> LLMOutput:
        payload = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        answer_text = data.get("output_text", "").strip()
        if not answer_text:
            raise ValueError("Resposta da OpenAI sem conteúdo em output_text.")

        return LLMOutput(text=answer_text, provider="openai", model=self._model)


class OllamaLLMGateway:
    """Gateway para execução local via Ollama."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    def generate(self, system_prompt: str, user_prompt: str) -> LLMOutput:
        payload = {
            "model": self._model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
        }

        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post(
                f"{self._base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        answer_text = str(data.get("response", "")).strip()
        if not answer_text:
            raise ValueError("Resposta do Ollama sem conteúdo em response.")

        return LLMOutput(text=answer_text, provider="ollama", model=self._model)


class MockLLMGateway:
    """Fallback determinístico quando OpenAI/Ollama não estão configurados."""

    def generate(self, system_prompt: str, user_prompt: str) -> LLMOutput:
        del system_prompt
        text = (
            "[MODO MOCK] Estruturei uma resposta completa baseada no seu objetivo.\n\n"
            f"Resumo do pedido: {user_prompt[:260]}\n\n"
            "Próximos passos: validar requisitos, configurar dados, "
            "integrar API da OpenAI ou Ollama, testar qualidade e "
            "monitorar em produção."
        )
        return LLMOutput(text=text, provider="mock", model="mock-hag-v1")
