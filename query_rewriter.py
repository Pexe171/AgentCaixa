"""Módulo de reescrita de perguntas para melhorar recuperação no RAG."""

from __future__ import annotations

from typing import Any

import requests

OLLAMA_URL: str = "http://localhost:11434/api/generate"
MODELO_REESCRITA: str = "llama3"
TIMEOUT_SEGUNDOS: int = 10
SYSTEM_PROMPT_REESCRITA: str = (
    "Você é um especialista em normas habitacionais da Caixa Econômica Federal. "
    "O usuário fará uma pergunta coloquial. Sua tarefa é reescrever essa pergunta usando "
    "termos técnicos bancários (ex: trocar \"quem pode comprar\" por \"público-alvo, "
    "proponente, exigências e impedimentos\", trocar \"MCMV\" por \"PMCMV\", "
    "trocar \"renda\" por \"capacidade de pagamento e avaliação de risco\"). "
    "Responda APENAS com a pergunta reescrita, sem introduções ou explicações."
)


def expandir_pergunta(pergunta_usuario: str) -> str:
    """Reescreve a pergunta do usuário com termos técnicos via Ollama.

    Em caso de timeout, indisponibilidade do serviço ou resposta inválida,
    retorna a pergunta original como fallback seguro.
    """
    payload: dict[str, Any] = {
        "model": MODELO_REESCRITA,
        "system": SYSTEM_PROMPT_REESCRITA,
        "prompt": pergunta_usuario,
        "stream": False,
    }

    try:
        resposta: requests.Response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=TIMEOUT_SEGUNDOS,
        )
        resposta.raise_for_status()
        corpo: dict[str, Any] = resposta.json()
    except (requests.RequestException, ValueError):
        return pergunta_usuario

    pergunta_expandida: str = str(corpo.get("response", "")).strip()
    return pergunta_expandida or pergunta_usuario
