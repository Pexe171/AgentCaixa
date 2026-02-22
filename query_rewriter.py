"""Módulo de reescrita de perguntas para melhorar recuperação no RAG."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

OLLAMA_URL: str = "http://localhost:11434/api/generate"
MODELO_REESCRITA: str = "llama3"
TIMEOUT_SEGUNDOS: int = 10
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_REESCRITA_PADRAO = "reescritor_tecnico.txt"
PROMPT_REESCRITA_FALLBACK: str = (
    "Você é um especialista em normas habitacionais da Caixa Econômica Federal. "
    "Reescreva a pergunta do usuário com termos técnicos bancários e habitacionais, "
    "preservando a intenção original. "
    "Responda APENAS com a pergunta reescrita, sem explicações."
)


def carregar_prompt(nome_arquivo: str) -> str:
    """Carrega um prompt da pasta `prompts/` com fallback seguro."""

    caminho_prompt = PROMPTS_DIR / nome_arquivo
    try:
        return caminho_prompt.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"Aviso: prompt não encontrado em '{caminho_prompt}'. Usando fallback de reescrita.")
        return PROMPT_REESCRITA_FALLBACK
    except OSError as erro:
        print(f"Aviso: falha ao ler prompt '{caminho_prompt}': {erro}. Usando fallback de reescrita.")
        return PROMPT_REESCRITA_FALLBACK


@lru_cache(maxsize=512)
def _expandir_pergunta_cached(pergunta_normalizada: str) -> str:
    """Executa a chamada ao Ollama e faz cache por pergunta normalizada."""

    payload: dict[str, Any] = {
        "model": MODELO_REESCRITA,
        "system": carregar_prompt(PROMPT_REESCRITA_PADRAO),
        "prompt": pergunta_normalizada,
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
        return pergunta_normalizada

    pergunta_expandida: str = str(corpo.get("response", "")).strip()
    return pergunta_expandida or pergunta_normalizada


def expandir_pergunta(pergunta_usuario: str) -> str:
    """Reescreve a pergunta do usuário com termos técnicos via Ollama.

    Em caso de timeout, indisponibilidade do serviço ou resposta inválida,
    retorna a pergunta original como fallback seguro.
    """
    pergunta_normalizada = pergunta_usuario.strip()
    if not pergunta_normalizada:
        return pergunta_usuario

    return _expandir_pergunta_cached(pergunta_normalizada)
