"""Módulo de reescrita de perguntas para melhorar recuperação no RAG."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI
from openai import OpenAIError

OLLAMA_URL: str = "http://localhost:11434/api/generate"
MODELO_REESCRITA_LOCAL: str = "llama3"
MODELO_REESCRITA_OPENAI: str = "gpt-4o-mini"
TIMEOUT_SEGUNDOS: int = 10
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_REESCRITA_PADRAO = "reescritor_tecnico.txt"
PROMPT_REESCRITA_FALLBACK: str = (
    "Você é um especialista em normas habitacionais da Caixa Econômica Federal. "
    "Reescreva a pergunta do usuário com termos técnicos bancários e habitacionais, "
    "preservando a intenção original. "
    "Responda APENAS com a pergunta reescrita, sem explicações."
)


_dotenv_carregado = False


def _carregar_variaveis_ambiente() -> None:
    """Carrega variáveis do `.env` uma única vez por processo."""

    global _dotenv_carregado
    if not _dotenv_carregado:
        load_dotenv()
        _dotenv_carregado = True


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
def _expandir_pergunta_local_cached(pergunta_normalizada: str) -> str:
    """Executa a chamada ao Ollama e faz cache por pergunta normalizada."""

    payload: dict[str, Any] = {
        "model": MODELO_REESCRITA_LOCAL,
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


@lru_cache(maxsize=512)
def _expandir_pergunta_openai_cached(pergunta_normalizada: str, modelo: str) -> str:
    """Executa a reescrita com OpenAI e cache por pergunta+modelo."""

    _carregar_variaveis_ambiente()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return pergunta_normalizada

    cliente = OpenAI(api_key=api_key, timeout=TIMEOUT_SEGUNDOS)

    try:
        resposta = cliente.chat.completions.create(
            model=modelo,
            temperature=0.0,
            messages=[
                {"role": "system", "content": carregar_prompt(PROMPT_REESCRITA_PADRAO)},
                {"role": "user", "content": pergunta_normalizada},
            ],
        )
    except (APIConnectionError, APITimeoutError, OpenAIError):
        return pergunta_normalizada

    texto = (resposta.choices[0].message.content or "").strip() if resposta.choices else ""
    return texto or pergunta_normalizada


def expandir_pergunta(
    pergunta_usuario: str,
    provedor: str = "local",
    modelo_openai: str = MODELO_REESCRITA_OPENAI,
) -> str:
    """Reescreve a pergunta com termos técnicos usando provedor local ou OpenAI.

    Em caso de timeout, indisponibilidade do serviço ou resposta inválida,
    retorna a pergunta original como fallback seguro.
    """

    pergunta_normalizada = pergunta_usuario.strip()
    if not pergunta_normalizada:
        return pergunta_usuario

    if provedor == "openai":
        return _expandir_pergunta_openai_cached(pergunta_normalizada, modelo_openai)

    return _expandir_pergunta_local_cached(pergunta_normalizada)
