"""Fase 3: Agente de resposta final com Ollama local.

Este módulo recebe os documentos recuperados na Fase 2 e a pergunta do usuário,
constrói um prompt estrito e consulta o Ollama local com temperatura 0.0.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI
from openai import OpenAIError

from query_rewriter import expandir_pergunta

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PADRAO_HABITACIONAL = "especialista_habitacional.txt"
PROMPT_FALLBACK_HABITACIONAL = (
    "Você é um especialista em crédito imobiliário habitacional da CAIXA. "
    "Responda APENAS com base no contexto fornecido. "
    "Se a resposta não estiver no contexto, diga EXATAMENTE: "
    "[Informação não encontrada no documento]. "
    "Não invente, não deduza."
)


class ErroOllama(RuntimeError):
    """Erro de integração com Ollama."""


class ErroOpenAI(RuntimeError):
    """Erro de integração com OpenAI."""


class ErroGemini(RuntimeError):
    """Erro de integração com Google Gemini."""


_dotenv_carregado = False


def _carregar_variaveis_ambiente() -> None:
    """Carrega variáveis do `.env` apenas uma vez por processo."""

    global _dotenv_carregado
    if not _dotenv_carregado:
        load_dotenv()
        _dotenv_carregado = True


def carregar_prompt(nome_arquivo: str) -> str:
    """Carrega um prompt da pasta `prompts/` com fallback resiliente."""

    caminho_prompt = PROMPTS_DIR / nome_arquivo
    try:
        return caminho_prompt.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"Aviso: prompt não encontrado em '{caminho_prompt}'. Usando fallback padrão.")
        return PROMPT_FALLBACK_HABITACIONAL
    except OSError as erro:
        print(f"Aviso: falha ao ler prompt '{caminho_prompt}': {erro}. Usando fallback padrão.")
        return PROMPT_FALLBACK_HABITACIONAL


def _normalizar_documento(documento: Any, indice: int) -> str:
    """Converte diferentes formatos de documento para texto contextual."""

    if hasattr(documento, "conteudo") and isinstance(getattr(documento, "conteudo"), str):
        return str(getattr(documento, "conteudo")).strip()

    if isinstance(documento, dict):
        conteudo = str(documento.get("conteudo", "")).strip()
        if conteudo:
            return conteudo

    if is_dataclass(documento):
        data = asdict(documento)
        conteudo = str(data.get("conteudo", "")).strip()
        if conteudo:
            return conteudo

    if isinstance(documento, str):
        return documento.strip()

    raise ValueError(f"Documento no índice {indice} possui formato inválido: {type(documento)!r}")


def montar_contexto(documentos: list[Any]) -> str:
    """Monta o bloco de contexto usado no prompt do modelo."""

    if not documentos:
        return ""

    trechos: list[str] = []
    for i, doc in enumerate(documentos, start=1):
        texto = _normalizar_documento(doc, i)
        if texto:
            trechos.append(f"[Trecho {i}]\n{texto}")

    return "\n\n".join(trechos)


def responder_com_ollama(
    documentos: list[Any],
    pergunta: str,
    modelo: str = "llama3",
    base_url: str = "http://localhost:11434",
    timeout_s: int = 600,
    prompt_sistema_arquivo: str = PROMPT_PADRAO_HABITACIONAL,
) -> str:
    """Gera a resposta final usando Ollama local com temperatura fixa em 0.0."""

    if not pergunta or not pergunta.strip():
        raise ValueError("A pergunta do usuário não pode ser vazia.")

    contexto = montar_contexto(documentos)
    prompt_sistema = carregar_prompt(prompt_sistema_arquivo)

    mensagem_usuario = (
        "Contexto:\n"
        f"{contexto if contexto else '[Sem contexto recuperado]'}\n\n"
        "Pergunta do usuário:\n"
        f"{pergunta.strip()}"
    )

    payload = {
        "model": modelo,
        "stream": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": mensagem_usuario},
        ],
    }

    try:
        resposta = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_s,
        )
        resposta.raise_for_status()
    except requests.RequestException as exc:
        raise ErroOllama(f"Falha ao consultar Ollama em {base_url}: {exc}") from exc

    corpo = resposta.json()
    resposta_modelo = (corpo.get("message") or {}).get("content", "")
    if not resposta_modelo.strip():
        raise ErroOllama("Ollama retornou uma resposta vazia.")

    return resposta_modelo.strip()


def responder_com_openai(
    documentos: list[Any],
    pergunta: str,
    modelo: str = "gpt-4o-mini",
    timeout_s: int = 60,
    prompt_sistema_arquivo: str = PROMPT_PADRAO_HABITACIONAL,
) -> str:
    """Gera a resposta final usando OpenAI com prompt externo da pasta `prompts/`."""

    if not pergunta or not pergunta.strip():
        raise ValueError("A pergunta do usuário não pode ser vazia.")

    _carregar_variaveis_ambiente()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ErroOpenAI("OPENAI_API_KEY não encontrada. Configure a chave no ambiente ou no arquivo .env.")

    contexto = montar_contexto(documentos)
    prompt_sistema = carregar_prompt(prompt_sistema_arquivo)

    mensagem_usuario = (
        "Contexto:\n"
        f"{contexto if contexto else '[Sem contexto recuperado]'}\n\n"
        "Pergunta do usuário:\n"
        f"{pergunta.strip()}"
    )

    cliente = OpenAI(api_key=api_key, timeout=timeout_s)

    try:
        resposta = cliente.chat.completions.create(
            model=modelo,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": mensagem_usuario},
            ],
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise ErroOpenAI(f"Falha de conexão/timeout com OpenAI: {exc}") from exc
    except OpenAIError as exc:
        raise ErroOpenAI(f"Falha ao consultar OpenAI: {exc}") from exc

    conteudo = (resposta.choices[0].message.content or "").strip() if resposta.choices else ""
    if not conteudo:
        raise ErroOpenAI("OpenAI retornou uma resposta vazia.")

    return conteudo


def _montar_mensagem_usuario(contexto: str, pergunta: str) -> str:
    """Monta o bloco de entrada padrão para os provedores de geração."""

    return (
        "Contexto:\n"
        f"{contexto if contexto else '[Sem contexto recuperado]'}\n\n"
        "Pergunta do usuário:\n"
        f"{pergunta.strip()}"
    )


def responder_com_gemini(
    documentos: list[Any],
    pergunta: str,
    modelo: str = "gemini-1.5-flash",
    timeout_s: int = 60,
    prompt_sistema_arquivo: str = PROMPT_PADRAO_HABITACIONAL,
) -> str:
    """Gera a resposta final usando Google Gemini com temperatura fixa em 0.0."""

    if not pergunta or not pergunta.strip():
        raise ValueError("A pergunta do usuário não pode ser vazia.")

    _carregar_variaveis_ambiente()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ErroGemini("GOOGLE_API_KEY não encontrada. Configure a chave no ambiente ou no arquivo .env.")

    contexto = montar_contexto(documentos)
    prompt_sistema = carregar_prompt(prompt_sistema_arquivo)
    mensagem_usuario = _montar_mensagem_usuario(contexto, pergunta)

    try:
        genai.configure(api_key=api_key)
        modelo_gemini = genai.GenerativeModel(
            model_name=modelo,
            system_instruction=prompt_sistema,
            generation_config={"temperature": 0.0},
        )
        resposta = modelo_gemini.generate_content(
            mensagem_usuario,
            request_options={"timeout": timeout_s},
        )
    except Exception as exc:  # noqa: BLE001
        raise ErroGemini(f"Falha ao consultar Gemini: {exc}") from exc

    conteudo = (getattr(resposta, "text", "") or "").strip()
    if not conteudo:
        raise ErroGemini("Gemini retornou uma resposta vazia.")

    return conteudo


def gerar_resposta_hibrida(
    provedor: str,
    documentos: list[Any],
    pergunta: str,
    modelo: str,
    ollama_url: str = "http://localhost:11434",
    timeout_s: int = 60,
    prompt_sistema_arquivo: str = PROMPT_PADRAO_HABITACIONAL,
) -> str:
    """Direciona a geração de resposta para Ollama, OpenAI ou Gemini."""

    provedor_normalizado = provedor.strip().lower()

    if provedor_normalizado in {"local", "ollama"}:
        return responder_com_ollama(
            documentos=documentos,
            pergunta=pergunta,
            modelo=modelo,
            base_url=ollama_url,
            timeout_s=timeout_s,
            prompt_sistema_arquivo=prompt_sistema_arquivo,
        )

    if provedor_normalizado == "openai":
        return responder_com_openai(
            documentos=documentos,
            pergunta=pergunta,
            modelo=modelo,
            timeout_s=timeout_s,
            prompt_sistema_arquivo=prompt_sistema_arquivo,
        )

    if provedor_normalizado == "gemini":
        return responder_com_gemini(
            documentos=documentos,
            pergunta=pergunta,
            modelo=modelo,
            timeout_s=timeout_s,
            prompt_sistema_arquivo=prompt_sistema_arquivo,
        )

    raise ValueError("Provedor inválido. Use: ollama/local, openai ou gemini.")


def parsear_argumentos() -> argparse.Namespace:
    """CLI para executar a Fase 3 usando índice híbrido da Fase 2."""

    parser = argparse.ArgumentParser(description="Fase 3: geração de resposta final com Ollama")
    parser.add_argument("--pergunta", type=str, required=True, help="Pergunta do usuário")
    parser.add_argument("--chunks-json", type=Path, help="JSON de chunks da Fase 1 para indexar")
    parser.add_argument("--top-k", type=int, default=4, help="Quantidade de trechos recuperados")
    parser.add_argument("--modelo-llm", type=str, default="llama3", help="Modelo do Ollama para resposta")
    parser.add_argument(
        "--modelo-embedding",
        type=str,
        default="nomic-embed-text",
        help="Modelo de embedding Ollama da Fase 2",
    )
    parser.add_argument("--chroma-dir", type=str, default="./chroma_db", help="Diretório do ChromaDB")
    parser.add_argument("--collection", type=str, default="documentos", help="Coleção do ChromaDB")
    parser.add_argument(
        "--lote-indexacao",
        type=int,
        default=50,
        help="Quantidade de chunks por lote durante indexação no ChromaDB",
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL base do Ollama")
    parser.add_argument("--limpar", action="store_true", help="Limpa coleção antes de indexar chunks")
    parser.add_argument(
        "--prompt-sistema",
        type=str,
        default=PROMPT_PADRAO_HABITACIONAL,
        help="Arquivo de prompt em prompts/ para instruções do especialista",
    )
    parser.add_argument(
        "--salvar-contexto",
        type=Path,
        default=None,
        help="Salva os documentos recuperados em JSON para auditoria",
    )
    return parser.parse_args()


def main() -> None:
    """Pipeline CLI: recuperar contexto (Fase 2) e gerar resposta (Fase 3)."""

    args = parsear_argumentos()

    from retriever import HybridRetriever

    retriever = HybridRetriever(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        ollama_model=args.modelo_embedding,
        lote_indexacao=args.lote_indexacao,
    )

    if args.chunks_json:
        retriever.carregar_chunks_do_json(args.chunks_json, limpar_colecao=args.limpar)
        print(f"Chunks indexados com sucesso na coleção '{args.collection}'.")

    pergunta_tecnica = expandir_pergunta(args.pergunta)
    documentos = retriever.buscar(pergunta_tecnica, top_k=args.top_k)

    if args.salvar_contexto is not None:
        serializado = [asdict(item) for item in documentos]
        args.salvar_contexto.write_text(
            json.dumps({"total": len(serializado), "documentos": serializado}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Contexto recuperado salvo em: {args.salvar_contexto}")

    resposta_final = responder_com_ollama(
        documentos=documentos,
        pergunta=args.pergunta,
        modelo=args.modelo_llm,
        base_url=args.ollama_url,
        prompt_sistema_arquivo=args.prompt_sistema,
    )

    print("\n=== Resposta Final ===\n")
    print(resposta_final)


if __name__ == "__main__":
    main()
