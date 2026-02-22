"""Fase 3: Agente de resposta final com Ollama local.

Este módulo recebe os documentos recuperados na Fase 2 e a pergunta do usuário,
constrói um prompt estrito e consulta o Ollama local com temperatura 0.0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import requests

SYSTEM_PROMPT_ESTRITO = (
    "Você é um especialista analítico. "
    "Responda APENAS com base no contexto fornecido. "
    "Se a resposta não estiver no contexto, diga EXATAMENTE: "
    "[Informação não encontrada no documento]. "
    "Não invente, não deduza."
)


class ErroOllama(RuntimeError):
    """Erro de integração com Ollama."""


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
) -> str:
    """Gera a resposta final usando Ollama local com temperatura fixa em 0.0."""

    if not pergunta or not pergunta.strip():
        raise ValueError("A pergunta do usuário não pode ser vazia.")

    contexto = montar_contexto(documentos)

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
            {"role": "system", "content": SYSTEM_PROMPT_ESTRITO},
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

    documentos = retriever.buscar(args.pergunta, top_k=args.top_k)

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
    )

    print("\n=== Resposta Final ===\n")
    print(resposta_final)


if __name__ == "__main__":
    main()
